// vtt: real-time target tracker with simulated gimbal (webcam, video file or headless).

#include "vtt/cli.hpp"
#include "vtt/csv.hpp"
#include "vtt/fusion.hpp"
#include "vtt/gimbal.hpp"
#include "vtt/metrics.hpp"
#include "vtt/overlay.hpp"
#include "vtt/selection.hpp"
#include "vtt/tracker.hpp"
#include "vtt/udp.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* kWindow = "vision-target-tracker";
constexpr int kKeyEscape = 27;
constexpr int kExitThreshold = 2;
const cv::Scalar kRubberBand(255, 0, 255);
const cv::Scalar kDetection(255, 255, 0);
constexpr int kDetectorMaxWidth = 640;
constexpr std::size_t kMaxDatagram = 60000;
constexpr long kShowDetectionFrames = 10;

/// JPEG-encode @p frame (at most 640 px wide) behind a VTF1 header and send it.
bool send_frame(vtt::UdpSocket& socket, const vtt::Endpoint& to, const cv::Mat& frame, long frame_id) {
    cv::Mat small = frame;
    if (frame.cols > kDetectorMaxWidth) {
        const double s = static_cast<double>(kDetectorMaxWidth) / static_cast<double>(frame.cols);
        cv::resize(frame, small, cv::Size(), s, s, cv::INTER_AREA);
    }
    std::vector<std::uint8_t> jpeg;
    for (int quality = 80; quality >= 30; quality -= 10) {
        if (!cv::imencode(".jpg", small, jpeg, {cv::IMWRITE_JPEG_QUALITY, quality})) {
            return false;
        }
        if (jpeg.size() + vtt::kFrameHeaderSize <= kMaxDatagram) {
            break;
        }
    }
    if (jpeg.size() + vtt::kFrameHeaderSize > kMaxDatagram) {
        return false;
    }
    vtt::FrameHeader header;
    header.frame_id = static_cast<std::uint32_t>(frame_id);
    header.width = static_cast<std::uint16_t>(frame.cols);
    header.height = static_cast<std::uint16_t>(frame.rows);
    std::vector<std::uint8_t> datagram = vtt::encode_frame_header(header);
    datagram.insert(datagram.end(), jpeg.begin(), jpeg.end());
    return socket.send_to(to.host, to.port, datagram);
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

vtt::TrackerConfig make_config(const vtt::AppOptions& options, vtt::Algorithm algorithm) {
    vtt::TrackerConfig config = vtt::default_tracker_config(algorithm);
    if (options.disable_scale) {
        config.estimate_scale = false;
    }
    if (options.psr_threshold) {
        config.psr_track = *options.psr_threshold;
        config.psr_update = std::max(config.psr_update, *options.psr_threshold);
        config.psr_reacquire = std::max(config.psr_reacquire, *options.psr_threshold);
    }
    return config;
}

vtt::Algorithm other(vtt::Algorithm algorithm) {
    return algorithm == vtt::Algorithm::Kcf ? vtt::Algorithm::Mosse : vtt::Algorithm::Kcf;
}

std::optional<vtt::MouseAction> map_mouse_event(int event) {
    switch (event) {
        case cv::EVENT_MOUSEMOVE: return vtt::MouseAction::Move;
        case cv::EVENT_LBUTTONDOWN: return vtt::MouseAction::LeftDown;
        case cv::EVENT_LBUTTONUP: return vtt::MouseAction::LeftUp;
        case cv::EVENT_RBUTTONDOWN: return vtt::MouseAction::RightDown;
        default: return std::nullopt;
    }
}

struct PointerContext {
    vtt::RoiSelector* selector = nullptr;
    cv::Size frame_size;
    std::vector<vtt::SelectionEvent> events;
};

/// Clears the window callback before the context it points at is destroyed.
struct MouseCallbackGuard {
    bool armed = false;

    void arm(cv::MouseCallback callback, void* userdata) {
        cv::setMouseCallback(kWindow, callback, userdata);
        armed = true;
    }

    ~MouseCallbackGuard() {
        if (armed) {
            cv::setMouseCallback(kWindow, nullptr, nullptr);
        }
    }

    MouseCallbackGuard() = default;
    MouseCallbackGuard(const MouseCallbackGuard&) = delete;
    MouseCallbackGuard& operator=(const MouseCallbackGuard&) = delete;
};

// The gimbal inset is painted onto the camera frame; draw_overlay does not change
// the canvas size. OpenCV reports positions in that canvas, so they are frame pixels.
// Presses that miss the frame are ignored. Once a drag is active, moves and the
// left-button release clamp onto the frame so the gesture can finish.
void on_mouse(int event, int x, int y, int /*flags*/, void* userdata) {
    auto* pointer = static_cast<PointerContext*>(userdata);
    if (pointer == nullptr || pointer->selector == nullptr) {
        return;
    }
    const std::optional<vtt::MouseAction> action = map_mouse_event(event);
    if (!action) {
        return;
    }
    const cv::Size frame = pointer->frame_size;
    if (frame.width <= 0 || frame.height <= 0) {
        return;
    }
    const bool outside = x < 0 || y < 0 || x >= frame.width || y >= frame.height;
    if (outside) {
        const bool finishing_drag = pointer->selector->rubber_band().has_value() &&
                                    (*action == vtt::MouseAction::Move || *action == vtt::MouseAction::LeftUp);
        if (!finishing_drag) {
            return;
        }
        x = std::clamp(x, 0, frame.width - 1);
        y = std::clamp(y, 0, frame.height - 1);
    }
    if (const std::optional<vtt::SelectionEvent> selection =
            pointer->selector->on_mouse(*action, x, y, frame)) {
        pointer->events.push_back(*selection);
    }
}

void draw_rubber_band(cv::Mat& canvas, const cv::Rect2d& band) {
    const cv::Rect rect(cvRound(band.x), cvRound(band.y), cvRound(band.width), cvRound(band.height));
    cv::rectangle(canvas, rect, kRubberBand, 2, cv::LINE_AA);
}

int run(const vtt::AppOptions& options) {
    cv::VideoCapture capture;
    if (options.input) {
        capture.open(*options.input);
    } else {
        capture.open(options.camera_index);
    }
    if (!capture.isOpened()) {
        std::cerr << "error: cannot open " << (options.input ? *options.input : "camera") << '\n';
        return 1;
    }
    cv::Mat frame;
    if (!capture.read(frame) || frame.empty()) {
        std::cerr << "error: no frames in input\n";
        return 1;
    }

    std::vector<std::optional<cv::Rect2d>> truth;
    if (options.ground_truth) {
        for (const vtt::GroundTruthRow& row : vtt::load_ground_truth(*options.ground_truth)) {
            if (row.frame < 0) {
                continue;
            }
            const auto index = static_cast<std::size_t>(row.frame);
            if (truth.size() <= index) {
                truth.resize(index + 1U);
            }
            truth[index] = row.box;
        }
    }
    const auto truth_at = [&truth](long index) -> std::optional<cv::Rect2d> {
        const auto i = static_cast<std::size_t>(index);
        return i < truth.size() ? truth[i] : std::nullopt;
    };

    const bool gui = !options.headless;
    std::optional<cv::Rect2d> roi = options.roi ? options.roi : truth_at(0);
    if (!roi && !gui) {
        std::cerr << "error: no target selected\n";
        return 1;
    }

    vtt::Algorithm algorithm = options.algorithm;
    vtt::Tracker tracker(make_config(options, algorithm));
    vtt::TrackOutput track;
    bool target_locked = false;
    if (roi) {
        tracker.init(frame, *roi);
        target_locked = true;
        track.state = vtt::TrackState::Tracking;
        track.box = *roi;
        track.predicted = (roi->tl() + roi->br()) * 0.5;
    }
    vtt::GimbalSim gimbal(options.gimbal, frame.size());
    const cv::Size frame_size = frame.size();

    const bool from_file = options.input.has_value();
    double source_fps = capture.get(cv::CAP_PROP_FPS);
    if (std::isnan(source_fps) || source_fps <= 1.0 || source_fps >= 1000.0) {
        source_fps = 30.0;
    }

    std::optional<vtt::CsvWriter> csv;
    if (options.csv_path) {
        csv.emplace(*options.csv_path);
    }
    cv::VideoWriter writer;
    if (options.output_video) {
        writer.open(*options.output_video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), source_fps,
                    frame.size());
        if (!writer.isOpened()) {
            std::cerr << "error: cannot write " << *options.output_video << '\n';
            return 1;
        }
    }

    vtt::LatencyStats latency;
    vtt::AccuracyStats accuracy;
    vtt::RateMeter loop_rate;
    vtt::OverlayOptions overlay;
    bool paused = false;
    vtt::RoiSelector selector(options.click_box);
    PointerContext pointer;
    pointer.selector = &selector;
    pointer.frame_size = frame.size();
    MouseCallbackGuard mouse_guard;
    if (gui) {
        cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
        mouse_guard.arm(on_mouse, &pointer);
    }

    // Detect + track: frames go to an external detector, its replies correct the tracker.
    std::optional<vtt::Endpoint> detector_at;
    std::optional<vtt::UdpSocket> detector_tx;
    std::optional<vtt::UdpSocket> detector_rx;
    if (options.detector) {
        detector_at = vtt::parse_endpoint(*options.detector);
        detector_tx.emplace(static_cast<std::uint16_t>(0));
        detector_rx.emplace(static_cast<std::uint16_t>(options.detector_listen));
    }
    vtt::FusionConfig fusion_config;
    fusion_config.min_confidence = options.detector_confidence;
    fusion_config.auto_lock = options.auto_lock;
    vtt::DetectionFusion fusion(fusion_config);
    std::optional<vtt::Detection> last_detection;
    long reanchors = 0;
    long refines = 0;
    long detections_in_window = 0;
    double detection_rate = 0.0;
    Clock::time_point detection_window = Clock::now();

    long index = 0;
    const Clock::time_point run_start = Clock::now();
    Clock::time_point last_loop = run_start;
    double last_latency = 0.0;
    bool quit = false;

    const auto release_target = [&]() {
        target_locked = false;
        tracker.reset();
        track = vtt::TrackOutput{};
        last_latency = 0.0;
        selector.cancel_drag();
        fusion.on_target_changed(false);
    };
    const auto lock_target = [&](const cv::Rect2d& box, bool from_detector = false) {
        if (box.width < 2.0 || box.height < 2.0) {
            return;
        }
        tracker.init(frame, box);
        target_locked = true;
        track = vtt::TrackOutput{};
        track.state = vtt::TrackState::Tracking;
        track.box = box;
        track.predicted = (box.tl() + box.br()) * 0.5;
        fusion.on_target_changed(true, from_detector);
    };
    const auto apply_events = [&]() {
        for (const vtt::SelectionEvent& event : pointer.events) {
            if (std::holds_alternative<vtt::Release>(event)) {
                release_target();
            } else {
                lock_target(std::get<vtt::Lock>(event).box);
            }
        }
        pointer.events.clear();
    };
    const auto handle_key = [&](int key) {
        key &= 0xFF;
        if (key == 'q' || key == kKeyEscape) {
            quit = true;
        } else if (key == ' ') {
            paused = !paused;
        } else if (key == 'h') {
            overlay.show_help = !overlay.show_help;
        } else if (key == 'g') {
            overlay.show_gimbal_view = !overlay.show_gimbal_view;
        } else if (key == 'c') {
            gimbal.reset();
        } else if (key == 'a') {
            algorithm = other(algorithm);
            tracker = vtt::Tracker(make_config(options, algorithm));
            if (target_locked) {
                tracker.init(frame, track.box);
            }
        } else if (key == 'x') {
            release_target();
        }
    };
    const auto compose = [&](bool is_paused) {
        cv::Mat canvas = frame.clone();
        const vtt::Telemetry telemetry{algorithm, index, loop_rate.rate(), last_latency, is_paused};
        vtt::draw_overlay(canvas, track, gimbal, telemetry, overlay);
        if (gui) {
            if (const std::optional<cv::Rect2d> band = selector.rubber_band()) {
                draw_rubber_band(canvas, *band);
            }
        }
        if (detector_tx) {
            if (last_detection && last_detection->box &&
                index - last_detection->frame_id <= kShowDetectionFrames) {
                const cv::Rect2d& b = *last_detection->box;
                cv::rectangle(canvas, cv::Rect(cvRound(b.x), cvRound(b.y), cvRound(b.width), cvRound(b.height)),
                              kDetection, 1, cv::LINE_AA);
                const std::string label = cv::format("YOLO %.2f", last_detection->confidence);
                cv::putText(canvas, label, cv::Point(cvRound(b.x), std::max(12, cvRound(b.y) - 4)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, kDetection, 1, cv::LINE_AA);
            }
            const std::string status =
                cv::format("DET on  %.1f/s  refine %ld  re-anchor %ld%s", detection_rate, refines, reanchors,
                           fusion.detector_backed() ? "  [backed]" : "");
            cv::putText(canvas, status, cv::Point(10, canvas.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(canvas, status, cv::Point(10, canvas.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        kDetection, 1, cv::LINE_AA);
        }
        return canvas;
    };

    while (!quit) {
        pointer.frame_size = frame.size();
        if (target_locked && index > 0) {
            const Clock::time_point t0 = Clock::now();
            track = tracker.update(frame);
            last_latency = elapsed_ms(t0, Clock::now());
            latency.add(last_latency);
        }
        if (detector_tx && detector_rx && detector_at) {
            fusion.record(index, target_locked ? track.state : vtt::TrackState::Idle, track.box);
            if (index % options.detect_every == 0) {
                static_cast<void>(send_frame(*detector_tx, *detector_at, frame, index));
            }
            while (const auto bytes = detector_rx->try_receive()) {
                const auto detection = vtt::parse_detection_line(std::string(bytes->begin(), bytes->end()));
                if (!detection) {
                    continue;
                }
                ++detections_in_window;
                last_detection = *detection;
                const vtt::FusionDecision decision = fusion.on_detection(*detection, index);
                if (decision.action != vtt::FusionAction::None) {
                    lock_target(decision.box, true);
                    if (decision.action == vtt::FusionAction::Reanchor) {
                        ++reanchors;
                    } else if (decision.action == vtt::FusionAction::Refine) {
                        ++refines;
                    }
                }
            }
            const double window_s = elapsed_ms(detection_window, Clock::now()) / 1000.0;
            if (window_s >= 1.0) {
                detection_rate = static_cast<double>(detections_in_window) / window_s;
                detections_in_window = 0;
                detection_window = Clock::now();
            }
        }
        const Clock::time_point now = Clock::now();
        const double wall_dt = elapsed_ms(last_loop, now) / 1000.0;
        last_loop = now;
        if (index > 0) {
            loop_rate.tick(wall_dt);
        }
        const double dt = from_file ? 1.0 / source_fps : std::max(wall_dt, 1e-3);
        const bool has_target = track.state == vtt::TrackState::Tracking ||
                                track.state == vtt::TrackState::Occluded;
        const std::optional<cv::Point2d> aim =
            has_target ? std::optional<cv::Point2d>((track.box.tl() + track.box.br()) * 0.5)
                       : std::nullopt;
        const vtt::GimbalState gimbal_state = gimbal.step(aim, dt);

        const std::optional<cv::Rect2d> gt = truth_at(index);
        if (target_locked && index > 0 && gt) {
            accuracy.add(track.box, *gt);
        }
        if (csv) {
            csv->write(index, last_latency, track, gimbal_state, gt);
        }

        if (gui || writer.isOpened()) {
            cv::Mat canvas = compose(false);
            if (writer.isOpened()) {
                writer.write(canvas);
            }
            if (gui) {
                cv::imshow(kWindow, canvas);
                int key = cv::waitKey(1);
                while (true) {
                    apply_events();
                    handle_key(key);
                    if (quit || !paused) {
                        break;
                    }
                    pointer.frame_size = frame.size();
                    canvas = compose(true);
                    cv::imshow(kWindow, canvas);
                    key = cv::waitKey(30);
                }
                last_loop = Clock::now();
            }
        }

        ++index;
        if (options.max_frames > 0 && index >= options.max_frames) {
            break;
        }
        if (!quit && (!capture.read(frame) || frame.empty())) {
            break;
        }
    }
    const double wall_s = elapsed_ms(run_start, Clock::now()) / 1000.0;

    std::cout << std::fixed << std::setprecision(3) << "vtt " << VTT_VERSION << " | "
              << vtt::to_string(algorithm) << " | " << index << " frames | " << frame_size.width
              << 'x' << frame_size.height << " | "
              << (options.input ? *options.input : std::string("camera")) << '\n';
    if (latency.count() > 0) {
        std::cout << "latency ms  mean " << latency.mean() << "  p50 " << latency.percentile(50.0)
                  << "  p95 " << latency.percentile(95.0) << "  max " << latency.max() << '\n'
                  << std::setprecision(1) << "throughput  tracker " << 1000.0 / latency.mean()
                  << " fps  pipeline " << static_cast<double>(index) / std::max(wall_s, 1e-9)
                  << " fps\n";
    }
    int exit_code = 0;
    if (accuracy.count() > 0) {
        std::cout << std::setprecision(3) << "accuracy    mean IoU " << accuracy.mean_iou()
                  << "  centre error " << std::setprecision(2) << accuracy.mean_center_error()
                  << " px  success@0.5 " << std::setprecision(3) << accuracy.success_rate()
                  << "  precision@20px " << accuracy.precision_20px() << '\n';
        if (options.min_iou && accuracy.mean_iou() < *options.min_iou) {
            std::cerr << "FAIL: mean IoU " << accuracy.mean_iou() << " < " << *options.min_iou << '\n';
            exit_code = kExitThreshold;
        }
        if (options.max_center_error && accuracy.mean_center_error() > *options.max_center_error) {
            std::cerr << "FAIL: mean centre error " << accuracy.mean_center_error() << " > "
                      << *options.max_center_error << " px\n";
            exit_code = kExitThreshold;
        }
    } else if (options.min_iou || options.max_center_error) {
        std::cerr << "FAIL: accuracy thresholds given but no ground truth matched\n";
        exit_code = kExitThreshold;
    }
    return exit_code;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::vector<std::string> args(argv + 1, argv + argc);
        const vtt::ParseResult parsed = vtt::parse_arguments(args);
        if (!parsed.ok()) {
            std::cerr << "error: " << parsed.error << "\n\n" << vtt::usage();
            return 1;
        }
        if (parsed.options.show_help) {
            std::cout << vtt::usage();
            return 0;
        }
        if (parsed.options.show_version) {
            std::cout << "vtt " << VTT_VERSION << " (OpenCV " << CV_VERSION << ")\n";
            return 0;
        }
        return run(parsed.options);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
