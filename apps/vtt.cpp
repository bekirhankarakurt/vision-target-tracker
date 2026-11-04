// vtt: real-time target tracker with simulated gimbal (webcam, video file or headless).

#include "vtt/cli.hpp"
#include "vtt/csv.hpp"
#include "vtt/gimbal.hpp"
#include "vtt/metrics.hpp"
#include "vtt/overlay.hpp"
#include "vtt/tracker.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* kWindow = "vision-target-tracker";
constexpr int kKeyEscape = 27;
constexpr int kExitThreshold = 2;

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

std::optional<cv::Rect2d> select_roi(const cv::Mat& frame) {
    const cv::Rect roi = cv::selectROI(kWindow, frame, true, false);
    if (roi.width < 4 || roi.height < 4) {
        return std::nullopt;
    }
    return cv::Rect2d(roi);
}

vtt::Algorithm other(vtt::Algorithm algorithm) {
    return algorithm == vtt::Algorithm::Kcf ? vtt::Algorithm::Mosse : vtt::Algorithm::Kcf;
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
    if (gui) {
        cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
    }
    std::optional<cv::Rect2d> roi = options.roi ? options.roi : truth_at(0);
    if (!roi && gui) {
        roi = select_roi(frame);
    }
    if (!roi) {
        std::cerr << "error: no target selected\n";
        return 1;
    }

    vtt::Algorithm algorithm = options.algorithm;
    vtt::Tracker tracker(make_config(options, algorithm));
    tracker.init(frame, *roi);
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

    vtt::TrackOutput track;
    track.state = vtt::TrackState::Tracking;
    track.box = *roi;
    track.predicted = (roi->tl() + roi->br()) * 0.5;

    long index = 0;
    const Clock::time_point run_start = Clock::now();
    Clock::time_point last_loop = run_start;
    double last_latency = 0.0;
    bool quit = false;
    while (!quit) {
        if (index > 0) {
            const Clock::time_point t0 = Clock::now();
            track = tracker.update(frame);
            last_latency = elapsed_ms(t0, Clock::now());
            latency.add(last_latency);
        }
        const Clock::time_point now = Clock::now();
        const double wall_dt = elapsed_ms(last_loop, now) / 1000.0;
        last_loop = now;
        if (index > 0) {
            loop_rate.tick(wall_dt);
        }
        const double dt = from_file ? 1.0 / source_fps : std::max(wall_dt, 1e-3);
        const bool has_target =
            track.state == vtt::TrackState::Tracking || track.state == vtt::TrackState::Occluded;
        const std::optional<cv::Point2d> aim =
            has_target ? std::optional<cv::Point2d>((track.box.tl() + track.box.br()) * 0.5)
                       : std::nullopt;
        const vtt::GimbalState gimbal_state = gimbal.step(aim, dt);

        const std::optional<cv::Rect2d> gt = truth_at(index);
        if (index > 0 && gt) {
            accuracy.add(track.box, *gt);
        }
        if (csv) {
            csv->write(index, last_latency, track, gimbal_state, gt);
        }

        if (gui || writer.isOpened()) {
            cv::Mat canvas = frame.clone();
            const vtt::Telemetry telemetry{algorithm, index, loop_rate.rate(), last_latency, false};
            vtt::draw_overlay(canvas, track, gimbal, telemetry, overlay);
            if (writer.isOpened()) {
                writer.write(canvas);
            }
            if (gui) {
                cv::imshow(kWindow, canvas);
                int key = cv::waitKey(1);
                while (true) {
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
                        tracker.init(frame, track.box);
                    } else if (key == 'r') {
                        if (const auto selected = select_roi(frame)) {
                            tracker.init(frame, *selected);
                            track.box = *selected;
                            track.state = vtt::TrackState::Tracking;
                        }
                    }
                    if (quit || !paused) {
                        break;
                    }
                    canvas = frame.clone();
                    const vtt::Telemetry paused_telemetry{algorithm, index, loop_rate.rate(),
                                                          last_latency, true};
                    vtt::draw_overlay(canvas, track, gimbal, paused_telemetry, overlay);
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
