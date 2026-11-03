#include "vtt/tracker.hpp"

#include "vtt/spectral.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vtt {

namespace {

constexpr int kMinPatchSide = 16;

int round_up_to_multiple(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

cv::Point2d rect_center(const cv::Rect2d& r) {
    return {r.x + 0.5 * r.width, r.y + 0.5 * r.height};
}

cv::Rect2d centered_rect(const cv::Point2d& center, const cv::Size2d& size) {
    return {center.x - 0.5 * size.width, center.y - 0.5 * size.height, size.width, size.height};
}

}  // namespace

cv::Mat to_gray(const cv::Mat& frame) {
    CV_Assert(!frame.empty());
    cv::Mat source = frame;
    if (frame.depth() != CV_8U) {
        frame.convertTo(source, CV_8U);
    }
    cv::Mat gray;
    switch (source.channels()) {
        case 1: return source;
        case 3: cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY); break;
        case 4: cv::cvtColor(source, gray, cv::COLOR_BGRA2GRAY); break;
        default: throw std::invalid_argument("unsupported channel count");
    }
    return gray;
}

TrackerConfig default_tracker_config(Algorithm algorithm) {
    TrackerConfig config;
    config.algorithm = algorithm;
    config.filter = default_filter_params(algorithm);
    if (algorithm == Algorithm::Mosse) {
        config.padding = 2.0;
        config.psr_track = 7.0;
        config.psr_update = 9.0;
        config.psr_reacquire = 10.0;
    }
    return config;
}

Tracker::Tracker(const TrackerConfig& config)
    : config_(config), scale_filter_(config.scale), kalman_(config.kalman) {
    if (config_.padding < 1.0 || config_.template_size < kMinPatchSide ||
        config_.search_spacing <= 0.0) {
        throw std::invalid_argument("invalid tracker configuration");
    }
}

void Tracker::reset() {
    filter_.reset();
    state_ = TrackState::Idle;
    frames_without_measurement_ = 0;
    frames_lost_ = 0;
    scale_ = 1.0;
}

cv::Size2d Tracker::window_size(double scale) const {
    return {base_target_size_.width * config_.padding * scale,
            base_target_size_.height * config_.padding * scale};
}

cv::Size2d Tracker::target_size() const {
    return {base_target_size_.width * scale_, base_target_size_.height * scale_};
}

cv::Point2d Tracker::clamp_to_frame(const cv::Point2d& p) const {
    return {std::clamp(p.x, 0.0, static_cast<double>(frame_size_.width - 1)),
            std::clamp(p.y, 0.0, static_cast<double>(frame_size_.height - 1))};
}

cv::Mat Tracker::extract(const cv::Mat& gray, const cv::Point2d& center, double scale,
                         cv::Rect& used_window) const {
    const cv::Size2d window = window_size(scale);
    const int w = std::max(1, cvRound(window.width));
    const int h = std::max(1, cvRound(window.height));
    used_window = cv::Rect(cvRound(center.x - 0.5 * window.width),
                           cvRound(center.y - 0.5 * window.height), w, h);
    const cv::Rect inside = used_window & cv::Rect(0, 0, gray.cols, gray.rows);
    cv::Mat crop;
    if (inside.area() == 0) {
        crop = cv::Mat(h, w, CV_8U, cv::Scalar(cv::mean(gray)[0]));
    } else if (inside == used_window) {
        crop = gray(inside);
    } else {
        cv::copyMakeBorder(gray(inside), crop, inside.y - used_window.y,
                           used_window.br().y - inside.br().y, inside.x - used_window.x,
                           used_window.br().x - inside.br().x, cv::BORDER_REPLICATE);
    }
    const cv::Size patch = filter_ ? filter_->patch_size() : cv::Size();
    cv::Mat resized;
    const int interpolation = w > patch.width ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(crop, resized, patch, 0.0, 0.0, interpolation);
    cv::Mat out;
    resized.convertTo(out, CV_32F, 1.0 / 255.0);
    return out;
}

void Tracker::init(const cv::Mat& frame, const cv::Rect2d& box) {
    if (frame.empty() || box.width < 2.0 || box.height < 2.0) {
        throw std::invalid_argument("tracker init requires a frame and a box of at least 2x2 px");
    }
    const cv::Mat gray = to_gray(frame);
    frame_size_ = gray.size();
    base_target_size_ = box.size();
    center_ = clamp_to_frame(rect_center(box));
    scale_ = 1.0;

    const cv::Size2d window = window_size(1.0);
    const double s = static_cast<double>(config_.template_size) /
                     std::sqrt(window.width * window.height);
    const cv::Size patch(
        std::max(kMinPatchSide, round_up_to_multiple(cvRound(window.width * s), 4)),
        std::max(kMinPatchSide, round_up_to_multiple(cvRound(window.height * s), 4)));

    filter_ = make_filter(config_.algorithm, config_.filter);
    // Size the filter by initialising on a blank patch, then train on the real one.
    filter_->init(cv::Mat::zeros(patch, CV_32F));
    cv::Rect used;
    filter_->init(extract(gray, center_, scale_, used));
    scale_filter_ = ScaleFilter(config_.scale);
    if (config_.estimate_scale) {
        scale_filter_.init(gray, center_, target_size());
    }

    kalman_ = ConstantVelocityKalman(config_.kalman);
    kalman_.init(center_);
    state_ = TrackState::Tracking;
    frames_without_measurement_ = 0;
    frames_lost_ = 0;
}

Tracker::Candidate Tracker::evaluate(const cv::Mat& gray, const cv::Point2d& center) const {
    cv::Rect window;
    const cv::Mat response = filter_->respond(extract(gray, center, scale_, window));
    const spectral::Peak peak = spectral::find_peak(response);
    const cv::Size patch = filter_->patch_size();
    const double sx = static_cast<double>(window.width) / static_cast<double>(patch.width);
    const double sy = static_cast<double>(window.height) / static_cast<double>(patch.height);
    const cv::Point2d origin = spectral::response_center(patch);
    const cv::Point2d displacement((peak.location.x - origin.x) * sx,
                                   (peak.location.y - origin.y) * sy);
    Candidate candidate;
    candidate.window = cv::Rect2d(window);
    candidate.center = rect_center(candidate.window) + displacement;
    candidate.psr = spectral::peak_to_sidelobe_ratio(response, peak);
    return candidate;
}

std::vector<cv::Point2d> Tracker::search_centers(const cv::Point2d& predicted) const {
    std::vector<cv::Point2d> centers{predicted};
    if (state_ != TrackState::Lost) {
        return centers;
    }
    const double radius_units = std::min(config_.max_search_radius,
                                         config_.search_growth * static_cast<double>(frames_lost_));
    if (radius_units < 0.25) {
        return centers;
    }
    const cv::Size2d window = window_size(scale_);
    const int half = static_cast<int>(std::ceil(radius_units / config_.search_spacing - 1e-9));
    const double step = radius_units / static_cast<double>(half);
    for (int j = -half; j <= half; ++j) {
        for (int i = -half; i <= half; ++i) {
            if (i == 0 && j == 0) {
                continue;
            }
            centers.push_back(clamp_to_frame(
                predicted + cv::Point2d(static_cast<double>(i) * step * window.width,
                                        static_cast<double>(j) * step * window.height)));
        }
    }
    return centers;
}

TrackOutput Tracker::update(const cv::Mat& frame) {
    TrackOutput out;
    if (state_ == TrackState::Idle || !filter_) {
        return out;
    }
    const cv::Mat gray = to_gray(frame);
    CV_Assert(gray.size() == frame_size_);

    kalman_.predict(1.0);
    if (state_ == TrackState::Lost) {
        kalman_.damp_velocity(config_.lost_velocity_damping);
    }
    kalman_.set_position(clamp_to_frame(kalman_.position()));
    const cv::Point2d predicted = kalman_.position();
    out.predicted = predicted;

    Candidate best;
    best.psr = -1e300;
    bool first = true;
    for (const cv::Point2d& c : search_centers(predicted)) {
        const Candidate candidate = evaluate(gray, c);
        out.search_region = first ? candidate.window : (out.search_region | candidate.window);
        first = false;
        if (candidate.psr > best.psr) {
            best = candidate;
        }
    }
    out.psr = best.psr;

    const double threshold =
        state_ == TrackState::Tracking ? config_.psr_track : config_.psr_reacquire;
    if (best.psr >= threshold) {
        const double min_scale = config_.min_target_px /
                                 std::min(base_target_size_.width, base_target_size_.height);
        const double max_scale =
            std::min(static_cast<double>(frame_size_.width) / base_target_size_.width,
                     static_cast<double>(frame_size_.height) / base_target_size_.height);
        center_ = clamp_to_frame(best.center);
        const bool confident = best.psr >= config_.psr_update;
        if (config_.estimate_scale && confident) {
            const double factor = scale_filter_.estimate(gray, center_, target_size());
            scale_ = std::clamp(scale_ * factor, min_scale, std::max(min_scale, max_scale));
        }
        kalman_.correct(center_);
        state_ = TrackState::Tracking;
        frames_without_measurement_ = 0;
        frames_lost_ = 0;
        if (confident) {
            cv::Rect used;
            filter_->update(extract(gray, center_, scale_, used));
            if (config_.estimate_scale) {
                scale_filter_.update(gray, center_, target_size());
            }
        }
    } else {
        ++frames_without_measurement_;
        if (frames_without_measurement_ > config_.max_occluded_frames) {
            state_ = TrackState::Lost;
            ++frames_lost_;
        } else {
            state_ = TrackState::Occluded;
        }
        center_ = kalman_.position();
    }

    out.state = state_;
    out.scale = scale_;
    out.velocity = kalman_.velocity();
    out.frames_without_measurement = frames_without_measurement_;
    out.box = centered_rect(center_, target_size());
    return out;
}

}  // namespace vtt
