#include "vtt/synthetic.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace vtt {

namespace {

constexpr double kTwoPi = 2.0 * CV_PI;

/// Smooth random texture: blurred uniform noise, stretched to [lo, hi].
cv::Mat smooth_noise(cv::Size size, double blur_sigma, double lo, double hi, cv::RNG& rng) {
    cv::Mat noise(size, CV_32F);
    rng.fill(noise, cv::RNG::UNIFORM, 0.0, 1.0);
    cv::GaussianBlur(noise, noise, cv::Size(), blur_sigma);
    cv::normalize(noise, noise, lo, hi, cv::NORM_MINMAX);
    return noise;
}

cv::Mat make_background(cv::Size size, cv::RNG& rng) {
    const cv::Mat coarse = smooth_noise(size, 18.0, 60.0, 170.0, rng);
    const cv::Mat fine = smooth_noise(size, 1.5, -12.0, 12.0, rng);
    const cv::Mat gray = coarse + fine;
    cv::Mat gray8;
    gray.convertTo(gray8, CV_8U);
    cv::Mat bgr;
    cv::cvtColor(gray8, bgr, cv::COLOR_GRAY2BGR);
    // Slight green/brown tint for a terrain look.
    bgr.forEach<cv::Vec3b>([](cv::Vec3b& p, const int*) {
        p[0] = cv::saturate_cast<uchar>(p[0] * 0.85);
        p[2] = cv::saturate_cast<uchar>(p[2] * 0.95);
    });
    return bgr;
}

cv::Mat make_target(cv::Size size, cv::RNG& rng) {
    const cv::Mat texture = smooth_noise(size, 2.5, 20.0, 235.0, rng);
    cv::Mat gray8;
    texture.convertTo(gray8, CV_8U);
    cv::Mat bgr;
    cv::cvtColor(gray8, bgr, cv::COLOR_GRAY2BGR);
    const int cx = size.width / 2;
    const int cy = size.height / 2;
    cv::rectangle(bgr, cv::Rect(0, 0, size.width, size.height), cv::Scalar(30, 30, 200), 3);
    cv::circle(bgr, cv::Point(cx, cy), std::min(cx, cy) / 2, cv::Scalar(240, 240, 240), -1,
               cv::LINE_AA);
    cv::line(bgr, cv::Point(cx, 4), cv::Point(cx, size.height - 5), cv::Scalar(20, 20, 20), 2);
    cv::line(bgr, cv::Point(4, cy), cv::Point(size.width - 5, cy), cv::Scalar(20, 20, 20), 2);
    return bgr;
}

cv::Mat make_occluder(cv::Size size, cv::RNG& rng) {
    const cv::Mat texture = smooth_noise(size, 6.0, 70.0, 110.0, rng);
    cv::Mat gray8;
    texture.convertTo(gray8, CV_8U);
    cv::Mat bgr;
    cv::cvtColor(gray8, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

/// Alpha-free paste of @p src resized into @p dst at @p box (clipped to the frame).
void paste(cv::Mat& dst, const cv::Mat& src, const cv::Rect2d& box) {
    const cv::Rect rect(cvRound(box.x), cvRound(box.y), std::max(1, cvRound(box.width)),
                        std::max(1, cvRound(box.height)));
    const cv::Rect clipped = rect & cv::Rect(0, 0, dst.cols, dst.rows);
    if (clipped.area() == 0) {
        return;
    }
    cv::Mat resized;
    cv::resize(src, resized, rect.size(), 0.0, 0.0, cv::INTER_AREA);
    resized(cv::Rect(clipped.tl() - rect.tl(), clipped.size())).copyTo(dst(clipped));
}

}  // namespace

SyntheticScene::SyntheticScene(const SyntheticConfig& config) : config_(config) {
    CV_Assert(config.frame_size.width >= 64 && config.frame_size.height >= 64);
    CV_Assert(config.target_size.width >= 8 && config.target_size.height >= 8);
    cv::RNG rng(config.seed);
    background_ = make_background(config.frame_size, rng);
    target_texture_ = make_target(config.target_size, rng);
    const cv::Size occluder_size(cvRound(config.target_size.width * (1.0 + config.scale_amplitude) * 1.4),
                                 cvRound(config.target_size.height * (1.0 + config.scale_amplitude) * 1.6));
    occluder_texture_ = make_occluder(occluder_size, rng);
}

cv::Rect2d SyntheticScene::target_box(int frame) const {
    const auto t = static_cast<double>(frame);
    const double cx = 0.5 * static_cast<double>(config_.frame_size.width) +
                      config_.amplitude.x * std::sin(kTwoPi * t / config_.period_x);
    const double cy = 0.5 * static_cast<double>(config_.frame_size.height) +
                      config_.amplitude.y * std::sin(kTwoPi * t / config_.period_y + 0.6);
    const double scale = 1.0 + config_.scale_amplitude * std::sin(kTwoPi * t / config_.scale_period);
    const double w = static_cast<double>(config_.target_size.width) * scale;
    const double h = static_cast<double>(config_.target_size.height) * scale;
    return {cx - 0.5 * w, cy - 0.5 * h, w, h};
}

cv::Rect2d SyntheticScene::occluder_box(int frame) const {
    const int start = config_.occlusion_start;
    const int ramp = std::max(1, config_.occlusion_ramp);
    const int hold = std::max(0, config_.occlusion_hold);
    const int end = start + 2 * ramp + hold;
    if (frame < start - ramp || frame > end) {
        return {};
    }
    const cv::Rect2d target = target_box(frame);
    const auto w = static_cast<double>(occluder_texture_.cols);
    const auto h = static_cast<double>(occluder_texture_.rows);
    // Horizontal offset: slides in from the left, holds over the target, slides out to the right.
    const double travel = 0.5 * (w + target.width) + 2.0;
    double offset = 0.0;
    if (frame < start + ramp) {
        offset = -travel * static_cast<double>(start + ramp - frame) / static_cast<double>(2 * ramp);
    } else if (frame > start + ramp + hold) {
        offset = travel * static_cast<double>(frame - start - ramp - hold) / static_cast<double>(2 * ramp);
    }
    const double cx = target.x + 0.5 * target.width + offset;
    const double cy = target.y + 0.5 * target.height;
    return {cx - 0.5 * w, cy - 0.5 * h, w, h};
}

cv::Mat SyntheticScene::render(int frame) const {
    cv::Mat image = background_.clone();
    paste(image, target_texture_, target_box(frame));
    const cv::Rect2d occluder = occluder_box(frame);
    if (occluder.area() > 0.0) {
        paste(image, occluder_texture_, occluder);
    }
    if (config_.noise_std > 0.0) {
        cv::RNG rng(config_.seed * 7919U + static_cast<std::uint64_t>(frame) + 1U);
        cv::Mat noise(image.size(), CV_16SC3);
        rng.fill(noise, cv::RNG::NORMAL, 0.0, config_.noise_std);
        cv::Mat noisy;
        image.convertTo(noisy, CV_16SC3);
        noisy += noise;
        noisy.convertTo(image, CV_8UC3);
    }
    return image;
}

GroundTruth SyntheticScene::ground_truth(int frame) const {
    GroundTruth gt;
    gt.box = target_box(frame);
    const cv::Rect2d frame_rect(0.0, 0.0, static_cast<double>(config_.frame_size.width),
                                static_cast<double>(config_.frame_size.height));
    const double area = gt.box.area();
    const double in_frame = (gt.box & frame_rect).area();
    const double covered = (gt.box & occluder_box(frame) & frame_rect).area();
    gt.visibility = area > 0.0 ? std::clamp((in_frame - covered) / area, 0.0, 1.0) : 0.0;
    return gt;
}

}  // namespace vtt
