#include "vtt/selection.hpp"

#include <algorithm>
#include <stdexcept>

namespace vtt {
namespace {

constexpr int kClickMovePx = 6;
constexpr double kMinDragPx = 8.0;

[[nodiscard]] bool frame_ok(cv::Size frame) noexcept {
    return frame.width > 0 && frame.height > 0;
}

[[nodiscard]] cv::Point clamp_pixel(int x, int y, cv::Size frame) noexcept {
    const int max_x = frame.width - 1;
    const int max_y = frame.height - 1;
    return {std::clamp(x, 0, max_x), std::clamp(y, 0, max_y)};
}

[[nodiscard]] cv::Rect2d normalised_rect(cv::Point a, cv::Point b) noexcept {
    const int left = std::min(a.x, b.x);
    const int top = std::min(a.y, b.y);
    const int right = std::max(a.x, b.x);
    const int bottom = std::max(a.y, b.y);
    return {static_cast<double>(left), static_cast<double>(top), static_cast<double>(right - left),
            static_cast<double>(bottom - top)};
}

/// True when the pointer travelled strictly less than @p pixels (Euclidean).
[[nodiscard]] bool moved_less_than(cv::Point a, cv::Point b, int pixels) noexcept {
    const long long dx = static_cast<long long>(a.x) - static_cast<long long>(b.x);
    const long long dy = static_cast<long long>(a.y) - static_cast<long long>(b.y);
    const long long limit = static_cast<long long>(pixels);
    return dx * dx + dy * dy < limit * limit;
}

[[nodiscard]] cv::Rect2d centred_square(cv::Point click, int side_px, cv::Size frame) noexcept {
    double side = static_cast<double>(side_px);
    side = std::min(side, static_cast<double>(frame.width));
    side = std::min(side, static_cast<double>(frame.height));
    const double max_x = static_cast<double>(frame.width) - side;
    const double max_y = static_cast<double>(frame.height) - side;
    const double x = std::clamp(static_cast<double>(click.x) - side * 0.5, 0.0, std::max(0.0, max_x));
    const double y = std::clamp(static_cast<double>(click.y) - side * 0.5, 0.0, std::max(0.0, max_y));
    return {x, y, side, side};
}

}  // namespace

RoiSelector::RoiSelector(int click_box_px) : click_box_px_(click_box_px) {
    if (click_box_px_ < 16 || click_box_px_ > 400) {
        throw std::invalid_argument("click box must be from 16 to 400 px");
    }
}

std::optional<SelectionEvent> RoiSelector::on_mouse(MouseAction action, int x, int y,
                                                    cv::Size frame_size) {
    if (!frame_ok(frame_size)) {
        cancel_drag();
        return std::nullopt;
    }
    const cv::Point pixel = clamp_pixel(x, y, frame_size);

    if (action == MouseAction::RightDown) {
        cancel_drag();
        return SelectionEvent{Release{}};
    }

    if (action == MouseAction::LeftDown) {
        dragging_ = true;
        anchor_ = pixel;
        current_ = pixel;
        return std::nullopt;
    }

    if (action == MouseAction::Move) {
        if (dragging_) {
            current_ = pixel;
        }
        return std::nullopt;
    }

    if (action != MouseAction::LeftUp || !dragging_) {
        return std::nullopt;
    }

    const cv::Point anchor = anchor_;
    const cv::Point release = pixel;
    cancel_drag();

    if (moved_less_than(anchor, release, kClickMovePx)) {
        return SelectionEvent{Lock{centred_square(anchor, click_box_px_, frame_size)}};
    }

    const cv::Rect2d box = normalised_rect(anchor, release);
    if (box.width < kMinDragPx || box.height < kMinDragPx) {
        return std::nullopt;
    }
    return SelectionEvent{Lock{box}};
}

std::optional<cv::Rect2d> RoiSelector::rubber_band() const noexcept {
    if (!dragging_) {
        return std::nullopt;
    }
    return normalised_rect(anchor_, current_);
}

void RoiSelector::cancel_drag() noexcept {
    dragging_ = false;
    anchor_ = {};
    current_ = {};
}

}  // namespace vtt
