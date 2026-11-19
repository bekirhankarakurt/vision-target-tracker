/// @file selection.hpp
/// @brief Live target-selection gestures. No OpenCV GUI calls.
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <optional>
#include <variant>

namespace vtt {

/// Application-facing mouse actions. Map `cv::EVENT_*` onto these in the GUI.
enum class MouseAction : std::uint8_t {
    Move,
    LeftDown,
    LeftUp,
    RightDown,
};

/// Lock the tracker onto a rectangle in frame pixels.
struct Lock {
    cv::Rect2d box;
};

/// Drop the locked target and return the tracker to idle.
struct Release {};

using SelectionEvent = std::variant<Lock, Release>;

/// Converts mouse gestures into lock and release events.
///
/// A left drag whose box is at least 8×8 px locks that rectangle. Corner order
/// does not matter: right-to-left and bottom-to-top drags are normalised.
/// A left press that moves less than 6 px before release locks a square of
/// @p click_box_px centred on the press and clamped inside the frame.
/// A right press releases. `rubber_band()` is set only while the left button is held.
class RoiSelector {
public:
    /// @p click_box_px must be in [16, 400].
    explicit RoiSelector(int click_box_px);

    /// Feed one mouse action. @p x and @p y are frame pixels and are clamped to
    /// @p frame_size. Returns a lock or release when a gesture completes.
    [[nodiscard]] std::optional<SelectionEvent> on_mouse(MouseAction action, int x, int y,
                                                         cv::Size frame_size);

    /// Rectangle between the press and the pointer while a left drag is active.
    [[nodiscard]] std::optional<cv::Rect2d> rubber_band() const noexcept;

    /// Forget an in-progress drag without emitting an event.
    void cancel_drag() noexcept;

private:
    int click_box_px_;
    bool dragging_ = false;
    cv::Point anchor_{};
    cv::Point current_{};
};

}  // namespace vtt
