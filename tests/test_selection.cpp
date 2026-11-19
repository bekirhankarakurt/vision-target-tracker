#include "test_harness.hpp"

#include "vtt/selection.hpp"

#include <optional>
#include <variant>

using namespace vtt;

namespace {

const cv::Size kFrame(640, 480);

Lock expect_lock(const std::optional<SelectionEvent>& event) {
    CHECK(event.has_value());
    CHECK(event && std::holds_alternative<Lock>(*event));
    if (!event || !std::holds_alternative<Lock>(*event)) {
        return Lock{};
    }
    return std::get<Lock>(*event);
}

void expect_release(const std::optional<SelectionEvent>& event) {
    CHECK(event.has_value());
    CHECK(event && std::holds_alternative<Release>(*event));
}

void expect_inside(const cv::Rect2d& box, cv::Size frame) {
    CHECK(box.width > 0.0);
    CHECK(box.height > 0.0);
    CHECK(box.x >= 0.0);
    CHECK(box.y >= 0.0);
    CHECK(box.x + box.width <= static_cast<double>(frame.width) + 1e-9);
    CHECK(box.y + box.height <= static_cast<double>(frame.height) + 1e-9);
}

}  // namespace

VTT_TEST(drag_locks_normalised_rect) {
    RoiSelector selector(80);
    CHECK(!selector.rubber_band());
    CHECK(!selector.on_mouse(MouseAction::LeftDown, 10, 20, kFrame));
    CHECK(selector.rubber_band().has_value());
    CHECK(!selector.on_mouse(MouseAction::Move, 50, 80, kFrame));
    const std::optional<cv::Rect2d> band = selector.rubber_band();
    CHECK(band.has_value());
    CHECK_NEAR(band->x, 10.0, 1e-9);
    CHECK_NEAR(band->y, 20.0, 1e-9);
    CHECK_NEAR(band->width, 40.0, 1e-9);
    CHECK_NEAR(band->height, 60.0, 1e-9);

    const Lock forward = expect_lock(selector.on_mouse(MouseAction::LeftUp, 50, 80, kFrame));
    CHECK_NEAR(forward.box.x, 10.0, 1e-9);
    CHECK_NEAR(forward.box.y, 20.0, 1e-9);
    CHECK_NEAR(forward.box.width, 40.0, 1e-9);
    CHECK_NEAR(forward.box.height, 60.0, 1e-9);
    expect_inside(forward.box, kFrame);
    CHECK(!selector.rubber_band());

    // Right-to-left and bottom-to-top still produce a positive top-left box.
    CHECK(!selector.on_mouse(MouseAction::LeftDown, 100, 90, kFrame));
    CHECK(!selector.on_mouse(MouseAction::Move, 40, 30, kFrame));
    const std::optional<cv::Rect2d> reverse_band = selector.rubber_band();
    CHECK(reverse_band.has_value());
    CHECK_NEAR(reverse_band->x, 40.0, 1e-9);
    CHECK_NEAR(reverse_band->y, 30.0, 1e-9);
    CHECK_NEAR(reverse_band->width, 60.0, 1e-9);
    CHECK_NEAR(reverse_band->height, 60.0, 1e-9);
    const Lock reverse = expect_lock(selector.on_mouse(MouseAction::LeftUp, 40, 30, kFrame));
    CHECK_NEAR(reverse.box.x, 40.0, 1e-9);
    CHECK_NEAR(reverse.box.y, 30.0, 1e-9);
    CHECK_NEAR(reverse.box.width, 60.0, 1e-9);
    CHECK_NEAR(reverse.box.height, 60.0, 1e-9);
    CHECK(!selector.rubber_band());

    // A point past the frame is clamped, and the locked box stays inside.
    CHECK(!selector.on_mouse(MouseAction::LeftDown, 0, 0, kFrame));
    const Lock clamped = expect_lock(selector.on_mouse(MouseAction::LeftUp, 5000, 5000, kFrame));
    CHECK_NEAR(clamped.box.x, 0.0, 1e-9);
    CHECK_NEAR(clamped.box.y, 0.0, 1e-9);
    CHECK_NEAR(clamped.box.width, 639.0, 1e-9);
    CHECK_NEAR(clamped.box.height, 479.0, 1e-9);
    expect_inside(clamped.box, kFrame);
}

VTT_TEST(tiny_drag_is_ignored) {
    RoiSelector selector(80);
    CHECK(!selector.on_mouse(MouseAction::LeftDown, 10, 10, kFrame));
    CHECK(selector.rubber_band().has_value());
    // 7 px wide: a drag, not a click, and under the 8x8 minimum.
    const std::optional<SelectionEvent> event =
        selector.on_mouse(MouseAction::LeftUp, 17, 40, kFrame);
    CHECK(!event);
    CHECK(!selector.rubber_band());
}

VTT_TEST(click_locks_centred_square) {
    RoiSelector selector(80);
    CHECK(!selector.on_mouse(MouseAction::LeftDown, 200, 150, kFrame));
    // 5 px of travel is still a click (threshold is 6).
    CHECK(!selector.on_mouse(MouseAction::Move, 205, 150, kFrame));
    CHECK(selector.rubber_band().has_value());
    const Lock lock = expect_lock(selector.on_mouse(MouseAction::LeftUp, 205, 150, kFrame));
    CHECK_NEAR(lock.box.x, 160.0, 1e-9);
    CHECK_NEAR(lock.box.y, 110.0, 1e-9);
    CHECK_NEAR(lock.box.width, 80.0, 1e-9);
    CHECK_NEAR(lock.box.height, 80.0, 1e-9);
    CHECK_NEAR(lock.box.x + lock.box.width * 0.5, 200.0, 1e-9);
    CHECK_NEAR(lock.box.y + lock.box.height * 0.5, 150.0, 1e-9);
    expect_inside(lock.box, kFrame);
    CHECK(!selector.rubber_band());
}

VTT_TEST(click_near_edge_is_clamped) {
    RoiSelector selector(80);
    CHECK(!selector.on_mouse(MouseAction::LeftDown, 2, 3, kFrame));
    const Lock top_left = expect_lock(selector.on_mouse(MouseAction::LeftUp, 2, 3, kFrame));
    CHECK_NEAR(top_left.box.x, 0.0, 1e-9);
    CHECK_NEAR(top_left.box.y, 0.0, 1e-9);
    CHECK_NEAR(top_left.box.width, 80.0, 1e-9);
    CHECK_NEAR(top_left.box.height, 80.0, 1e-9);
    expect_inside(top_left.box, kFrame);

    CHECK(!selector.on_mouse(MouseAction::LeftDown, 639, 479, kFrame));
    const Lock bottom_right = expect_lock(selector.on_mouse(MouseAction::LeftUp, 639, 479, kFrame));
    CHECK_NEAR(bottom_right.box.x, 560.0, 1e-9);
    CHECK_NEAR(bottom_right.box.y, 400.0, 1e-9);
    CHECK_NEAR(bottom_right.box.width, 80.0, 1e-9);
    CHECK_NEAR(bottom_right.box.height, 80.0, 1e-9);
    expect_inside(bottom_right.box, kFrame);

    const cv::Size kSmall(30, 20);
    CHECK(!selector.on_mouse(MouseAction::LeftDown, 15, 10, kSmall));
    const Lock shrunk = expect_lock(selector.on_mouse(MouseAction::LeftUp, 15, 10, kSmall));
    CHECK_NEAR(shrunk.box.width, 20.0, 1e-9);
    CHECK_NEAR(shrunk.box.height, 20.0, 1e-9);
    expect_inside(shrunk.box, kSmall);
}

VTT_TEST(right_click_releases) {
    RoiSelector selector(80);
    CHECK(!selector.on_mouse(MouseAction::LeftDown, 30, 30, kFrame));
    CHECK(!selector.on_mouse(MouseAction::Move, 80, 70, kFrame));
    CHECK(selector.rubber_band().has_value());
    expect_release(selector.on_mouse(MouseAction::RightDown, 80, 70, kFrame));
    CHECK(!selector.rubber_band());

    expect_release(selector.on_mouse(MouseAction::RightDown, 10, 10, kFrame));
    CHECK(!selector.rubber_band());
}

VTT_TEST(rubber_band_only_while_dragging) {
    RoiSelector selector(80);
    CHECK(!selector.rubber_band());
    CHECK(!selector.on_mouse(MouseAction::Move, 12, 12, kFrame));
    CHECK(!selector.rubber_band());
    CHECK(!selector.on_mouse(MouseAction::LeftDown, 12, 12, kFrame));
    CHECK(selector.rubber_band().has_value());
    CHECK(!selector.on_mouse(MouseAction::Move, 40, 50, kFrame));
    const std::optional<cv::Rect2d> band = selector.rubber_band();
    CHECK(band.has_value());
    CHECK_NEAR(band->width, 28.0, 1e-9);
    CHECK_NEAR(band->height, 38.0, 1e-9);
    CHECK(selector.on_mouse(MouseAction::LeftUp, 40, 50, kFrame).has_value());
    CHECK(!selector.rubber_band());
    CHECK(!selector.on_mouse(MouseAction::Move, 70, 70, kFrame));
    CHECK(!selector.rubber_band());
}

VTT_TEST_MAIN()
