#include "vtt/overlay.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <string>

namespace vtt {

namespace {

const cv::Scalar kWhite(240, 240, 240);
const cv::Scalar kCyan(255, 220, 0);
const cv::Scalar kGray(150, 150, 150);
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kFontScale = 0.45;

/// Fixed-point number, right-aligned to @p width, with an optional explicit sign.
std::string num(double value, int precision, int width = 0, bool sign = false) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << std::setw(width)
        << (sign ? std::showpos : std::noshowpos) << value;
    return out.str();
}

cv::Point to_point(const cv::Point2d& p) { return {cvRound(p.x), cvRound(p.y)}; }

cv::Rect to_rect(const cv::Rect2d& r) {
    return {cvRound(r.x), cvRound(r.y), cvRound(r.width), cvRound(r.height)};
}

void shade(cv::Mat& canvas, const cv::Rect& area, double alpha) {
    const cv::Rect clipped = area & cv::Rect(0, 0, canvas.cols, canvas.rows);
    if (clipped.area() > 0) {
        cv::Mat roi = canvas(clipped);
        roi.convertTo(roi, -1, 1.0 - alpha, 0.0);
    }
}

void draw_corners(cv::Mat& canvas, const cv::Rect& r, const cv::Scalar& color, int thickness) {
    const int len = std::max(4, std::min(r.width, r.height) / 4);
    const std::array<cv::Point, 4> corners{r.tl(), cv::Point(r.br().x, r.y), r.br(),
                                           cv::Point(r.x, r.br().y)};
    const std::array<cv::Point, 4> dx{cv::Point(len, 0), cv::Point(-len, 0), cv::Point(-len, 0),
                                      cv::Point(len, 0)};
    const std::array<cv::Point, 4> dy{cv::Point(0, len), cv::Point(0, len), cv::Point(0, -len),
                                      cv::Point(0, -len)};
    for (std::size_t i = 0; i < corners.size(); ++i) {
        cv::line(canvas, corners[i], corners[i] + dx[i], color, thickness, cv::LINE_AA);
        cv::line(canvas, corners[i], corners[i] + dy[i], color, thickness, cv::LINE_AA);
    }
}

void draw_reticle(cv::Mat& canvas, const cv::Point& c, int radius, const cv::Scalar& color) {
    cv::circle(canvas, c, radius, color, 1, cv::LINE_AA);
    const int gap = radius / 3;
    cv::line(canvas, c + cv::Point(-radius - 6, 0), c + cv::Point(-gap, 0), color, 1, cv::LINE_AA);
    cv::line(canvas, c + cv::Point(gap, 0), c + cv::Point(radius + 6, 0), color, 1, cv::LINE_AA);
    cv::line(canvas, c + cv::Point(0, -radius - 6), c + cv::Point(0, -gap), color, 1, cv::LINE_AA);
    cv::line(canvas, c + cv::Point(0, gap), c + cv::Point(0, radius + 6), color, 1, cv::LINE_AA);
}

void draw_gimbal_view(cv::Mat& canvas, const cv::Mat& source, const GimbalSim& gimbal) {
    const double zoom = std::max(1.0, gimbal.config().zoom);
    const cv::Size crop_size(std::max(8, cvRound(canvas.cols / zoom)),
                             std::max(8, cvRound(canvas.rows / zoom)));
    const cv::Point2d bore = gimbal.boresight_pixel();
    const cv::Rect crop(cvRound(bore.x - 0.5 * crop_size.width),
                        cvRound(bore.y - 0.5 * crop_size.height), crop_size.width, crop_size.height);
    const cv::Rect inside = crop & cv::Rect(0, 0, source.cols, source.rows);
    cv::Mat view(crop_size, source.type(), cv::Scalar::all(0));
    if (inside.area() > 0) {
        source(inside).copyTo(view(cv::Rect(inside.tl() - crop.tl(), inside.size())));
    }
    const int inset_w = canvas.cols * 3 / 10;
    const int inset_h = inset_w * canvas.rows / std::max(1, canvas.cols);
    const cv::Rect inset(canvas.cols - inset_w - 10, canvas.rows - inset_h - 10, inset_w, inset_h);
    if (inset.x < 0 || inset.y < 0) {
        return;
    }
    cv::Mat resized;
    cv::resize(view, resized, inset.size(), 0.0, 0.0, cv::INTER_LINEAR);
    resized.copyTo(canvas(inset));
    cv::rectangle(canvas, inset, kCyan, 1);
    draw_reticle(canvas, (inset.tl() + inset.br()) / 2, inset_h / 8, kCyan);
    cv::putText(canvas, "GIMBAL VIEW x" + num(zoom, 0), inset.tl() + cv::Point(6, 16), kFont,
                kFontScale, kCyan, 1, cv::LINE_AA);
}

void draw_help(cv::Mat& canvas) {
    static const std::array<const char*, 9> lines{
        "KEYS",           "q / ESC  quit",       "SPACE    pause",
        "r        select ROI", "a        MOSSE <-> KCF", "g        gimbal view",
        "c        center gimbal", "s        snapshot", "h        help"};
    const cv::Rect box(canvas.cols - 230, 10, 220, 22 + 18 * static_cast<int>(lines.size()));
    shade(canvas, box, 0.6);
    int y = box.y + 22;
    for (const char* line : lines) {
        cv::putText(canvas, line, cv::Point(box.x + 10, y), kFont, kFontScale, kWhite, 1,
                    cv::LINE_AA);
        y += 18;
    }
}

}  // namespace

cv::Scalar state_color(TrackState state) noexcept {
    switch (state) {
        case TrackState::Tracking: return {80, 220, 60};
        case TrackState::Occluded: return {0, 200, 255};
        case TrackState::Lost: return {60, 60, 255};
        case TrackState::Idle: break;
    }
    return {150, 150, 150};
}

void draw_overlay(cv::Mat& canvas, const TrackOutput& track, const GimbalSim& gimbal,
                  const Telemetry& telemetry, const OverlayOptions& options) {
    CV_Assert(canvas.type() == CV_8UC3);
    const cv::Mat source = options.show_gimbal_view ? canvas.clone() : cv::Mat();
    const cv::Scalar color = state_color(track.state);

    // Frame centre and gimbal boresight.
    const cv::Point center(canvas.cols / 2, canvas.rows / 2);
    cv::drawMarker(canvas, center, kGray, cv::MARKER_CROSS, 14, 1, cv::LINE_AA);
    draw_reticle(canvas, to_point(gimbal.boresight_pixel()), 22, kCyan);

    if (track.state != TrackState::Idle) {
        if (track.state != TrackState::Tracking) {
            cv::rectangle(canvas, to_rect(track.search_region), kGray, 1, cv::LINE_AA);
        }
        const cv::Rect box = to_rect(track.box);
        if (track.state == TrackState::Tracking) {
            cv::rectangle(canvas, box, color, 1, cv::LINE_AA);
            draw_corners(canvas, box, color, 2);
        } else {
            draw_corners(canvas, box, color, 1);
        }
        const cv::Point predicted = to_point(track.predicted);
        cv::circle(canvas, predicted, 3, kWhite, -1, cv::LINE_AA);
        cv::arrowedLine(canvas, predicted, to_point(track.predicted + track.velocity * 5.0), kWhite,
                        1, cv::LINE_AA, 0, 0.25);
        cv::putText(canvas, std::string(to_string(track.state)), box.tl() + cv::Point(0, -6), kFont,
                    kFontScale, color, 1, cv::LINE_AA);
    }

    if (options.show_gimbal_view) {
        draw_gimbal_view(canvas, source, gimbal);
    }

    // Telemetry HUD.
    const GimbalState& g = gimbal.state();
    const std::array<std::string, 7> lines{
        std::string("VTT ") + std::string(to_string(telemetry.algorithm)) + "  " +
            std::string(to_string(track.state)) + (telemetry.paused ? "  [PAUSED]" : ""),
        "FRAME " + std::to_string(telemetry.frame_index),
        "PSR  " + num(track.psr, 1, 6) + "   SCALE " + num(track.scale, 2),
        "FPS  " + num(telemetry.fps, 1, 6) + "   LAT " + num(telemetry.latency_ms, 2) + " ms",
        "PAN  " + num(g.pan_deg, 2, 7, true) + " deg " + num(g.pan_rate_dps, 1, 7, true) + " dps",
        "TILT " + num(g.tilt_deg, 2, 7, true) + " deg " + num(g.tilt_rate_dps, 1, 7, true) + " dps",
        "ERR  " + num(g.pan_error_deg, 2, 6, true) + " / " + num(g.tilt_error_deg, 2, 6, true) +
            " deg"};
    const cv::Rect panel(10, 10, 270, 14 + 19 * static_cast<int>(lines.size()));
    shade(canvas, panel, 0.55);
    int y = panel.y + 20;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        cv::putText(canvas, lines[i], cv::Point(panel.x + 8, y), kFont, kFontScale,
                    i == 0 ? color : kWhite, 1, cv::LINE_AA);
        y += 19;
    }
    if (options.show_help) {
        draw_help(canvas);
    }
}

}  // namespace vtt
