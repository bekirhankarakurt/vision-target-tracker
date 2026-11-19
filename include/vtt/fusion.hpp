/// @file fusion.hpp
/// @brief Detect + track: an external detector corrects the correlation tracker.
///
/// The wire protocol and the decision logic are pure (no sockets, no GUI) so they
/// can be unit-tested. See README "Detect + track".
#pragma once

#include "vtt/types.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace vtt {

/// Frame datagram header sent to the detector: "VTF1", frame id, original size.
struct FrameHeader {
    std::uint32_t frame_id = 0;
    std::uint16_t width = 0;
    std::uint16_t height = 0;
};

inline constexpr std::size_t kFrameHeaderSize = 12;

/// 12 bytes, little endian: magic "VTF1", uint32 frame id, uint16 width, uint16 height.
[[nodiscard]] std::vector<std::uint8_t> encode_frame_header(const FrameHeader& header);
[[nodiscard]] std::optional<FrameHeader> decode_frame_header(const std::vector<std::uint8_t>& bytes);

/// One detector reply. `box` is empty for "NONE <frame>".
struct Detection {
    long frame_id = 0;
    std::optional<cv::Rect2d> box;  ///< Original-frame pixels.
    double confidence = 0.0;
    /// Detector inference time in milliseconds, when the reply includes it.
    std::optional<double> infer_ms;
};

/// Parse "DET <frame> <x> <y> <w> <h> <conf> [<infer_ms>]" or "NONE <frame> [<infer_ms>]".
/// The inference-time field is optional so older replies still parse. Trailing newline allowed.
[[nodiscard]] std::optional<Detection> parse_detection_line(const std::string& line);

struct FusionConfig {
    double min_confidence = 0.5;  ///< Detections below this are ignored.
    double match_iou = 0.3;       ///< IoU that counts as "same object".
    int confirmations = 2;        ///< Consecutive detections needed to act.
    long max_age_frames = 30;     ///< Older detections are ignored.
    bool auto_lock = false;       ///< Lock onto the detected object when nothing is locked.
    double refine_iou = 0.75;     ///< Fresh detections overlapping less than this refine the box.
    long refine_max_age = 6;      ///< Only detections at most this many frames old refine.
};

enum class FusionAction : std::uint8_t { None, AutoLock, Reanchor, Refine };

struct FusionDecision {
    FusionAction action = FusionAction::None;
    cv::Rect2d box;
};

/// Decides when a detection should lock or re-anchor the tracker.
///
/// - The tracker box of every frame is recorded, because a detection arrives a few
///   frames late and must be compared with the box of the frame it was computed on.
/// - A locked target becomes "detector-backed" once a detection overlaps it. Targets
///   the detector never confirms (the user clicked a cup) are never overridden.
/// - A detector-backed target whose box disagrees with `confirmations` consecutive
///   detections has drifted: re-anchor on the detection. A LOST target re-anchors at once.
/// - With auto-lock, `confirmations` consistent detections lock an idle tracker.
/// - Small drift: a fresh detection that still overlaps (IoU in [match_iou, refine_iou))
///   refines the box at once. It is shifted by the tracker's own motion since the
///   detected frame, so the correction does not lag behind a moving object.
class DetectionFusion {
public:
    explicit DetectionFusion(FusionConfig config = {});

    /// Call once per frame after the tracker update.
    void record(long frame_id, TrackState state, const cv::Rect2d& box);

    /// Call whenever a target is locked or released. Pass @p detector_backed = true
    /// when the lock came from a detection (AutoLock / Reanchor).
    void on_target_changed(bool locked, bool detector_backed = false) noexcept;

    [[nodiscard]] FusionDecision on_detection(const Detection& detection, long current_frame);

    [[nodiscard]] bool detector_backed() const noexcept { return backed_; }
    [[nodiscard]] const FusionConfig& config() const noexcept { return config_; }

private:
    struct Entry {
        long frame_id;
        TrackState state;
        cv::Rect2d box;
    };
    [[nodiscard]] const Entry* find(long frame_id) const noexcept;

    FusionConfig config_;
    std::deque<Entry> history_;
    bool locked_ = false;
    bool backed_ = false;
    int disagreements_ = 0;
    int idle_hits_ = 0;
    std::optional<cv::Rect2d> last_idle_box_;
};

}  // namespace vtt
