#include "vtt/fusion.hpp"

#include "vtt/metrics.hpp"

#include <cmath>
#include <istream>
#include <sstream>

namespace vtt {
namespace {

constexpr std::size_t kHistoryFrames = 120;
constexpr std::uint8_t kMagic[4] = {'V', 'T', 'F', '1'};

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((v >> shift) & 0xFFU));
    }
}

void put_u16(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 8U) & 0xFFU));
}

std::uint32_t get_u32(const std::vector<std::uint8_t>& in, std::size_t at) {
    std::uint32_t v = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        v |= static_cast<std::uint32_t>(in[at + i]) << (8U * i);
    }
    return v;
}

std::uint16_t get_u16(const std::vector<std::uint8_t>& in, std::size_t at) {
    return static_cast<std::uint16_t>(in[at] | (in[at + 1] << 8U));
}

/// Optional trailing inference time. Absent is success with @p infer_ms left empty.
/// A present value must be finite and >= 0, and it must be the last token.
bool read_optional_infer_ms(std::istream& in, std::optional<double>& infer_ms) {
    in >> std::ws;
    if (in.eof() || in.peek() == std::istream::traits_type::eof()) {
        return true;
    }
    double ms = 0.0;
    if (!(in >> ms) || !std::isfinite(ms) || ms < 0.0) {
        return false;
    }
    in >> std::ws;
    if (!(in.eof() || in.peek() == std::istream::traits_type::eof())) {
        return false;
    }
    infer_ms = ms;
    return true;
}

}  // namespace

std::vector<std::uint8_t> encode_frame_header(const FrameHeader& header) {
    std::vector<std::uint8_t> out(std::begin(kMagic), std::end(kMagic));
    out.reserve(kFrameHeaderSize);
    put_u32(out, header.frame_id);
    put_u16(out, header.width);
    put_u16(out, header.height);
    return out;
}

std::optional<FrameHeader> decode_frame_header(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < kFrameHeaderSize) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        if (bytes[i] != kMagic[i]) {
            return std::nullopt;
        }
    }
    return FrameHeader{get_u32(bytes, 4), get_u16(bytes, 8), get_u16(bytes, 10)};
}

std::optional<Detection> parse_detection_line(const std::string& line) {
    std::istringstream in(line);
    std::string kind;
    long frame = 0;
    if (!(in >> kind >> frame) || frame < 0) {
        return std::nullopt;
    }
    Detection detection;
    detection.frame_id = frame;
    if (kind == "NONE") {
        if (!read_optional_infer_ms(in, detection.infer_ms)) {
            return std::nullopt;
        }
        return detection;
    }
    if (kind != "DET") {
        return std::nullopt;
    }
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
    double conf = 0.0;
    if (!(in >> x >> y >> w >> h >> conf) || !read_optional_infer_ms(in, detection.infer_ms)) {
        return std::nullopt;
    }
    for (const double v : {x, y, w, h, conf}) {
        if (!std::isfinite(v)) {
            return std::nullopt;
        }
    }
    if (w <= 0.0 || h <= 0.0 || conf < 0.0 || conf > 1.0) {
        return std::nullopt;
    }
    detection.box = cv::Rect2d(x, y, w, h);
    detection.confidence = conf;
    return detection;
}

DetectionFusion::DetectionFusion(FusionConfig config) : config_(config) {}

void DetectionFusion::record(long frame_id, TrackState state, const cv::Rect2d& box) {
    history_.push_back(Entry{frame_id, state, box});
    while (history_.size() > kHistoryFrames) {
        history_.pop_front();
    }
}

void DetectionFusion::on_target_changed(bool locked, bool detector_backed) noexcept {
    locked_ = locked;
    backed_ = locked && detector_backed;
    disagreements_ = 0;
    idle_hits_ = 0;
    last_idle_box_.reset();
}

const DetectionFusion::Entry* DetectionFusion::find(long frame_id) const noexcept {
    for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
        if (it->frame_id == frame_id) {
            return &*it;
        }
    }
    return nullptr;
}

FusionDecision DetectionFusion::on_detection(const Detection& detection, long current_frame) {
    const long age = current_frame - detection.frame_id;
    if (age < 0 || age > config_.max_age_frames) {
        return {};
    }
    const bool usable = detection.box && detection.confidence >= config_.min_confidence;

    if (!locked_) {
        if (!config_.auto_lock || !usable) {
            idle_hits_ = 0;
            last_idle_box_.reset();
            return {};
        }
        if (last_idle_box_ && iou(*last_idle_box_, *detection.box) >= config_.match_iou) {
            ++idle_hits_;
        } else {
            idle_hits_ = 1;
        }
        last_idle_box_ = *detection.box;
        if (idle_hits_ >= config_.confirmations) {
            return {FusionAction::AutoLock, *detection.box};
        }
        return {};
    }

    // A missed detection is no evidence of drift: the object may just be hidden.
    if (!usable) {
        return {};
    }
    const Entry* entry = find(detection.frame_id);
    if (entry == nullptr) {
        return {};
    }
    const double overlap = iou(entry->box, *detection.box);
    if (overlap >= config_.match_iou) {
        backed_ = true;
        disagreements_ = 0;
        if (overlap < config_.refine_iou && age <= config_.refine_max_age && !history_.empty()) {
            // The tracker is roughly right locally, so its motion since the detected
            // frame is a good estimate of how far the object has moved meanwhile.
            const Entry& now = history_.back();
            const cv::Point2d shift = (now.box.tl() + now.box.br()) * 0.5 -
                                      (entry->box.tl() + entry->box.br()) * 0.5;
            cv::Rect2d refined = *detection.box;
            refined.x += shift.x;
            refined.y += shift.y;
            return {FusionAction::Refine, refined};
        }
        return {};
    }
    if (!backed_) {
        return {};  // The user tracks something the detector does not know.
    }
    if (entry->state == TrackState::Lost) {
        disagreements_ = 0;
        return {FusionAction::Reanchor, *detection.box};
    }
    if (++disagreements_ >= config_.confirmations) {
        disagreements_ = 0;
        return {FusionAction::Reanchor, *detection.box};
    }
    return {};
}

}  // namespace vtt
