/// @file csv.hpp
/// @brief Per-frame CSV telemetry output and ground-truth loading.
#pragma once

#include "vtt/gimbal.hpp"
#include "vtt/tracker.hpp"

#include <opencv2/core.hpp>

#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace vtt {

/// One ground-truth row: frame index and box.
struct GroundTruthRow {
    int frame = 0;
    cv::Rect2d box;
};

/// Load `frame,x,y,w,h[,...]` rows (header line optional). Throws std::runtime_error on failure.
[[nodiscard]] std::vector<GroundTruthRow> load_ground_truth(const std::string& path);

/// Writes per-frame telemetry rows.
class CsvWriter {
public:
    /// Opens @p path for writing; throws std::runtime_error on failure.
    explicit CsvWriter(const std::string& path);

    void write(long frame, double latency_ms, const TrackOutput& track, const GimbalState& gimbal,
               const std::optional<cv::Rect2d>& truth);

    [[nodiscard]] static std::string header();

private:
    std::ofstream out_;
};

}  // namespace vtt
