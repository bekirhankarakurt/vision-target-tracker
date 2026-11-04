/// @file types.hpp
/// @brief Shared enumerations used across the tracking pipeline.
#pragma once

#include <optional>
#include <string_view>

namespace vtt {

/// Correlation-filter algorithm used by the tracker.
enum class Algorithm {
    Mosse,  ///< Minimum Output Sum of Squared Error filter (single channel).
    Kcf,    ///< Kernelized Correlation Filter with a Gaussian kernel.
};

/// High-level tracker state machine.
enum class TrackState {
    Idle,      ///< Not initialised.
    Tracking,  ///< Confident measurement this frame.
    Occluded,  ///< Low confidence, coasting on the Kalman prediction.
    Lost,      ///< Occluded for too long; searching an expanding window.
};

[[nodiscard]] std::string_view to_string(Algorithm algorithm) noexcept;
[[nodiscard]] std::string_view to_string(TrackState state) noexcept;
[[nodiscard]] std::optional<Algorithm> parse_algorithm(std::string_view text) noexcept;

}  // namespace vtt
