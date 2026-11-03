#include "vtt/types.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace vtt {

std::string_view to_string(Algorithm algorithm) noexcept {
    switch (algorithm) {
        case Algorithm::Mosse: return "MOSSE";
        case Algorithm::Kcf: return "KCF";
    }
    return "?";
}

std::string_view to_string(TrackState state) noexcept {
    switch (state) {
        case TrackState::Idle: return "IDLE";
        case TrackState::Tracking: return "TRACKING";
        case TrackState::Occluded: return "OCCLUDED";
        case TrackState::Lost: return "LOST";
    }
    return "?";
}

std::optional<Algorithm> parse_algorithm(std::string_view text) noexcept {
    std::string lower(text);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "mosse") {
        return Algorithm::Mosse;
    }
    if (lower == "kcf") {
        return Algorithm::Kcf;
    }
    return std::nullopt;
}

}  // namespace vtt
