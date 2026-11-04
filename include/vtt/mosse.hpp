/// @file mosse.hpp
/// @brief MOSSE correlation filter (Bolme et al., CVPR 2010).
#pragma once

#include "vtt/correlation_filter.hpp"

namespace vtt {

/// Single-channel MOSSE filter: H* = sum(G . F*) / sum(F . F*).
class MosseFilter final : public CorrelationFilter {
public:
    explicit MosseFilter(const FilterParams& params);

    void init(const cv::Mat& patch) override;
    [[nodiscard]] cv::Mat respond(const cv::Mat& patch) const override;
    void update(const cv::Mat& patch) override;

    [[nodiscard]] Algorithm algorithm() const noexcept override { return Algorithm::Mosse; }
    [[nodiscard]] cv::Size patch_size() const noexcept override { return size_; }

private:
    void refresh_filter();

    FilterParams params_;
    cv::Size size_;
    cv::Mat window_;
    cv::Mat target_spectrum_;  ///< G
    cv::Mat numerator_;        ///< A = G . F*
    cv::Mat denominator_;      ///< B = F . F*
    cv::Mat filter_;           ///< H* = A / (B + eps)
};

}  // namespace vtt
