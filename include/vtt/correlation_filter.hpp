/// @file correlation_filter.hpp
/// @brief Common interface for fixed-size correlation filters.
#pragma once

#include "vtt/types.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace vtt {

/// Hyper-parameters shared by MOSSE and KCF.
struct FilterParams {
    double output_sigma = 2.0;     ///< Std-dev of the desired Gaussian response (patch pixels).
    double learning_rate = 0.1;    ///< Model interpolation factor per update.
    double regularization = 1e-2;  ///< MOSSE: relative to mean energy; KCF: lambda.
    double kernel_sigma = 0.5;     ///< KCF Gaussian kernel bandwidth.
};

/// A correlation filter trained on and evaluated against fixed-size patches.
///
/// Patches are CV_32F grayscale images in [0, 1]. The response peak is at
/// (width/2, height/2) when the target is centred in the patch, so
/// displacement = peak - centre.
class CorrelationFilter {
public:
    virtual ~CorrelationFilter() = default;

    /// Train the filter from scratch on @p patch (defines the patch size).
    virtual void init(const cv::Mat& patch) = 0;

    /// Correlation response (CV_32F, same size as the patch).
    [[nodiscard]] virtual cv::Mat respond(const cv::Mat& patch) const = 0;

    /// Blend a new training sample into the model with the configured learning rate.
    virtual void update(const cv::Mat& patch) = 0;

    [[nodiscard]] virtual Algorithm algorithm() const noexcept = 0;
    [[nodiscard]] virtual cv::Size patch_size() const noexcept = 0;

protected:
    CorrelationFilter() = default;
    CorrelationFilter(const CorrelationFilter&) = default;
    CorrelationFilter& operator=(const CorrelationFilter&) = default;
    CorrelationFilter(CorrelationFilter&&) = default;
    CorrelationFilter& operator=(CorrelationFilter&&) = default;
};

/// Sensible defaults for each algorithm.
[[nodiscard]] FilterParams default_filter_params(Algorithm algorithm) noexcept;

/// Factory for the concrete filter implementations.
[[nodiscard]] std::unique_ptr<CorrelationFilter> make_filter(Algorithm algorithm,
                                                             const FilterParams& params);

}  // namespace vtt
