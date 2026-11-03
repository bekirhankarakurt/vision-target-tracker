#include "vtt/correlation_filter.hpp"

#include "vtt/kcf.hpp"
#include "vtt/mosse.hpp"

namespace vtt {

FilterParams default_filter_params(Algorithm algorithm) noexcept {
    FilterParams params;
    switch (algorithm) {
        case Algorithm::Mosse:
            params.output_sigma = 2.0;
            params.learning_rate = 0.125;
            params.regularization = 0.01;
            break;
        case Algorithm::Kcf:
            params.output_sigma = 2.5;
            params.learning_rate = 0.075;
            params.regularization = 1e-4;
            params.kernel_sigma = 0.2;
            break;
    }
    return params;
}

std::unique_ptr<CorrelationFilter> make_filter(Algorithm algorithm, const FilterParams& params) {
    switch (algorithm) {
        case Algorithm::Mosse: return std::make_unique<MosseFilter>(params);
        case Algorithm::Kcf: return std::make_unique<KcfFilter>(params);
    }
    return std::make_unique<KcfFilter>(params);
}

}  // namespace vtt
