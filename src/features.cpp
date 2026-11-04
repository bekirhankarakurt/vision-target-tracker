#include "vtt/features.hpp"

#include <opencv2/imgproc.hpp>

namespace vtt::features {

cv::Mat mosse_preprocess(const cv::Mat& patch, const cv::Mat& window) {
    CV_Assert(patch.type() == CV_32F && patch.size() == window.size());
    cv::Mat out;
    cv::log(patch * 255.0 + 1.0, out);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(out, mean, stddev);
    out = (out - mean[0]) / (stddev[0] + 1e-5);
    return out.mul(window);
}

std::vector<cv::Mat> kcf_channels(const cv::Mat& patch, const cv::Mat& window) {
    CV_Assert(patch.type() == CV_32F && patch.size() == window.size());
    std::vector<cv::Mat> channels(3);
    const cv::Scalar mean = cv::mean(patch);
    channels[0] = (patch - mean[0]).mul(window);
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(patch, gx, CV_32F, 1, 0, 3, 0.25, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(patch, gy, CV_32F, 0, 1, 3, 0.25, 0.0, cv::BORDER_REPLICATE);
    channels[1] = gx.mul(window);
    channels[2] = gy.mul(window);
    return channels;
}

}  // namespace vtt::features
