// vtt_synth: render the deterministic synthetic test sequence and its ground truth.

#include "vtt/synthetic.hpp"

#include <opencv2/videoio.hpp>

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr const char* kUsage = R"(Usage: vtt_synth --out VIDEO.avi [--gt GT.csv] [options]

  --out PATH              Output video (MJPG .avi)
  --gt PATH               Ground-truth CSV (frame,x,y,w,h,visibility)
  --frames N              Number of frames (default 300)
  --size WxH              Frame size (default 640x480)
  --fps X                 Frame rate (default 30)
  --seed N                Texture/noise seed (default 7)
  --occlusion-start N     First occluded frame (default 120)
  --occlusion-hold N      Frames of full occlusion (default 10)
  --noise X               Sensor noise std-dev in gray levels (default 4)
)";

std::optional<long> to_long(const std::string& s) {
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    return (!s.empty() && end != nullptr && *end == '\0') ? std::optional<long>(v) : std::nullopt;
}

std::optional<double> to_double(const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    return (!s.empty() && end != nullptr && *end == '\0') ? std::optional<double>(v) : std::nullopt;
}

int fail(const std::string& message) {
    std::cerr << "error: " << message << "\n\n" << kUsage;
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::vector<std::string> args(argv + 1, argv + argc);
        vtt::SyntheticConfig config;
        std::string video_path;
        std::string gt_path;
        for (std::size_t i = 0; i < args.size(); ++i) {
            const std::string& flag = args[i];
            if (flag == "-h" || flag == "--help") {
                std::cout << kUsage;
                return 0;
            }
            if (i + 1 >= args.size()) {
                return fail("missing value for " + flag);
            }
            const std::string& value = args[++i];
            const auto integer = to_long(value);
            const auto real = to_double(value);
            if (flag == "--out") {
                video_path = value;
            } else if (flag == "--gt") {
                gt_path = value;
            } else if (flag == "--frames" && integer && *integer > 0) {
                config.num_frames = static_cast<int>(*integer);
            } else if (flag == "--fps" && real && *real > 0.0) {
                config.fps = *real;
            } else if (flag == "--seed" && integer && *integer >= 0) {
                config.seed = static_cast<std::uint64_t>(*integer);
            } else if (flag == "--occlusion-start" && integer) {
                config.occlusion_start = static_cast<int>(*integer);
            } else if (flag == "--occlusion-hold" && integer && *integer >= 0) {
                config.occlusion_hold = static_cast<int>(*integer);
            } else if (flag == "--noise" && real && *real >= 0.0) {
                config.noise_std = *real;
            } else if (flag == "--size") {
                const std::size_t x = value.find('x');
                const auto w = x == std::string::npos ? std::nullopt : to_long(value.substr(0, x));
                const auto h = x == std::string::npos ? std::nullopt : to_long(value.substr(x + 1));
                if (!w || !h || *w < 64 || *h < 64) {
                    return fail("invalid size: " + value);
                }
                config.frame_size = cv::Size(static_cast<int>(*w), static_cast<int>(*h));
            } else {
                return fail("invalid option or value: " + flag);
            }
        }
        if (video_path.empty()) {
            return fail("--out is required");
        }

        const vtt::SyntheticScene scene(config);
        cv::VideoWriter writer(video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), config.fps,
                               config.frame_size);
        if (!writer.isOpened()) {
            return fail("cannot open video writer for " + video_path);
        }
        writer.set(cv::VIDEOWRITER_PROP_QUALITY, 95.0);
        std::ofstream gt;
        if (!gt_path.empty()) {
            gt.open(gt_path);
            if (!gt) {
                return fail("cannot open " + gt_path);
            }
            gt.setf(std::ios::fixed);
            gt.precision(3);
            gt << "frame,x,y,w,h,visibility\n";
        }
        for (int f = 0; f < config.num_frames; ++f) {
            writer.write(scene.render(f));
            if (gt.is_open()) {
                const vtt::GroundTruth truth = scene.ground_truth(f);
                gt << f << ',' << truth.box.x << ',' << truth.box.y << ',' << truth.box.width << ','
                   << truth.box.height << ',' << truth.visibility << '\n';
            }
        }
        const cv::Rect2d first = scene.ground_truth(0).box;
        std::cout << "wrote " << config.num_frames << " frames to " << video_path
                  << "\ninitial ROI: " << cvRound(first.x) << ',' << cvRound(first.y) << ','
                  << cvRound(first.width) << ',' << cvRound(first.height) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
