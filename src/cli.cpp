#include "vtt/cli.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <sstream>

namespace vtt {

namespace {

std::optional<double> to_double(const std::string& text) {
    if (text.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end == nullptr || *end != '\0') {
        return std::nullopt;
    }
    return value;
}

std::optional<long> to_long(const std::string& text) {
    if (text.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0') {
        return std::nullopt;
    }
    return value;
}

bool takes_value(const std::string& flag) {
    static const std::array<const char*, 18> flags{
        "--input", "-i", "--camera", "--algo", "--roi", "--gt", "--csv", "--output",
        "--max-frames", "--psr-threshold", "--hfov", "--pid-kp", "--pid-ki", "--pid-kd",
        "--max-rate", "--max-accel", "--min-iou", "--max-center-error"};
    return std::any_of(flags.begin(), flags.end(), [&flag](const char* f) { return flag == f; });
}

}  // namespace

std::optional<cv::Rect2d> parse_rect(const std::string& text) {
    std::array<double, 4> v{};
    std::istringstream in(text);
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (!(in >> v[i])) {
            return std::nullopt;
        }
        if (i + 1 < v.size()) {
            char comma = 0;
            if (!(in >> comma) || comma != ',') {
                return std::nullopt;
            }
        }
    }
    std::string rest;
    if (in >> rest || v[2] <= 0.0 || v[3] <= 0.0) {
        return std::nullopt;
    }
    return cv::Rect2d(v[0], v[1], v[2], v[3]);
}

ParseResult parse_arguments(const std::vector<std::string>& args) {
    ParseResult result;
    AppOptions& o = result.options;
    auto fail = [&result](const std::string& message) {
        result.error = message;
        return result;
    };
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string flag = args[i];
        std::string value;
        const std::size_t eq = flag.find('=');
        const bool inline_value = flag.rfind("--", 0) == 0 && eq != std::string::npos;
        if (inline_value) {
            value = flag.substr(eq + 1);
            flag.resize(eq);
        }
        if (takes_value(flag) && !inline_value) {
            if (i + 1 >= args.size()) {
                return fail("missing value for " + flag);
            }
            value = args[++i];
        } else if (!takes_value(flag) && inline_value) {
            return fail("option " + flag + " does not take a value");
        }

        const auto number = [&]() { return to_double(value); };
        if (flag == "--help" || flag == "-h") {
            o.show_help = true;
        } else if (flag == "--version") {
            o.show_version = true;
        } else if (flag == "--headless") {
            o.headless = true;
        } else if (flag == "--no-scale") {
            o.disable_scale = true;
        } else if (flag == "--input" || flag == "-i") {
            o.input = value;
        } else if (flag == "--camera") {
            const auto index = to_long(value);
            if (!index || *index < 0) {
                return fail("invalid camera index: " + value);
            }
            o.camera_index = static_cast<int>(*index);
        } else if (flag == "--algo") {
            const auto algorithm = parse_algorithm(value);
            if (!algorithm) {
                return fail("unknown algorithm: " + value + " (expected mosse or kcf)");
            }
            o.algorithm = *algorithm;
        } else if (flag == "--roi") {
            o.roi = parse_rect(value);
            if (!o.roi) {
                return fail("invalid ROI '" + value + "' (expected x,y,w,h)");
            }
        } else if (flag == "--gt") {
            o.ground_truth = value;
        } else if (flag == "--csv") {
            o.csv_path = value;
        } else if (flag == "--output") {
            o.output_video = value;
        } else if (flag == "--max-frames") {
            const auto frames = to_long(value);
            if (!frames || *frames < 0) {
                return fail("invalid frame count: " + value);
            }
            o.max_frames = *frames;
        } else if (flag == "--psr-threshold") {
            if (!number() || *number() <= 0.0) {
                return fail("invalid PSR threshold: " + value);
            }
            o.psr_threshold = number();
        } else if (flag == "--hfov") {
            if (!number() || *number() <= 0.0 || *number() >= 180.0) {
                return fail("invalid horizontal FOV: " + value);
            }
            o.gimbal.horizontal_fov_deg = *number();
        } else if (flag == "--pid-kp" || flag == "--pid-ki" || flag == "--pid-kd") {
            if (!number() || *number() < 0.0) {
                return fail("invalid gain: " + value);
            }
            double PidConfig::*gain = &PidConfig::kd;
            if (flag == "--pid-kp") {
                gain = &PidConfig::kp;
            } else if (flag == "--pid-ki") {
                gain = &PidConfig::ki;
            }
            o.gimbal.pan_pid.*gain = *number();
            o.gimbal.tilt_pid.*gain = *number();
        } else if (flag == "--max-rate") {
            if (!number() || *number() <= 0.0) {
                return fail("invalid rate limit: " + value);
            }
            o.gimbal.pan_limits.max_rate_dps = *number();
            o.gimbal.tilt_limits.max_rate_dps = *number();
            o.gimbal.pan_pid.output_limit = *number();
            o.gimbal.tilt_pid.output_limit = *number();
        } else if (flag == "--max-accel") {
            if (!number() || *number() <= 0.0) {
                return fail("invalid acceleration limit: " + value);
            }
            o.gimbal.pan_limits.max_accel_dps2 = *number();
            o.gimbal.tilt_limits.max_accel_dps2 = *number();
        } else if (flag == "--min-iou") {
            if (!number() || *number() < 0.0 || *number() > 1.0) {
                return fail("invalid IoU threshold: " + value);
            }
            o.min_iou = number();
        } else if (flag == "--max-center-error") {
            if (!number() || *number() <= 0.0) {
                return fail("invalid centre error threshold: " + value);
            }
            o.max_center_error = number();
        } else {
            return fail("unknown option: " + flag);
        }
    }
    if (o.headless && !o.roi && !o.ground_truth) {
        return fail("--headless requires --roi or --gt to initialise the target");
    }
    return result;
}

std::string usage() {
    return R"(Usage: vtt [options]

Input
  -i, --input PATH          Video file (default: webcam)
      --camera N            Webcam index (default: 0)
      --max-frames N        Stop after N frames (0 = all)

Tracker
      --algo mosse|kcf      Correlation filter (default: kcf)
      --roi x,y,w,h         Initial target box (default: select with the mouse)
      --no-scale            Disable the scale pyramid
      --psr-threshold X     Override the PSR tracking threshold

Gimbal simulation
      --hfov DEG            Scene camera horizontal FOV (default: 60)
      --pid-kp/--pid-ki/--pid-kd X   PID gains for both axes (default: 10 / 20 / 0.1)
      --max-rate DPS        Axis rate limit (default: 120)
      --max-accel DPS2      Axis acceleration limit (default: 600)

Output
      --headless            No window; requires --roi or --gt
      --csv PATH            Per-frame telemetry CSV
      --output PATH         Write the overlay video (MJPG .avi)

Evaluation
      --gt PATH             Ground-truth CSV (frame,x,y,w,h); also initialises the ROI
      --min-iou X           Exit with code 2 if mean IoU < X
      --max-center-error PX Exit with code 2 if mean centre error > PX

  -h, --help                Show this help
      --version             Show the version
)";
}

}  // namespace vtt
