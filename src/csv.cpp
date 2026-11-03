#include "vtt/csv.hpp"

#include "vtt/metrics.hpp"

#include <sstream>
#include <stdexcept>

namespace vtt {

std::vector<GroundTruthRow> load_ground_truth(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open ground truth file: " + path);
    }
    std::vector<GroundTruthRow> rows;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#' || (line_number == 1 && line.rfind("frame", 0) == 0)) {
            continue;
        }
        std::istringstream fields(line);
        GroundTruthRow row;
        char c1 = 0;
        char c2 = 0;
        char c3 = 0;
        char c4 = 0;
        if (!(fields >> row.frame >> c1 >> row.box.x >> c2 >> row.box.y >> c3 >> row.box.width >> c4 >>
              row.box.height) ||
            c1 != ',' || c2 != ',' || c3 != ',' || c4 != ',') {
            throw std::runtime_error("malformed ground truth at " + path + ":" +
                                     std::to_string(line_number));
        }
        rows.push_back(row);
    }
    return rows;
}

CsvWriter::CsvWriter(const std::string& path) : out_(path) {
    if (!out_) {
        throw std::runtime_error("cannot open CSV output: " + path);
    }
    out_.setf(std::ios::fixed);
    out_.precision(3);
    out_ << header() << '\n';
}

std::string CsvWriter::header() {
    return "frame,latency_ms,state,x,y,w,h,cx,cy,psr,scale,pred_x,pred_y,vx,vy,"
           "pan_deg,tilt_deg,pan_rate_dps,tilt_rate_dps,pan_err_deg,tilt_err_deg,"
           "gt_x,gt_y,gt_w,gt_h,iou,center_err";
}

void CsvWriter::write(long frame, double latency_ms, const TrackOutput& track,
                      const GimbalState& gimbal, const std::optional<cv::Rect2d>& truth) {
    const cv::Rect2d& b = track.box;
    out_ << frame << ',' << latency_ms << ',' << to_string(track.state) << ',' << b.x << ',' << b.y
         << ',' << b.width << ',' << b.height << ',' << b.x + 0.5 * b.width << ','
         << b.y + 0.5 * b.height << ',' << track.psr << ',' << track.scale << ','
         << track.predicted.x << ',' << track.predicted.y << ',' << track.velocity.x << ','
         << track.velocity.y << ',' << gimbal.pan_deg << ',' << gimbal.tilt_deg << ','
         << gimbal.pan_rate_dps << ',' << gimbal.tilt_rate_dps << ',' << gimbal.pan_error_deg
         << ',' << gimbal.tilt_error_deg;
    if (truth) {
        out_ << ',' << truth->x << ',' << truth->y << ',' << truth->width << ',' << truth->height
             << ',' << iou(b, *truth) << ',' << center_error(b, *truth);
    } else {
        out_ << ",,,,,,";
    }
    out_ << '\n';
}

}  // namespace vtt
