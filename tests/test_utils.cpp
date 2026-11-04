#include "test_harness.hpp"

#include "vtt/cli.hpp"
#include "vtt/csv.hpp"
#include "vtt/metrics.hpp"
#include "vtt/overlay.hpp"
#include "vtt/types.hpp"

#include <cstdio>
#include <fstream>
#include <string>

using namespace vtt;

VTT_TEST(iou_and_center_error) {
    const cv::Rect2d a(0, 0, 10, 10);
    CHECK_NEAR(iou(a, a), 1.0, 1e-12);
    CHECK_NEAR(iou(a, cv::Rect2d(5, 0, 10, 10)), 50.0 / 150.0, 1e-12);
    CHECK_NEAR(iou(a, cv::Rect2d(20, 20, 5, 5)), 0.0, 1e-12);
    CHECK_NEAR(iou(cv::Rect2d(), cv::Rect2d()), 0.0, 1e-12);
    CHECK_NEAR(center_error(a, cv::Rect2d(3, 4, 10, 10)), 5.0, 1e-12);
}

VTT_TEST(accuracy_stats_aggregate) {
    AccuracyStats stats;
    CHECK_NEAR(stats.mean_iou(), 0.0, 1e-12);
    stats.add(cv::Rect2d(0, 0, 10, 10), cv::Rect2d(0, 0, 10, 10));
    stats.add(cv::Rect2d(0, 0, 10, 10), cv::Rect2d(100, 0, 10, 10));
    CHECK(stats.count() == 2U);
    CHECK_NEAR(stats.mean_iou(), 0.5, 1e-12);
    CHECK_NEAR(stats.mean_center_error(), 50.0, 1e-12);
    CHECK_NEAR(stats.success_rate(), 0.5, 1e-12);
    CHECK_NEAR(stats.precision_20px(), 0.5, 1e-12);
}

VTT_TEST(latency_percentiles) {
    LatencyStats stats;
    CHECK_NEAR(stats.percentile(50.0), 0.0, 1e-12);
    for (int i = 1; i <= 100; ++i) {
        stats.add(static_cast<double>(i));
    }
    CHECK_NEAR(stats.mean(), 50.5, 1e-12);
    CHECK_NEAR(stats.percentile(50.0), 50.0, 1e-12);
    CHECK_NEAR(stats.percentile(95.0), 95.0, 1e-12);
    CHECK_NEAR(stats.percentile(0.0), 1.0, 1e-12);
    CHECK_NEAR(stats.max(), 100.0, 1e-12);
}

VTT_TEST(rate_meter_smooths) {
    RateMeter meter(0.5);
    meter.tick(0.0);
    CHECK_NEAR(meter.rate(), 0.0, 1e-12);
    meter.tick(0.1);
    CHECK_NEAR(meter.rate(), 10.0, 1e-9);
    meter.tick(0.05);
    CHECK_NEAR(meter.rate(), 15.0, 1e-9);
}

VTT_TEST(enum_strings_roundtrip) {
    CHECK(to_string(Algorithm::Mosse) == "MOSSE");
    CHECK(to_string(TrackState::Lost) == "LOST");
    CHECK(to_string(TrackState::Idle) == "IDLE");
    CHECK(parse_algorithm("KcF") == Algorithm::Kcf);
    CHECK(parse_algorithm("mosse") == Algorithm::Mosse);
    CHECK(!parse_algorithm("csrt"));
}

VTT_TEST(cli_parses_full_command_line) {
    const ParseResult r = parse_arguments({"-i", "video.avi", "--algo=mosse", "--roi", "10,20,30,40",
                                           "--headless", "--csv", "out.csv", "--max-frames", "50",
                                           "--pid-kp", "3.5", "--max-rate", "90", "--hfov", "45",
                                           "--min-iou", "0.6", "--no-scale"});
    CHECK(r.ok());
    const AppOptions& o = r.options;
    CHECK(o.input && *o.input == "video.avi");
    CHECK(o.algorithm == Algorithm::Mosse);
    CHECK(o.roi && o.roi->x == 10.0 && o.roi->height == 40.0);
    CHECK(o.headless && o.disable_scale);
    CHECK(o.max_frames == 50);
    CHECK_NEAR(o.gimbal.pan_pid.kp, 3.5, 1e-12);
    CHECK_NEAR(o.gimbal.tilt_pid.kp, 3.5, 1e-12);
    CHECK_NEAR(o.gimbal.tilt_limits.max_rate_dps, 90.0, 1e-12);
    CHECK_NEAR(o.gimbal.horizontal_fov_deg, 45.0, 1e-12);
    CHECK(o.min_iou && *o.min_iou == 0.6);
}

VTT_TEST(cli_rejects_invalid_input) {
    CHECK(!parse_arguments({"--algo", "csrt"}).ok());
    CHECK(!parse_arguments({"--roi", "1,2,3"}).ok());
    CHECK(!parse_arguments({"--roi", "1,2,0,4"}).ok());
    CHECK(!parse_arguments({"--camera"}).ok());
    CHECK(!parse_arguments({"--bogus"}).ok());
    CHECK(!parse_arguments({"--headless"}).ok());
    CHECK(!parse_arguments({"--headless=1"}).ok());
    CHECK(!parse_arguments({"--hfov", "200"}).ok());
    CHECK(!parse_arguments({"--min-iou", "x"}).ok());
    CHECK(parse_arguments({"--help"}).options.show_help);
    CHECK(!usage().empty());
    CHECK(parse_rect("1.5,2,3,4").has_value());
    CHECK(!parse_rect("1,2,3,4,5").has_value());
}

VTT_TEST(ground_truth_loads_and_csv_writes) {
    const std::string gt_path = "vtt_test_gt.csv";
    {
        std::ofstream out(gt_path);
        out << "frame,x,y,w,h,visibility\n0,1,2,3,4,1\n1,5.5,6,7,8,0.5\n";
    }
    const auto rows = load_ground_truth(gt_path);
    CHECK(rows.size() == 2U);
    CHECK(rows[1].frame == 1 && rows[1].box.x == 5.5 && rows[1].box.height == 8.0);
    {
        std::ofstream bad(gt_path);
        bad << "0;1;2;3;4\n";
    }
    bool threw = false;
    try {
        static_cast<void>(load_ground_truth(gt_path));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    std::remove(gt_path.c_str());

    const std::string csv_path = "vtt_test_out.csv";
    {
        CsvWriter writer(csv_path);
        TrackOutput track;
        track.state = TrackState::Tracking;
        track.box = cv::Rect2d(1, 2, 3, 4);
        writer.write(0, 1.5, track, GimbalState{}, cv::Rect2d(1, 2, 3, 4));
        writer.write(1, 1.5, track, GimbalState{}, std::nullopt);
    }
    std::ifstream in(csv_path);
    std::string header;
    std::string row;
    std::getline(in, header);
    std::getline(in, row);
    CHECK(header == CsvWriter::header());
    CHECK(row.find("TRACKING") != std::string::npos);
    CHECK(row.find(",1.000,0.000") != std::string::npos);
    std::remove(csv_path.c_str());
}

VTT_TEST(overlay_draws_without_touching_size) {
    cv::Mat canvas(480, 640, CV_8UC3, cv::Scalar::all(50));
    GimbalSim gimbal(GimbalConfig{}, canvas.size());
    static_cast<void>(gimbal.step(cv::Point2d(400.0, 200.0), 0.1));
    for (const TrackState state :
         {TrackState::Idle, TrackState::Tracking, TrackState::Occluded, TrackState::Lost}) {
        TrackOutput track;
        track.state = state;
        track.box = cv::Rect2d(300, 200, 40, 30);
        track.search_region = cv::Rect2d(250, 150, 140, 130);
        track.predicted = cv::Point2d(320, 215);
        cv::Mat image = canvas.clone();
        draw_overlay(image, track, gimbal, Telemetry{}, OverlayOptions{true, true});
        CHECK(image.size() == canvas.size());
        CHECK(cv::norm(image, canvas, cv::NORM_L1) > 0.0);
    }
    CHECK(state_color(TrackState::Tracking) != state_color(TrackState::Lost));
}

VTT_TEST_MAIN()
