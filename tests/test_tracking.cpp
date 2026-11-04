#include "test_harness.hpp"

#include "vtt/metrics.hpp"
#include "vtt/synthetic.hpp"
#include "vtt/tracker.hpp"

#include <iomanip>
#include <iostream>

using namespace vtt;

namespace {

struct RunSummary {
    AccuracyStats accuracy;
    AccuracyStats visible_accuracy;
    int occluded_frames = 0;
    int lost_frames = 0;
    int tracking_after_occlusion = 0;
    bool reacquired = false;
};

RunSummary run_sequence(Algorithm algorithm, const SyntheticConfig& config) {
    const SyntheticScene scene(config);
    Tracker tracker(default_tracker_config(algorithm));
    tracker.init(scene.render(0), scene.ground_truth(0).box);
    RunSummary summary;
    const int occlusion_end = config.occlusion_start + 2 * config.occlusion_ramp + config.occlusion_hold;
    bool lost_seen = false;
    for (int f = 1; f < config.num_frames; ++f) {
        const TrackOutput out = tracker.update(scene.render(f));
        const GroundTruth gt = scene.ground_truth(f);
        summary.accuracy.add(out.box, gt.box);
        if (gt.visibility > 0.99) {
            summary.visible_accuracy.add(out.box, gt.box);
        }
        summary.occluded_frames += out.state == TrackState::Occluded ? 1 : 0;
        summary.lost_frames += out.state == TrackState::Lost ? 1 : 0;
        lost_seen = lost_seen || out.state == TrackState::Lost;
        if (f > occlusion_end + 15 && out.state == TrackState::Tracking &&
            iou(out.box, gt.box) > 0.5) {
            ++summary.tracking_after_occlusion;
            summary.reacquired = summary.reacquired || lost_seen;
        }
    }
    std::cout << std::fixed << std::setprecision(3) << "  " << to_string(algorithm) << ": IoU "
              << summary.accuracy.mean_iou() << "  CE " << summary.accuracy.mean_center_error()
              << " px  success " << summary.accuracy.success_rate() << "  (visible IoU "
              << summary.visible_accuracy.mean_iou() << ")  occluded " << summary.occluded_frames
              << "  lost " << summary.lost_frames << '\n';
    return summary;
}

void check_default_sequence(Algorithm algorithm) {
    const SyntheticConfig config;
    const RunSummary s = run_sequence(algorithm, config);
    const int occlusion_end = config.occlusion_start + 2 * config.occlusion_ramp + config.occlusion_hold;
    CHECK(s.accuracy.mean_iou() > 0.65);
    CHECK(s.accuracy.mean_center_error() < 8.0);
    CHECK(s.accuracy.success_rate() > 0.85);
    CHECK(s.visible_accuracy.mean_iou() > 0.7);
    CHECK(s.occluded_frames >= config.occlusion_hold / 2);
    CHECK(s.tracking_after_occlusion >= (config.num_frames - occlusion_end - 16) * 9 / 10);
}

}  // namespace

VTT_TEST(mosse_tracks_synthetic_sequence) { check_default_sequence(Algorithm::Mosse); }

VTT_TEST(kcf_tracks_synthetic_sequence) { check_default_sequence(Algorithm::Kcf); }

VTT_TEST(long_occlusion_goes_lost_and_reacquires) {
    SyntheticConfig config;
    config.occlusion_hold = 35;  // target changes scale by ~25 % while hidden
    for (const Algorithm algorithm : {Algorithm::Mosse, Algorithm::Kcf}) {
        const RunSummary s = run_sequence(algorithm, config);
        CHECK(s.lost_frames > 0);
        CHECK(s.reacquired);
        CHECK(s.tracking_after_occlusion > 80);
        CHECK(s.accuracy.mean_iou() > 0.7);
    }
}

VTT_TEST(scale_estimation_follows_target_size) {
    SyntheticConfig config;
    config.num_frames = 150;
    config.occlusion_start = 10000;
    config.scale_period = 150.0;  // scale peaks at frame ~37
    const SyntheticScene scene(config);
    Tracker tracker(default_tracker_config(Algorithm::Kcf));
    tracker.init(scene.render(0), scene.ground_truth(0).box);
    TrackOutput out;
    for (int f = 1; f <= 37; ++f) {
        out = tracker.update(scene.render(f));
    }
    const double true_scale = scene.ground_truth(37).box.width / scene.ground_truth(0).box.width;
    CHECK(true_scale > 1.25);
    CHECK_NEAR(out.scale, true_scale, 0.12);
}

VTT_TEST(tracker_reports_idle_before_init_and_rejects_bad_box) {
    Tracker tracker(default_tracker_config(Algorithm::Kcf));
    const cv::Mat frame(120, 160, CV_8UC3, cv::Scalar::all(80));
    CHECK(tracker.update(frame).state == TrackState::Idle);
    bool threw = false;
    try {
        tracker.init(frame, cv::Rect2d(10, 10, 1, 1));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    tracker.init(frame, cv::Rect2d(150, 110, 30, 30));  // partly outside the frame
    const TrackOutput out = tracker.update(frame);
    CHECK(out.state != TrackState::Idle);
    tracker.reset();
    CHECK(tracker.state() == TrackState::Idle);
}

VTT_TEST(synthetic_scene_is_deterministic) {
    const SyntheticScene a;
    const SyntheticScene b;
    CHECK(cv::norm(a.render(42), b.render(42), cv::NORM_INF) == 0.0);
    const GroundTruth gt = a.ground_truth(0);
    CHECK(gt.visibility > 0.99);
    const SyntheticConfig& c = a.config();
    const GroundTruth hidden = a.ground_truth(c.occlusion_start + c.occlusion_ramp + c.occlusion_hold / 2);
    CHECK(hidden.visibility < 0.01);
}

VTT_TEST_MAIN()
