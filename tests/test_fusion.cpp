#include "test_harness.hpp"

#include "vtt/fusion.hpp"
#include "vtt/udp.hpp"

#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace vtt;

namespace {

Detection det(long frame, double x, double y, double w, double h, double conf = 0.9) {
    Detection d;
    d.frame_id = frame;
    d.box = cv::Rect2d(x, y, w, h);
    d.confidence = conf;
    return d;
}

/// A fusion whose locked target at (100,100,50,150) is already confirmed by the detector.
DetectionFusion backed_fusion(bool auto_lock = false) {
    FusionConfig config;
    config.auto_lock = auto_lock;
    DetectionFusion fusion(config);
    fusion.on_target_changed(true);
    for (long f = 0; f <= 3; ++f) {
        fusion.record(f, TrackState::Tracking, cv::Rect2d(100, 100, 50, 150));
    }
    CHECK(fusion.on_detection(det(0, 102, 98, 50, 150), 3).action == FusionAction::None);
    CHECK(fusion.detector_backed());
    return fusion;
}

}  // namespace

VTT_TEST(frame_header_round_trip_and_magic) {
    FrameHeader header;
    header.frame_id = 123456;
    header.width = 1280;
    header.height = 720;
    const std::vector<std::uint8_t> bytes = encode_frame_header(header);
    CHECK(bytes.size() == kFrameHeaderSize);
    CHECK(bytes[0] == 'V' && bytes[3] == '1');
    const auto back = decode_frame_header(bytes);
    CHECK(back && back->frame_id == 123456U && back->width == 1280U && back->height == 720U);

    std::vector<std::uint8_t> bad = bytes;
    bad[0] = 'X';
    CHECK(!decode_frame_header(bad));
    CHECK(!decode_frame_header(std::vector<std::uint8_t>(bytes.begin(), bytes.begin() + 11)));
}

VTT_TEST(detection_lines_parse_and_reject) {
    const auto d = parse_detection_line("DET 42 10.5 20 30 40 0.87\n");
    CHECK(d && d->frame_id == 42 && d->box && d->box->x == 10.5 && d->box->height == 40.0);
    CHECK(d && d->confidence > 0.86 && d->confidence < 0.88);
    CHECK(d && !d->infer_ms);
    const auto none = parse_detection_line("NONE 7\n");
    CHECK(none && none->frame_id == 7 && !none->box && !none->infer_ms);

    const auto timed = parse_detection_line("DET 42 10.5 20 30 40 0.87 12.5\n");
    CHECK(timed && timed->frame_id == 42 && timed->box && timed->infer_ms);
    if (timed && timed->infer_ms) {
        CHECK_NEAR(*timed->infer_ms, 12.5, 1e-9);
    }
    const auto none_ms = parse_detection_line("NONE 7 3.5");
    CHECK(none_ms && none_ms->frame_id == 7 && !none_ms->box && none_ms->infer_ms);
    if (none_ms && none_ms->infer_ms) {
        CHECK_NEAR(*none_ms->infer_ms, 3.5, 1e-9);
    }
    const auto zero_ms = parse_detection_line("NONE 4 0\n");
    CHECK(zero_ms && zero_ms->infer_ms);
    if (zero_ms && zero_ms->infer_ms) {
        CHECK_NEAR(*zero_ms->infer_ms, 0.0, 1e-12);
    }

    CHECK(!parse_detection_line(""));
    CHECK(!parse_detection_line("DET 1 2 3 4"));
    CHECK(!parse_detection_line("DET 1 2 3 0 4 0.5"));
    CHECK(!parse_detection_line("DET 1 2 3 4 5 1.5"));
    CHECK(!parse_detection_line("DET 1 nan 3 4 5 0.5"));
    CHECK(!parse_detection_line("DET -1 2 3 4 5 0.5"));
    CHECK(!parse_detection_line("DET 1 2 3 4 5 0.5 extra"));
    CHECK(!parse_detection_line("DET 1 2 3 4 5 0.5 -1"));
    CHECK(!parse_detection_line("DET 1 2 3 4 5 0.5 -0.01"));
    CHECK(!parse_detection_line("DET 1 2 3 4 5 0.5 nan"));
    CHECK(!parse_detection_line("DET 1 2 3 4 5 0.5 inf"));
    CHECK(!parse_detection_line("DET 1 2 3 4 5 0.5 8 9"));
    CHECK(!parse_detection_line("NONE 3 extra"));
    CHECK(!parse_detection_line("NONE 3 -4"));
    CHECK(!parse_detection_line("NONE 3 nan"));
    CHECK(!parse_detection_line("NONE 3 1.5 extra"));
    CHECK(!parse_detection_line("HELLO 3"));
}

VTT_TEST(drifted_backed_target_reanchors_after_two_bad_detections) {
    DetectionFusion fusion = backed_fusion();
    // Tracker stays on the background while the object moved right.
    fusion.record(4, TrackState::Tracking, cv::Rect2d(100, 100, 50, 150));
    fusion.record(5, TrackState::Tracking, cv::Rect2d(100, 100, 50, 150));
    CHECK(fusion.on_detection(det(4, 400, 100, 50, 150), 5).action == FusionAction::None);
    const FusionDecision second = fusion.on_detection(det(5, 410, 100, 50, 150), 6);
    CHECK(second.action == FusionAction::Reanchor);
    CHECK(second.box.x == 410.0);
}

VTT_TEST(one_bad_detection_or_a_miss_does_not_reanchor) {
    DetectionFusion fusion = backed_fusion();
    fusion.record(4, TrackState::Tracking, cv::Rect2d(100, 100, 50, 150));
    fusion.record(5, TrackState::Tracking, cv::Rect2d(100, 100, 50, 150));
    fusion.record(6, TrackState::Tracking, cv::Rect2d(100, 100, 50, 150));
    CHECK(fusion.on_detection(det(4, 400, 100, 50, 150), 6).action == FusionAction::None);
    // A good detection in between resets the count.
    CHECK(fusion.on_detection(det(5, 101, 100, 50, 150), 6).action == FusionAction::None);
    CHECK(fusion.on_detection(det(6, 400, 100, 50, 150), 6).action == FusionAction::None);
    // Misses and low confidence are not evidence of drift.
    Detection miss;
    miss.frame_id = 6;
    CHECK(fusion.on_detection(miss, 6).action == FusionAction::None);
    CHECK(fusion.on_detection(det(6, 400, 100, 50, 150, 0.2), 6).action == FusionAction::None);
}

VTT_TEST(user_target_unknown_to_detector_is_never_overridden) {
    DetectionFusion fusion;
    fusion.on_target_changed(true);  // The user clicked a cup.
    for (long f = 0; f < 6; ++f) {
        fusion.record(f, TrackState::Tracking, cv::Rect2d(10, 10, 40, 40));
        CHECK(fusion.on_detection(det(f, 400, 100, 50, 150), f).action == FusionAction::None);
    }
    CHECK(!fusion.detector_backed());
}

VTT_TEST(small_drift_is_refined_with_motion_compensation) {
    DetectionFusion fusion = backed_fusion();
    // Tracker box at frame 4, then it moved +20 px in x by frame 6.
    fusion.record(4, TrackState::Tracking, cv::Rect2d(100, 100, 50, 150));
    fusion.record(5, TrackState::Tracking, cv::Rect2d(110, 100, 50, 150));
    fusion.record(6, TrackState::Tracking, cv::Rect2d(120, 100, 50, 150));
    // Detection of frame 4 says the object was 15 px further right (IoU ~0.54).
    const FusionDecision refine = fusion.on_detection(det(4, 115, 100, 50, 150), 6);
    CHECK(refine.action == FusionAction::Refine);
    CHECK(refine.box.x > 134.9 && refine.box.x < 135.1);  // 115 + 20 px of tracker motion
    // A detection that already agrees closely changes nothing.
    CHECK(fusion.on_detection(det(6, 121, 100, 50, 150), 6).action == FusionAction::None);
    // An old detection is not used for refining.
    fusion.record(20, TrackState::Tracking, cv::Rect2d(120, 100, 50, 150));
    CHECK(fusion.on_detection(det(6, 135, 100, 50, 150), 20).action == FusionAction::None);
}

VTT_TEST(lost_backed_target_reanchors_at_once) {
    DetectionFusion fusion = backed_fusion();
    fusion.record(4, TrackState::Lost, cv::Rect2d(100, 100, 50, 150));
    const FusionDecision decision = fusion.on_detection(det(4, 300, 120, 50, 150), 4);
    CHECK(decision.action == FusionAction::Reanchor);
}

VTT_TEST(auto_lock_needs_two_consistent_detections) {
    FusionConfig config;
    config.auto_lock = true;
    DetectionFusion fusion(config);
    for (long f = 0; f < 4; ++f) {
        fusion.record(f, TrackState::Idle, cv::Rect2d());
    }
    CHECK(fusion.on_detection(det(0, 200, 100, 50, 150), 1).action == FusionAction::None);
    // Far away: not the same object, the count restarts.
    CHECK(fusion.on_detection(det(1, 500, 100, 50, 150), 2).action == FusionAction::None);
    const FusionDecision lock = fusion.on_detection(det(2, 505, 102, 50, 150), 3);
    CHECK(lock.action == FusionAction::AutoLock);
    fusion.on_target_changed(true, true);
    CHECK(fusion.detector_backed());

    DetectionFusion manual;  // auto-lock off
    CHECK(manual.on_detection(det(0, 1, 1, 50, 50), 0).action == FusionAction::None);
    CHECK(manual.on_detection(det(0, 1, 1, 50, 50), 0).action == FusionAction::None);
}

VTT_TEST(stale_and_future_detections_are_ignored) {
    DetectionFusion fusion = backed_fusion();
    fusion.record(40, TrackState::Lost, cv::Rect2d(100, 100, 50, 150));
    CHECK(fusion.on_detection(det(3, 400, 100, 50, 150), 40).action == FusionAction::None);
    CHECK(fusion.on_detection(det(41, 400, 100, 50, 150), 40).action == FusionAction::None);
}

VTT_TEST(endpoint_parsing) {
    const auto e = parse_endpoint("127.0.0.1:5601");
    CHECK(e && e->host == "127.0.0.1" && e->port == 5601);
    CHECK(!parse_endpoint("127.0.0.1"));
    CHECK(!parse_endpoint("127.0.0.1:0"));
    CHECK(!parse_endpoint("127.0.0.1:70000"));
    CHECK(!parse_endpoint("localhost:5601"));
    CHECK(!parse_endpoint(":5601"));
}

VTT_TEST(udp_loopback_round_trip) {
    UdpSocket listener(static_cast<std::uint16_t>(56021));
    UdpSocket sender;
    CHECK(!listener.try_receive());
    const std::string line = "NONE 9\n";
    CHECK(sender.send_to("127.0.0.1", 56021, std::vector<std::uint8_t>(line.begin(), line.end())));
    std::optional<std::vector<std::uint8_t>> got;
    for (int i = 0; i < 200 && !got; ++i) {
        got = listener.try_receive();
        if (!got) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    CHECK(got && std::string(got->begin(), got->end()) == line);
}

VTT_TEST_MAIN()
