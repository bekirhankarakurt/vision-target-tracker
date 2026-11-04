# vision-target-tracker API

Real-time visual target tracker: correlation filters (MOSSE, Gaussian-kernel KCF),
a 1-D scale filter, PSR confidence, constant-velocity Kalman prediction,
lost/reacquire logic and a PID-driven 2-axis gimbal simulation.

## Entry points

| Component | Header | Purpose |
|---|---|---|
| `vtt::Tracker` | `vtt/tracker.hpp` | Full pipeline: detect, estimate scale, gate on PSR, predict, reacquire |
| `vtt::MosseFilter` / `vtt::KcfFilter` | `vtt/mosse.hpp`, `vtt/kcf.hpp` | Translation correlation filters |
| `vtt::ScaleFilter` | `vtt/scale_filter.hpp` | DSST-style scale estimation over a pyramid |
| `vtt::ConstantVelocityKalman` | `vtt/kalman.hpp` | Centre prediction during occlusion |
| `vtt::Pid`, `vtt::GimbalSim` | `vtt/pid.hpp`, `vtt/gimbal.hpp` | Pan/tilt control with anti-windup and rate limits |
| `vtt::SyntheticScene` | `vtt/synthetic.hpp` | Deterministic test sequence with ground truth |

## Minimal use

```cpp
vtt::Tracker tracker(vtt::default_tracker_config(vtt::Algorithm::Kcf));
tracker.init(first_frame, cv::Rect2d(x, y, w, h));
while (capture.read(frame)) {
    const vtt::TrackOutput out = tracker.update(frame);
    // out.state, out.box, out.psr, out.predicted, out.scale
}
```

Project page: <https://github.com/bekirhankarakurt/vision-target-tracker>
