# AI detection + depth

Optional Python companion for [vision-target-tracker](../README.md). It runs a YOLO model on a colour frame, draws each box, and, when an Orbbec depth camera is connected, prints the distance in metres. `vtt_detector_server.py` is the separate YOLO process for detect + track: `vtt` sends it frames over UDP and applies the boxes it sends back. The C++ build stays free of a Python dependency.

Python 3.10 or newer. The same script runs on Windows and on a Raspberry Pi 5 (Linux aarch64). Ultralytics uses the CPU when CUDA is not present.

## Install

From the repository root:

```bash
python -m pip install -r python/requirements.txt
```

`pyorbbecsdk2` is only needed for Orbbec depth cameras. Webcam and video-file input do not use it. The import name of that package is `pyorbbecsdk`. To skip it:

```bash
python -m pip install "ultralytics>=8.3" "opencv-python>=4.8" numpy
```

The first run of `--model yolo11n.pt` downloads the weights.

## Usage

```text
python python/vtt_detect.py [options]
  --source orbbec|webcam|file   Frame source (orbbec). Missing SDK or no device
                                falls back to webcam index 0 and prints a warning.
  --webcam-index N              Webcam index when --source webcam (0)
  --input PATH                  Video file when --source file
  --model PATH                  Ultralytics model (yolo11n.pt)
  --classes bottle,person       Keep these COCO names. Unknown names are an error.
  --conf 0.4                    Confidence threshold
  --track                       Ultralytics tracker (model.track persist=True)
  --csv PATH                    Detection log
  --no-show                     No window
```

```bash
python python/vtt_detect.py
python python/vtt_detect.py --source webcam --classes bottle,person --track --csv detections.csv
python python/vtt_detect.py --source file --input clip.mp4 --no-show --csv detections.csv
```

Press `q` or `Esc` to quit, or close the window. Headless live sources stop with Ctrl+C.

Each box is labelled `label #id conf`, and with depth `label #id conf 0.62 m` (for example `bottle #1 0.82 0.62 m`). The corner shows the source name (`orbbec`, `webcam`, or `file`) and FPS.

If depth cannot start, which is common on a USB 2 link that lacks the bandwidth for colour and depth together, the program warns and continues colour-only from the same Orbbec camera. A missing SDK or no device at all falls back to webcam 0.

## Distance

The depth value is the median of the finite, non-zero pixels in the central 40% of the box (40% of the width and 40% of the height, centred). The Orbbec map is in millimetres and is shown in metres. The box is clipped to the image. If that window has no valid pixel, the distance is omitted. When the aligned depth size differs from the colour size, depth is resized with nearest-neighbour.

## CSV

`--csv` writes one row per detection:

```text
frame,timestamp_s,track_id,label,conf,x,y,w,h,distance_m
```

`timestamp_s` is seconds since the run started. `track_id` is empty when `--track` is off. `distance_m` is empty when there is no depth (or no valid pixel). `x,y,w,h` is the top-left box in pixels.

## Train a detector for your own object

`vtt_custom.py` takes one object from camera frames to a trained YOLO segmentation model. Example: a deodorant.

1. **Capture sessions.** Use one place or angle per session. SPACE starts and stops recording; `--autostart` records at once.

   ```bash
   python vtt_custom.py capture --name deodorant --session s0 --count 150
   python vtt_custom.py capture --name deodorant --session neg1 --negative --count 60   # object NOT in view
   python vtt_custom.py capture --name deodorant --session val1 --count 80             # a place not used for training
   ```

2. **Label with SAM 2.** Click the object once in a session. SAM 2.1 segments it and propagates the mask through every frame of that session. Negative sessions get empty labels.

   ```bash
   python vtt_custom.py samlabel --name deodorant
   python vtt_custom.py samlabel --name deodorant --prompt s1:5:818,760   # scripted click: SESSION:FRAME:X,Y[/-X,Y]
   ```

3. **Build and train.** Validation uses whole sessions, so no near-duplicate frame leaks from train to val.

   ```bash
   python vtt_custom.py build --name deodorant --val-session val1 --val-session negv
   python vtt_custom.py train --name deodorant_seg --data-yaml datasets/deodorant/data.yaml --base yolo11n-seg.pt --epochs 60
   python vtt_detect.py --model models/deodorant_seg.pt --track --source orbbec
   ```

What made the difference in practice:

- **Several sessions.** The first model was trained on one desk corner. It reached a high score on frames from that same place and failed everywhere else.
- **Hard negatives.** The model mistook a dark jacket and a shirt for the object. Recording 120 frames of those without the object and fine-tuning from the previous weights removed the false alarms.
- **A negative validation session.** It makes precision count false alarms, not only misses.

Datasets, weights and run outputs are git-ignored.

## Detect + track server

`vtt_detector_server.py` is the YOLO process for the C++ tracker's detect + track mode. `vtt` keeps the camera and the correlation tracker, sends JPEG frames over UDP, and uses the reply to refine or re-anchor the track. See the [Detect + track](../README.md#detect--track-yolo-corrects-the-tracker) section in the top-level README.

```text
python python/vtt_detector_server.py [options]
  --model PATH          .pt, .onnx, or TensorRT .engine (models/deodorant_seg2.pt)
  --listen N            UDP port for frames from vtt (5601)
  --reply-host HOST     Where to send replies (127.0.0.1)
  --reply-port N        Reply port (5602)
  --conf 0.35           YOLO confidence threshold
  --device auto|cpu|N   Inference device (auto)
  --imgsz N             Inference size (640)
  --half / --no-half    FP16 on CUDA. Default: on when CUDA is available
```

Before it prints `detector ready`, the server runs one inference on a blank image so the first live frames skip CUDA setup.

A frame datagram is `VTF1`, a uint32 frame id, uint16 width, uint16 height (little endian), then a JPEG. The reply is `DET <id> <x> <y> <w> <h> <conf> <infer_ms>` or `NONE <id> <infer_ms>`. The box is in the original frame's pixels. `infer_ms` is this frame's inference time in milliseconds. While YOLO is busy the server keeps only the newest datagram.

Optional TensorRT speed-up. Export locally; the server loads the engine with the same `--model` path:

```bash
yolo export model=models/deodorant_seg2.pt format=engine half=True
python vtt_detector_server.py --model models/deodorant_seg2.engine
```

`run_detect_track.ps1` starts the server in its own window, waits for the model to load, then runs `vtt.exe` with `--detector 127.0.0.1:5601 --auto-lock`.

```powershell
.\run_detect_track.ps1 -Camera 2 -Model models\deodorant_seg2.pt -Vtt C:\b\vtt\Release\vtt.exe
```

The server tests cover the datagram header, box scaling, reply lines with and without `infer_ms`, and the FP16 choice. They do not load a model.

```bash
python -m pytest python/test_vtt_detector_server.py
```

## Tests

The tests cover the distance helper, class-name parsing, labels, CSV rows, and the detect + track server. They do not open a camera and they do not download a model.

```bash
python -m pip install pytest
python -m pytest python/test_vtt_detect.py python/test_vtt_detector_server.py
```
