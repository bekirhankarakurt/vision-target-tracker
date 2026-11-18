#!/usr/bin/env python3
"""YOLO detection and optional Orbbec depth for vision-target-tracker.

Companion CLI for Python 3.10+ on Windows and Raspberry Pi 5 (Linux aarch64).
The C++ correlation tracker is a separate program and does not import this module.

Distance, when a depth frame is available, is the median of the valid (non-zero)
depth pixels in the central 40% of each box, converted from millimetres to metres.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

CSV_HEADER: tuple[str, ...] = (
    "frame",
    "timestamp_s",
    "track_id",
    "label",
    "conf",
    "x",
    "y",
    "w",
    "h",
    "distance_m",
)
DEFAULT_INNER_FRACTION = 0.4
DEFAULT_CONFIDENCE = 0.4
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_SOURCE = "orbbec"
FALLBACK_WEBCAM_INDEX = 0
ORBBEC_FRAME_TIMEOUT_MS = 100
MAX_LIVE_READ_FAILURES = 100
PREVIEW_MAX_WIDTH = 1280
WINDOW_NAME = "vtt_detect"
QUIT_KEYS = frozenset({ord("q"), ord("Q"), 27})
_PIXEL_SNAP = 1e-6


class DependencyError(Exception):
    """A required third-party package is not installed."""


class SourceError(Exception):
    """A webcam or video file could not be opened."""


class OrbbecUnavailable(Exception):
    """The Orbbec SDK is missing or no usable device was found."""


@dataclass(frozen=True)
class Detection:
    """One detection in top-left ``(x, y, w, h)`` pixel coordinates."""

    track_id: int | None
    label: str
    conf: float
    x: float
    y: float
    w: float
    h: float
    distance_m: float | None


def box_distance_m(
    depth_mm: np.ndarray,
    box: Sequence[float],
    inner: float = DEFAULT_INNER_FRACTION,
) -> float | None:
    """Median depth in metres inside the central portion of a box.

    ``depth_mm`` is a 2-D map in millimetres. ``box`` is ``(x, y, w, h)`` with
    ``(x, y)`` the top-left corner in pixels. ``inner`` is the fraction of the
    box width and height that is kept, centred (``0.4`` keeps the central 40%).

    Valid samples are finite and non-zero. Zeros are missing depth. The box is
    clipped to the image before the median is taken. Returns ``None`` when the
    clipped window contains no valid pixel.
    """
    if not math.isfinite(inner) or not 0.0 < inner <= 1.0:
        raise ValueError("inner must be a finite fraction in (0, 1]")
    depth = np.asarray(depth_mm)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError("depth_mm must be a 2-D array in millimetres")
    if len(box) != 4:
        raise ValueError("box must be (x, y, w, h)")
    x, y, w, h = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return None
    height, width = int(depth.shape[0]), int(depth.shape[1])
    cols = _axis_bounds(x, w, inner, width)
    rows = _axis_bounds(y, h, inner, height)
    if cols is None or rows is None:
        return None
    x0, x1 = cols
    y0, y1 = rows
    roi = depth[y0:y1, x0:x1]
    valid = roi[np.isfinite(roi) & (roi != 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / 1000.0


def parse_classes(spec: str | None, names: Mapping[int, str]) -> list[int] | None:
    """Map a comma-separated class-name filter to model class ids.

    ``None`` and a blank string mean "no filter". Names are matched
    case-insensitively against ``names``. Unknown names raise ``ValueError``.
    Ids follow the order the user wrote and are de-duplicated.
    """
    if spec is None:
        return None
    if not isinstance(spec, str):
        raise TypeError("class filter must be a comma-separated string or None")
    if spec.strip() == "":
        return None

    by_exact: dict[str, int] = {}
    by_fold: dict[str, int] = {}
    ordered_names: list[str] = []
    for key, name in names.items():
        label = str(name).strip()
        index = int(key)
        ordered_names.append(label)
        by_exact.setdefault(label, index)
        by_fold.setdefault(label.casefold(), index)

    chosen: list[int] = []
    seen: set[int] = set()
    unknown: list[str] = []
    for part in spec.split(","):
        label = part.strip()
        if label == "":
            continue
        index = by_exact.get(label)
        if index is None:
            index = by_fold.get(label.casefold())
        if index is None:
            unknown.append(label)
            continue
        if index not in seen:
            seen.add(index)
            chosen.append(index)
    if unknown:
        known = ", ".join(ordered_names) if ordered_names else "(none)"
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown class name(s): {joined}. Known classes: {known}")
    if not chosen:
        raise ValueError("Class filter did not contain any class names.")
    return chosen


def format_label(
    label: str,
    track_id: int | None,
    conf: float,
    distance_m: float | None = None,
) -> str:
    """Overlay text ``label #id conf``, plus metres when depth is known.

    Without a track id the ``#id`` token is omitted. Without depth the distance
    suffix is omitted.

    ``format_label("bottle", 1, 0.82)`` is ``"bottle #1 0.82"``.
    ``format_label("bottle", 1, 0.82, 0.62)`` is ``"bottle #1 0.82 0.62 m"``.
    """
    text = label
    if track_id is not None:
        text = f"{text} #{int(track_id)}"
    text = f"{text} {float(conf):.2f}"
    if distance_m is not None:
        text = f"{text} {float(distance_m):.2f} m"
    return text


def detection_row(frame_index: int, timestamp_s: float, det: Detection) -> list[object]:
    """One CSV record. Track id and distance are empty when they are unknown."""
    track = "" if det.track_id is None else str(int(det.track_id))
    distance = "" if det.distance_m is None else f"{det.distance_m:.4f}"
    return [
        int(frame_index),
        f"{float(timestamp_s):.3f}",
        track,
        det.label,
        f"{float(det.conf):.4f}",
        f"{float(det.x):.2f}",
        f"{float(det.y):.2f}",
        f"{float(det.w):.2f}",
        f"{float(det.h):.2f}",
        distance,
    ]


def _axis_bounds(origin: float, size: float, inner: float, limit: int) -> tuple[int, int] | None:
    """Inclusive-exclusive pixel span of the centred inner window, clipped to ``[0, limit)``."""
    if limit <= 0 or size <= 0.0:
        return None
    margin = (1.0 - inner) * size / 2.0
    start = min(max(origin + margin, 0.0), float(limit))
    end = min(max(origin + size - margin, 0.0), float(limit))
    if end <= start:
        return None
    # Snap values that are an ulp away from an integer back onto that integer
    # so a box on pixel boundaries does not grow by one pixel.
    i0 = math.floor(start + _PIXEL_SNAP)
    i1 = math.ceil(end - _PIXEL_SNAP)
    i0 = min(max(i0, 0), limit)
    i1 = min(max(i1, 0), limit)
    if i1 <= i0:
        return None
    return i0, i1


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise DependencyError(
            "opencv-python is not installed. Install it with: pip install -r python/requirements.txt"
        ) from exc
    return cv2


def _import_orbbec() -> tuple[Any, Any, Any, Any, Any, Any]:
    # Package name on pip is pyorbbecsdk2; the import name is pyorbbecsdk.
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBStreamType, AlignFilter

    return Pipeline, Config, OBSensorType, OBFormat, OBStreamType, AlignFilter


def _as_flat(raw: Any, dtype: Any) -> np.ndarray | None:
    if raw is None:
        return None
    np_dtype = np.dtype(dtype)
    try:
        if isinstance(raw, np.ndarray):
            flat = np.ascontiguousarray(raw).reshape(-1)
            if flat.dtype == np_dtype:
                return flat
            return flat.view(np_dtype)
        return np.frombuffer(raw, dtype=np_dtype)
    except (TypeError, ValueError, BufferError):
        return None


def _format_is(fmt: Any, expected: Any, name: str) -> bool:
    if expected is not None and fmt == expected:
        return True
    expected_value = getattr(expected, "value", None)
    fmt_value = getattr(fmt, "value", fmt)
    if expected_value is not None and fmt_value == expected_value:
        return True
    label = getattr(fmt, "name", None)
    if not isinstance(label, str):
        label = str(fmt)
    token = label.upper().rsplit(".", 1)[-1]
    return token == name.upper()


def _frame_format_name(frame: Any) -> str:
    try:
        fmt = frame.get_format()
    except Exception:
        return "unknown"
    label = getattr(fmt, "name", None)
    if isinstance(label, str) and label:
        return label
    return str(fmt)


def color_frame_to_bgr(color_frame: Any, ob_format: Any) -> np.ndarray | None:
    """Decode an Orbbec colour frame to an owned BGR ``uint8`` image.

    MJPG is decoded with ``cv2.imdecode``. RGB is converted to BGR. BGR is
    copied as-is. YUYV (and the YUY2 alias) uses ``COLOR_YUV2BGR_YUYV``.
    NV12 and I420 use ``cvtColor``. Returns ``None`` instead of raising.
    """
    if color_frame is None:
        return None
    cv2 = load_cv2()
    try:
        width = int(color_frame.get_width())
        height = int(color_frame.get_height())
        fmt = color_frame.get_format()
        raw = color_frame.get_data()
    except Exception:
        return None
    if width <= 0 or height <= 0 or raw is None:
        return None
    flat = _as_flat(raw, np.uint8)
    if flat is None:
        return None

    def owned(count: int, shape: tuple[int, ...]) -> np.ndarray | None:
        if count <= 0 or flat.size < count:
            return None
        try:
            return flat[:count].copy().reshape(shape)
        except ValueError:
            return None

    try:
        if _format_is(fmt, getattr(ob_format, "MJPG", None), "MJPG"):
            decoded = cv2.imdecode(np.ascontiguousarray(flat), cv2.IMREAD_COLOR)
            if decoded is None:
                return None
            return np.ascontiguousarray(decoded)
        if _format_is(fmt, getattr(ob_format, "RGB", None), "RGB"):
            image = owned(height * width * 3, (height, width, 3))
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if _format_is(fmt, getattr(ob_format, "BGR", None), "BGR"):
            return owned(height * width * 3, (height, width, 3))
        if _format_is(fmt, getattr(ob_format, "YUYV", None), "YUYV") or _format_is(
            fmt, getattr(ob_format, "YUY2", None), "YUY2"
        ):
            image = owned(height * width * 2, (height, width, 2))
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        if _format_is(fmt, getattr(ob_format, "NV12", None), "NV12"):
            image = owned(height * width * 3 // 2, (height * 3 // 2, width))
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)
        if _format_is(fmt, getattr(ob_format, "I420", None), "I420"):
            image = owned(height * width * 3 // 2, (height * 3 // 2, width))
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
    except cv2.error:
        return None
    return None


def depth_frame_to_mm(depth_frame: Any) -> np.ndarray | None:
    """Raw Orbbec depth to a float32 map in millimetres, or ``None``.

    ``get_depth_scale()`` converts the uint16 samples to millimetres. The
    returned array does not alias the SDK buffer.
    """
    if depth_frame is None:
        return None
    try:
        height = int(depth_frame.get_height())
        width = int(depth_frame.get_width())
        raw = depth_frame.get_data()
        if height <= 0 or width <= 0 or raw is None:
            return None
        flat = _as_flat(raw, np.uint16)
        if flat is None or flat.size < height * width:
            return None
        try:
            scale = float(depth_frame.get_depth_scale())
        except Exception:
            scale = 1.0
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        depth_mm = flat[: height * width].astype(np.float32).reshape(height, width) * scale
    except (TypeError, ValueError, BufferError):
        return None
    if depth_mm.shape != (height, width):
        return None
    return depth_mm


def resize_depth_if_needed(depth_mm: np.ndarray | None, color: np.ndarray) -> np.ndarray | None:
    """Match depth to the colour image with nearest-neighbour resampling."""
    if depth_mm is None:
        return None
    color_h, color_w = int(color.shape[0]), int(color.shape[1])
    if depth_mm.shape[:2] == (color_h, color_w):
        return depth_mm
    cv2 = load_cv2()
    try:
        return cv2.resize(depth_mm, (color_w, color_h), interpolation=cv2.INTER_NEAREST)
    except cv2.error:
        return None


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def model_class_names(model: Any) -> dict[int, str]:
    raw = model.names
    if isinstance(raw, dict):
        return {int(key): str(name) for key, name in raw.items()}
    return {index: str(name) for index, name in enumerate(list(raw))}


def extract_detections(
    result: Any,
    names: Mapping[int, str],
    depth_mm: np.ndarray | None,
) -> list[Detection]:
    """Read Ultralytics boxes. Distance is omitted when ``depth_mm`` is ``None``."""
    boxes = None if result is None else getattr(result, "boxes", None)
    if boxes is None:
        return []
    try:
        count = len(boxes)
    except TypeError:
        return []
    if count == 0:
        return []
    xyxy = _numpy(boxes.xyxy)
    if xyxy.ndim != 2 or xyxy.shape[0] == 0 or xyxy.shape[1] < 4:
        return []
    confs = _numpy(boxes.conf).reshape(-1)
    classes = _numpy(boxes.cls).reshape(-1)
    ids = None
    raw_ids = getattr(boxes, "id", None)
    if raw_ids is not None:
        ids = _numpy(raw_ids).reshape(-1)
    result_names = getattr(result, "names", None)
    if isinstance(result_names, dict) and result_names:
        names = {int(key): str(name) for key, name in result_names.items()}

    detections: list[Detection] = []
    total = int(xyxy.shape[0])
    for index in range(total):
        if index >= confs.size or index >= classes.size:
            break
        x1, y1, x2, y2 = (float(value) for value in xyxy[index, :4])
        width = x2 - x1
        height = y2 - y1
        if width <= 0.0 or height <= 0.0:
            continue
        class_id = int(classes[index])
        track_id = None
        if ids is not None and index < ids.size:
            track_id = int(ids[index])
        distance = None
        if depth_mm is not None:
            distance = box_distance_m(depth_mm, (x1, y1, width, height))
        detections.append(
            Detection(
                track_id=track_id,
                label=str(names.get(class_id, str(class_id))),
                conf=float(confs[index]),
                x=x1,
                y=y1,
                w=width,
                h=height,
                distance_m=distance,
            )
        )
    return detections


def prepare_color(image: np.ndarray) -> np.ndarray:
    """Return a contiguous BGR ``uint8`` image for Ultralytics and OpenCV."""
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        cv2 = load_cv2()
        gray = image if image.ndim == 2 else image[:, :, 0]
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        cv2 = load_cv2()
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    return image


def draw_masks(image: np.ndarray, result: Any, alpha: float = 0.45) -> np.ndarray:
    """Blend segmentation masks (YOLO-seg models) onto a copy of ``image``; no-op for box-only models."""
    masks = getattr(result, "masks", None) if result is not None else None
    polygons = getattr(masks, "xy", None) if masks is not None else None
    if not polygons:
        return image
    cv2 = load_cv2()
    overlay = image.copy()
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], (40, 220, 40))
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_overlay(
    image: np.ndarray,
    detections: Sequence[Detection],
    fps: float,
    source_name: str,
) -> np.ndarray:
    """Copy ``image`` and draw boxes, labels, FPS, and the source name."""
    cv2 = load_cv2()
    vis = image.copy()
    height, width = vis.shape[:2]
    for det in detections:
        x1 = min(max(int(round(det.x)), 0), max(0, width - 1))
        y1 = min(max(int(round(det.y)), 0), max(0, height - 1))
        x2 = min(max(int(round(det.x + det.w)), 0), max(0, width - 1))
        y2 = min(max(int(round(det.y + det.h)), 0), max(0, height - 1))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 220, 40), 2)
        _draw_label(
            cv2,
            vis,
            format_label(det.label, det.track_id, det.conf, det.distance_m),
            x1,
            y1,
        )
    _draw_hud(cv2, vis, f"{source_name}  {fps:0.1f} FPS")
    return vis


def _draw_label(cv2: Any, image: np.ndarray, text: str, x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    height, width = image.shape[:2]
    left = min(max(x, 0), max(0, width - 1))
    baseline_y = y - 6
    if baseline_y < text_h + 4:
        baseline_y = min(height - 2, y + text_h + 8)
    top = max(0, baseline_y - text_h - 3)
    bottom = min(height - 1, baseline_y + baseline + 2)
    right = min(width - 1, left + text_w + 4)
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        image,
        text,
        (left + 2, min(height - 2, baseline_y)),
        font,
        scale,
        (40, 220, 40),
        thickness,
        cv2.LINE_AA,
    )


def _draw_hud(cv2: Any, image: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (6, 6), (14 + text_w, 14 + text_h + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (10, 10 + text_h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def append_detections(
    writer: Any,
    csv_file: Any,
    frame_index: int,
    timestamp_s: float,
    detections: Sequence[Detection],
) -> None:
    if writer is None:
        return
    for det in detections:
        writer.writerow(detection_row(frame_index, timestamp_s, det))
    csv_file.flush()


def infer_frame(
    model: Any,
    frame: np.ndarray,
    *,
    track: bool,
    conf: float,
    classes: list[int] | None,
) -> Any:
    kwargs: dict[str, Any] = {"conf": conf, "verbose": False}
    if classes is not None:
        kwargs["classes"] = classes
    if track:
        results = model.track(frame, persist=True, **kwargs)
    else:
        results = model.predict(frame, **kwargs)
    if results is None:
        return None
    if isinstance(results, (list, tuple)):
        return results[0] if len(results) else None
    return results


class FrameSource:
    """Colour frames plus an optional depth map in millimetres."""

    name = "source"
    live = True
    ended = False

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


class CvSource(FrameSource):
    def __init__(self, capture: Any, name: str, live: bool) -> None:
        self.name = name
        self.live = live
        self.ended = False
        self._capture = capture
        self._failures = 0

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._capture is None or self.ended:
            self.ended = True
            return None, None
        try:
            ok, frame = self._capture.read()
        except Exception:
            ok, frame = False, None
        if not ok or frame is None:
            self._failures += 1
            if not self.live:
                self.ended = True
            elif self._failures >= MAX_LIVE_READ_FAILURES:
                print(
                    f"Error: {self.name} returned no frames {self._failures} times in a row. "
                    "Is another program (e.g. Orbbec Viewer or the Camera app) using it?",
                    file=sys.stderr,
                )
                self.ended = True
            return None, None
        self._failures = 0
        return frame, None

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()


class OrbbecSource(FrameSource):
    def __init__(self, pipeline: Any, align: Any, ob_format: Any) -> None:
        self.name = "orbbec"
        self.live = True
        self.ended = False
        self._pipeline = pipeline
        self._align = align
        self._ob_format = ob_format
        self._warned: set[str] = set()

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ``(bgr, depth_mm)``. Timeouts and empty frames yield ``(None, None)``."""
        if self._pipeline is None:
            self.ended = True
            return None, None
        try:
            frames = self._pipeline.wait_for_frames(ORBBEC_FRAME_TIMEOUT_MS)  # may return None
        except Exception as exc:
            self._warn_once("wait", f"Warning: Orbbec wait_for_frames failed ({exc}).")
            return None, None
        if frames is None:
            return None, None

        use_depth = self._align is not None
        if use_depth:
            try:
                processed = self._align.process(frames)
                aligned = processed.as_frame_set() if processed is not None else None
            except Exception as exc:
                self._warn_once("align", f"Warning: depth alignment failed ({exc}).")
                aligned = None
            if aligned is None:
                use_depth = False
            else:
                frames = aligned
        if frames is None:
            return None, None

        try:
            color = frames.get_color_frame()
        except Exception as exc:
            self._warn_once("color-get", f"Warning: get_color_frame failed ({exc}).")
            return None, None
        if color is None:
            return None, None
        bgr = color_frame_to_bgr(color, self._ob_format)
        if bgr is None:
            self._warn_once(
                "color-decode",
                "Warning: could not decode an Orbbec colour frame "
                f"(format {_frame_format_name(color)}).",
            )
            return None, None

        depth_mm = None
        if use_depth:
            depth_mm = self._read_depth(frames, bgr)
        return bgr, depth_mm

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._align = None
        _stop_quietly(pipeline)

    def _read_depth(self, frames: Any, bgr: np.ndarray) -> np.ndarray | None:
        try:
            depth = frames.get_depth_frame()
        except Exception as exc:
            self._warn_once("depth-get", f"Warning: get_depth_frame failed ({exc}).")
            return None
        if depth is None:
            return None
        before = None
        try:
            before = (int(depth.get_height()), int(depth.get_width()))
        except Exception:
            before = None
        depth_mm = depth_frame_to_mm(depth)
        if depth_mm is None:
            self._warn_once(
                "depth-decode",
                "Warning: a depth frame arrived but could not be decoded. Distance will be omitted.",
            )
            return None
        resized = resize_depth_if_needed(depth_mm, bgr)
        if resized is None:
            self._warn_once("depth-size", "Warning: depth could not be matched to the colour frame.")
            return None
        if before is not None and before != resized.shape[:2]:
            self._warn_once(
                "depth-resize",
                "Warning: aligned depth "
                f"{before[1]}x{before[0]} differs from colour "
                f"{resized.shape[1]}x{resized.shape[0]}; "
                "resampling depth with nearest neighbour.",
            )
        return resized

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(message, file=sys.stderr)


def _stop_quietly(pipeline: Any) -> None:
    if pipeline is None:
        return
    try:
        pipeline.stop()
    except Exception:
        return None


def _make_align(align_filter: Any, stream_type: Any) -> Any:
    try:
        return align_filter(align_to_stream=stream_type.COLOR_STREAM)
    except TypeError:
        return align_filter(stream_type.COLOR_STREAM)


def _default_video_profile(pipeline: Any, sensor_type: Any) -> Any:
    return pipeline.get_stream_profile_list(sensor_type).get_default_video_stream_profile()


def open_orbbec() -> OrbbecSource:
    """Start colour and, when the link allows it, aligned depth.

    A depth-start failure (commonly USB 2 bandwidth) keeps the same camera in
    colour-only mode. A missing SDK or a missing colour stream raises
    ``OrbbecUnavailable`` so the caller can fall back to a webcam.
    """
    try:
        Pipeline, Config, OBSensorType, OBFormat, OBStreamType, AlignFilter = _import_orbbec()
    except ImportError as exc:
        raise OrbbecUnavailable(
            "Orbbec SDK (pyorbbecsdk) is not installed or failed to load"
            f" ({exc})"
        ) from exc

    pipeline: Any = None
    try:
        pipeline = Pipeline()
        color_profile = _default_video_profile(pipeline, OBSensorType.COLOR_SENSOR)
        config = Config()
        config.enable_stream(color_profile)
        depth_failure: str | None = None
        try:
            depth_profile = _default_video_profile(pipeline, OBSensorType.DEPTH_SENSOR)
            config.enable_stream(depth_profile)
        except Exception as exc:
            depth_failure = f"depth profile unavailable: {exc}"
        try:
            pipeline.start(config)
        except Exception as exc:
            _stop_quietly(pipeline)
            pipeline = None
            if depth_failure is not None:
                raise OrbbecUnavailable(f"no Orbbec device was found ({exc})") from exc
            pipeline = Pipeline()
            retry = Config()
            retry.enable_stream(_default_video_profile(pipeline, OBSensorType.COLOR_SENSOR))
            try:
                pipeline.start(retry)
            except Exception as colour_exc:
                raise OrbbecUnavailable(
                    f"no Orbbec device was found ({colour_exc})"
                ) from colour_exc
            depth_failure = (
                "depth stream failed to start, often because a USB 2 link does not "
                f"have enough bandwidth for colour and depth ({exc})"
            )
        align = None
        if depth_failure is None:
            try:
                align = _make_align(AlignFilter, OBStreamType)
            except Exception as exc:
                depth_failure = f"depth alignment filter unavailable: {exc}"
        if depth_failure is not None:
            print(
                f"Warning: {depth_failure}. Continuing colour-only from the Orbbec camera.",
                file=sys.stderr,
            )
        source = OrbbecSource(pipeline, align, OBFormat)
        pipeline = None
        return source
    except OrbbecUnavailable:
        raise
    except Exception as exc:
        raise OrbbecUnavailable(f"no Orbbec device was found ({exc})") from exc
    finally:
        _stop_quietly(pipeline)


def _open_capture(cv2: Any, index: int) -> Any:
    capture = cv2.VideoCapture(index)
    if capture.isOpened():
        return capture
    capture.release()
    # DirectShow succeeds on some Windows cameras that the default backend rejects.
    # Leave the default backend on Linux so Raspberry Pi V4L2 is unchanged.
    if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    return None


def open_webcam(index: int) -> CvSource:
    cv2 = load_cv2()
    capture = _open_capture(cv2, index)
    if capture is None:
        raise SourceError(f"could not open webcam index {index}")
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass
    return CvSource(capture, name="webcam", live=True)


def open_file(path: str) -> CvSource:
    cv2 = load_cv2()
    if not os.path.isfile(path):
        raise SourceError(f"video file not found: {path}")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise SourceError(f"could not open video file: {path}")
    return CvSource(capture, name="file", live=False)


def open_source(args: argparse.Namespace) -> FrameSource:
    if args.source == "file":
        return open_file(args.input)
    if args.source == "webcam":
        return open_webcam(args.webcam_index)
    try:
        return open_orbbec()
    except OrbbecUnavailable as exc:
        print(
            "Warning: Orbbec SDK is missing or no device was found "
            f"({exc}). Falling back to webcam index {FALLBACK_WEBCAM_INDEX}.",
            file=sys.stderr,
        )
        return open_webcam(FALLBACK_WEBCAM_INDEX)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vtt_detect",
        description=(
            "Detect objects with YOLO and, on an Orbbec camera, show the median "
            "depth inside each box. The C++ gimbal tracker is not involved."
        ),
        epilog=(
            "Press q or Esc to quit. Distance is the median of the valid depth "
            "pixels in the central 40% of each box, in metres."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=("orbbec", "webcam", "file"),
        default=DEFAULT_SOURCE,
        help=(
            "Frame source. If the Orbbec SDK is missing or no device is found, "
            f"fall back to webcam index {FALLBACK_WEBCAM_INDEX}."
        ),
    )
    parser.add_argument(
        "--webcam-index",
        type=int,
        default=FALLBACK_WEBCAM_INDEX,
        help="Webcam index used with --source webcam.",
    )
    parser.add_argument("--input", default=None, help="Video file used when --source file.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ultralytics model name or path. Weights are downloaded on first use.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Optional comma-separated COCO names to keep, for example bottle,person.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Keep a stable id per object with Ultralytics tracking (persist=True).",
    )
    parser.add_argument("--csv", default=None, help="Write detections to this CSV path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window.")
    return parser


def validate_args(args: argparse.Namespace) -> int:
    if not math.isfinite(args.conf) or not 0.0 < args.conf <= 1.0:
        print("error: --conf must be in the range (0, 1].", file=sys.stderr)
        return 2
    if args.webcam_index < 0:
        print("error: --webcam-index must be zero or positive.", file=sys.stderr)
        return 2
    if args.source == "file":
        if args.input is None or not str(args.input).strip():
            print("error: --source file requires --input PATH.", file=sys.stderr)
            return 2
        if not os.path.isfile(args.input):
            print(f"error: video file not found: {args.input}", file=sys.stderr)
            return 2
    if args.input and args.source != "file":
        print("Warning: --input is only used with --source file; ignoring it.", file=sys.stderr)
    return 0


def load_model(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DependencyError(
            "ultralytics is not installed. Install it with: pip install -r python/requirements.txt"
        ) from exc
    try:
        return YOLO(model_name)
    except Exception as exc:
        raise DependencyError(f"could not load model {model_name}: {exc}") from exc


def run_loop(
    source: FrameSource,
    model: Any,
    names: Mapping[int, str],
    *,
    track: bool,
    conf: float,
    classes: list[int] | None,
    writer: Any,
    csv_file: Any,
    headless: bool,
) -> int:
    if headless:
        hint = " Press Ctrl+C to stop." if source.live else ""
        print(f"Headless {source.name}.{hint}", file=sys.stderr)

    started = time.perf_counter()
    previous = started
    fps = 0.0
    frame_index = 0
    window_open = False
    cv2 = None
    try:
        while True:
            image, depth_mm = source.read()
            if image is None:
                if source.ended:
                    break
                time.sleep(0.01)
                continue
            image = prepare_color(image)
            if image.size == 0:
                continue
            try:
                result = infer_frame(model, image, track=track, conf=conf, classes=classes)
            except Exception as exc:
                print(f"error: detection failed: {exc}", file=sys.stderr)
                return 1
            detections = extract_detections(result, names, depth_mm)
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            if elapsed > 0.0:
                sample = 1.0 / elapsed
                fps = sample if fps == 0.0 else (0.9 * fps + 0.1 * sample)
            append_detections(writer, csv_file, frame_index, now - started, detections)
            frame_index += 1
            if headless:
                continue
            vis = draw_overlay(draw_masks(image, result), detections, fps, source.name)
            cv2 = load_cv2()
            try:
                if not window_open:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    if vis.shape[1] > PREVIEW_MAX_WIDTH:
                        scale = PREVIEW_MAX_WIDTH / vis.shape[1]
                        cv2.resizeWindow(WINDOW_NAME, PREVIEW_MAX_WIDTH, int(vis.shape[0] * scale))
                    window_open = True
                cv2.imshow(WINDOW_NAME, vis)
                key = cv2.waitKey(1)
            except cv2.error as exc:
                print(
                    f"error: cannot open a display ({exc}). Re-run with --no-show.",
                    file=sys.stderr,
                )
                return 1
            if key != -1 and (key & 0xFF) in QUIT_KEYS:
                break
            # -1 means this backend does not report visibility. 0 means the user
            # closed the window. Ignore the first frame, before the window is mapped.
            if frame_index > 1:
                try:
                    visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    break
                if visible == 0:
                    break
    finally:
        if window_open and cv2 is not None:
            cv2.destroyAllWindows()
    print(f"Stopped after {frame_index} frame(s) from {source.name}.", file=sys.stderr)
    return 0


def _run(argv: list[str] | None) -> int:
    args = build_parser().parse_args(argv)
    usage = validate_args(args)
    if usage != 0:
        return usage
    model = load_model(args.model)
    names = model_class_names(model)
    try:
        classes = parse_classes(args.classes, names)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    csv_file = None
    try:
        with open_source(args) as source:
            writer = None
            if args.csv:
                try:
                    csv_file = open(args.csv, "w", newline="", encoding="utf-8")
                except OSError as exc:
                    print(f"error: could not open CSV {args.csv}: {exc}", file=sys.stderr)
                    return 1
                writer = csv.writer(csv_file)
                writer.writerow(CSV_HEADER)
            return run_loop(
                source,
                model,
                names,
                track=bool(args.track),
                conf=float(args.conf),
                classes=classes,
                writer=writer,
                csv_file=csv_file,
                headless=bool(args.no_show),
            )
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if csv_file is not None:
            csv_file.close()


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except KeyboardInterrupt:
        return 0
    except DependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
