#!/usr/bin/env python3
"""Teach YOLO a new object from your own camera: capture -> autolabel -> review -> train.

Example (a deodorant can):

    python vtt_custom.py capture   --name deodorant
    python vtt_custom.py autolabel --name deodorant --seed-classes bottle,cup
    python vtt_custom.py review    --name deodorant
    python vtt_custom.py train     --name deodorant
    python vtt_detect.py --model models/deodorant.pt --track
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

import vtt_detect as vd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NEGATIVE_MARKER = "NEGATIVE"
SAM_WORK_WIDTH = 1280  # frames are downscaled to this width for SAM 2 video propagation


# --------------------------------------------------------------------------- pure helpers


def to_yolo_line(box_xywh: Sequence[float], img_w: int, img_h: int, cls: int = 0) -> str:
    """Pixel (x, y, w, h) with top-left origin -> YOLO 'cls cx cy w h' (normalised, clipped)."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError("image size must be positive")
    x, y, w, h = (float(v) for v in box_xywh)
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(img_w), x + w), min(float(img_h), y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box lies outside the image")
    cx = (x0 + x1) / 2.0 / img_w
    cy = (y0 + y1) / 2.0 / img_h
    nw = (x1 - x0) / img_w
    nh = (y1 - y0) / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def pick_seed_box(
    detections: Iterable[tuple[int, float, Sequence[float]]], seed_class_ids: Iterable[int]
) -> tuple[float, float, float, float] | None:
    """Best (class_id, conf, xywh) among seed classes: highest conf, ties -> larger area."""
    seeds = set(seed_class_ids)
    best: tuple[float, float, tuple[float, float, float, float]] | None = None
    for cls_id, conf, box in detections:
        if cls_id not in seeds:
            continue
        b = tuple(float(v) for v in box)
        key = (float(conf), b[2] * b[3])
        if best is None or key > best[:2]:
            best = (key[0], key[1], b)  # type: ignore[assignment]
    return None if best is None else best[2]


def filter_shape(
    detections: Iterable[tuple[int, float, Sequence[float]]],
    img_h: int,
    min_height: float = 0.0,
    min_aspect: float = 0.0,
) -> list[tuple[int, float, Sequence[float]]]:
    """Drop boxes shorter than min_height * img_h or with height/width below min_aspect."""
    kept = []
    for cls_id, conf, box in detections:
        _x, _y, w, h = (float(v) for v in box)
        if w <= 0 or h <= 0:
            continue
        if h < min_height * img_h or h / w < min_aspect:
            continue
        kept.append((cls_id, conf, box))
    return kept


def split_items(items: Sequence[str], val_fraction: float, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministic shuffle + split. At least one val item when there are >= 2 items."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val fraction must be in [0, 1)")
    ordered = sorted(items)
    random.Random(seed).shuffle(ordered)
    n_val = int(round(len(ordered) * val_fraction))
    if len(ordered) >= 2 and val_fraction > 0:
        n_val = max(1, min(n_val, len(ordered) - 1))
    return ordered[n_val:], ordered[:n_val]


def next_index(existing_names: Iterable[str], name: str) -> int:
    """Next free number for files called <name>_<NNNN>.<ext>."""
    pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.[A-Za-z]+$")
    numbers = [int(m.group(1)) for n in existing_names if (m := pattern.match(n))]
    return max(numbers, default=-1) + 1


# --------------------------------------------------------------------------- commands


def _data_dir(args: argparse.Namespace) -> Path:
    return Path(args.data) if args.data else Path("datasets") / args.name


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def cmd_capture(args: argparse.Namespace) -> int:
    cv2 = vd.load_cv2()
    if args.session:
        out = _data_dir(args) / "raw" / args.session
    else:
        out = _data_dir(args) / "images" / "all"
    out.mkdir(parents=True, exist_ok=True)
    if args.negative:
        (out / NEGATIVE_MARKER).write_text("frames in this session contain no target object\n", encoding="utf-8")
    index = next_index((p.name for p in out.iterdir()), args.name)
    saved = len(_images(out))
    recording = bool(getattr(args, "autostart", False))
    last_save = 0.0
    window = f"capture: {args.name}"
    print(f"Saving to {out.resolve()}. SPACE = start/stop recording, q/Esc = quit.", file=sys.stderr)
    with vd.open_source(args) as source:
        while saved < args.count:
            image, _depth = source.read()
            if image is None:
                if source.ended:
                    break
                time.sleep(0.01)
                continue
            image = vd.prepare_color(image)
            now = time.perf_counter()
            if recording and now - last_save >= args.interval:
                path = out / f"{args.name}_{index:04d}.jpg"
                cv2.imwrite(str(path), image)
                index += 1
                saved += 1
                last_save = now
            view = image.copy()
            if recording:
                cv2.circle(view, (24, 24), 10, (0, 0, 255), -1)
            status = f"{'REC' if recording else 'PAUSED (SPACE to record)'}  saved {saved}/{args.count}"
            cv2.putText(view, status, (44, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, status, (44, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            tip = args.hint or "rotate the object, change distance and background"
            cv2.putText(view, tip, (14, view.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            scale = min(1.0, args.preview_width / view.shape[1])
            if scale < 1.0:
                view = cv2.resize(view, (int(view.shape[1] * scale), int(view.shape[0] * scale)))
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
                last_save = 0.0
    cv2.destroyAllWindows()
    print(f"Captured {saved} image(s) in {out}.", file=sys.stderr)
    return 0


def _contact_sheet(cv2, items: list[tuple[Path, tuple[float, float, float, float]]], dest: Path) -> None:
    thumbs = []
    for path, (x, y, w, h) in items[:48]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), max(2, img.shape[1] // 200))
        scale = 160 / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        tile = np.zeros((160, 160, 3), dtype=np.uint8)
        tile[: img.shape[0], : img.shape[1]] = img
        thumbs.append(tile)
    if not thumbs:
        return
    cols = 8
    while len(thumbs) % cols:
        thumbs.append(np.zeros((160, 160, 3), dtype=np.uint8))
    rows = [np.hstack(thumbs[i : i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite(str(dest), np.vstack(rows))


def cmd_autolabel(args: argparse.Namespace) -> int:
    cv2 = vd.load_cv2()
    data = _data_dir(args)
    images = _images(data / "images" / "all")
    if not images:
        print(f"Error: no images in {data / 'images' / 'all'}. Run capture first.", file=sys.stderr)
        return 2
    model = vd.load_model(args.model)
    names = vd.model_class_names(model)
    seed_ids = vd.parse_classes(args.seed_classes, names) or []
    labelled: list[tuple[Path, tuple[float, float, float, float], str]] = []
    unlabeled_dir = data / "unlabeled"
    unlabeled = 0
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        result = model.predict(img, conf=args.conf, verbose=False)[0]
        dets = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                x0, y0, x1, y1 = (float(v) for v in xyxy)
                dets.append((int(cls), float(conf), (x0, y0, x1 - x0, y1 - y0)))
        dets = filter_shape(dets, img.shape[0], args.min_height, args.min_aspect)
        box = pick_seed_box(dets, seed_ids)
        if box is None:
            unlabeled_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), unlabeled_dir / path.name)
            unlabeled += 1
            continue
        labelled.append((path, box, to_yolo_line(box, img.shape[1], img.shape[0])))
    if not labelled:
        print("Error: no image got a box. Try --conf 0.15 or other --seed-classes.", file=sys.stderr)
        return 3
    train, val = split_items([p.name for p, _, _ in labelled], args.val, args.seed)
    split_of = {n: "train" for n in train} | {n: "val" for n in val}
    for path, _box, line in labelled:
        split = split_of[path.name]
        (data / "images" / split).mkdir(parents=True, exist_ok=True)
        (data / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), data / "images" / split / path.name)
        (data / "labels" / split / f"{path.stem}.txt").write_text(line + "\n", encoding="utf-8")
    yaml = (
        f"path: {data.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n  0: {args.name}\n"
    )
    (data / "data.yaml").write_text(yaml, encoding="utf-8")
    preview = [(data / "images" / split_of[p.name] / p.name, b) for p, b, _ in labelled]
    _contact_sheet(cv2, preview, data / "preview.jpg")
    print(
        f"labelled {len(labelled)}, unlabeled {unlabeled}, train {len(train)}, val {len(val)}\n"
        f"data.yaml: {(data / 'data.yaml').resolve()}\npreview:   {(data / 'preview.jpg').resolve()}",
        file=sys.stderr,
    )
    return 0


def _read_label(path: Path, w: int, h: int) -> tuple[int, int, int, int] | None:
    try:
        parts = path.read_text(encoding="utf-8").split()
        _cls, cx, cy, bw, bh = (float(v) for v in parts[:5])
    except (OSError, ValueError):
        return None
    return int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)


def cmd_review(args: argparse.Namespace) -> int:
    cv2 = vd.load_cv2()
    data = _data_dir(args)
    pairs = [
        (img, data / "labels" / split / f"{img.stem}.txt")
        for split in ("train", "val")
        for img in _images(data / "images" / split)
    ]
    if not pairs:
        print("Error: nothing to review. Run autolabel first.", file=sys.stderr)
        return 2
    rejected = data / "rejected"
    i = 0
    while 0 <= i < len(pairs):
        img_path, label_path = pairs[i]
        img = cv2.imread(str(img_path))
        if img is None:
            pairs.pop(i)
            continue
        box = _read_label(label_path, img.shape[1], img.shape[0])
        if box:
            cv2.rectangle(img, box[:2], box[2:], (0, 255, 0), 3)
        text = f"{i + 1}/{len(pairs)}  n=next p=prev d=delete q=quit"
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("review", img)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key in (ord("p"), 2424832):
            i = max(0, i - 1)
        elif key == ord("d"):
            rejected.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_path), rejected / img_path.name)
            if label_path.exists():
                shutil.move(str(label_path), rejected / label_path.name)
            pairs.pop(i)
        else:
            i += 1
    cv2.destroyAllWindows()
    return 0


def polygon_line(points_norm: Sequence[Sequence[float]], cls: int = 0, min_area: float = 5e-4) -> str | None:
    """Normalised polygon -> YOLO-seg line 'cls x1 y1 x2 y2 ...'. None when degenerate/tiny."""
    pts = np.clip(np.asarray(points_norm, dtype=np.float64).reshape(-1, 2), 0.0, 1.0)
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < min_area:
        return None
    return f"{cls} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1))


def _sessions(data: Path) -> list[Path]:
    raw = data / "raw"
    return sorted(p for p in raw.iterdir() if p.is_dir()) if raw.is_dir() else []


def _seg_sheet(cv2, items: list[tuple[Path, np.ndarray | None]], dest: Path) -> None:
    thumbs = []
    step = max(1, len(items) // 48)
    for path, poly in items[::step][:48]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        if poly is not None:
            pts = (poly * [img.shape[1], img.shape[0]]).astype(np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
            cv2.polylines(img, [pts], True, (0, 255, 0), max(2, img.shape[1] // 300))
        scale = 160 / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        tile = np.zeros((160, 160, 3), dtype=np.uint8)
        tile[: img.shape[0], : img.shape[1]] = img
        thumbs.append(tile)
    if not thumbs:
        return
    while len(thumbs) % 8:
        thumbs.append(np.zeros((160, 160, 3), dtype=np.uint8))
    cv2.imwrite(str(dest), np.vstack([np.hstack(thumbs[i : i + 8]) for i in range(0, len(thumbs), 8)]))


def _pick_prompt(cv2, sam, frames: list[Path], session: str) -> tuple[int, list[tuple[int, int]], list[int]] | None:
    """Window: choose a frame, click the object. Returns (start_index, points_px, labels) or None to skip."""
    state = {"idx": 0, "points": [], "labels": [], "dirty": True}
    window = f"samlabel {session}: L-click object, R-click background, ENTER run, n/p frame, r reset, s skip, q quit"
    img = None
    scale = 1.0

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            state["points"].append((int(x / scale), int(y / scale)))
            state["labels"].append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
            state["dirty"] = True

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    view = None
    while True:
        if state["dirty"]:
            img = cv2.imread(str(frames[state["idx"]]))
            scale = min(1.0, 1280 / img.shape[1])
            view = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            if state["points"]:
                res = sam(img, points=[state["points"]], labels=[state["labels"]], verbose=False)[0]
                if res.masks is not None and len(res.masks.xy):
                    pts = (res.masks.xy[0] * scale).astype(np.int32)
                    if len(pts) >= 3:
                        overlay = view.copy()
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                        view = cv2.addWeighted(overlay, 0.45, view, 0.55, 0)
                for (px, py), lab in zip(state["points"], state["labels"]):
                    cv2.circle(view, (int(px * scale), int(py * scale)), 7, (0, 255, 0) if lab else (0, 0, 255), -1)
            cv2.putText(view, f"frame {state['idx'] + 1}/{len(frames)}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            state["dirty"] = False
        cv2.imshow(window, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and state["points"]:
            cv2.destroyWindow(window)
            return state["idx"], state["points"], state["labels"]
        if key == ord("s"):
            cv2.destroyWindow(window)
            return None
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt
        if key == ord("r"):
            state.update(points=[], labels=[], dirty=True)
        if key in (ord("n"), ord("p")):
            step = 1 if key == ord("n") else -1
            state.update(idx=min(len(frames) - 1, max(0, state["idx"] + step)), points=[], labels=[], dirty=True)


def parse_prompt(spec: str) -> tuple[str, tuple[int, list[tuple[int, int]], list[int]]]:
    """'s2:5:1030,500/-1140,630' -> ('s2', (5, [(1030, 500), (1140, 630)], [1, 0])). '-' = background point."""
    try:
        session, frame, pts = spec.split(":", 2)
        points: list[tuple[int, int]] = []
        labels: list[int] = []
        for item in pts.split("/"):
            neg = item.startswith("-")
            x, y = item.lstrip("-").split(",")
            points.append((int(x), int(y)))
            labels.append(0 if neg else 1)
        return session, (int(frame), points, labels)
    except ValueError as exc:
        raise ValueError(f"bad --prompt {spec!r}; expected SESSION:FRAME:X,Y[/-X,Y]") from exc


def _auto_prompt(detector, frames: list[Path], conf: float) -> tuple[int, list[tuple[int, int]], list[int]] | None:
    """Use an existing detector instead of a human click: first frame with a confident box -> its centre."""
    for i, f in enumerate(frames):
        res = detector.predict(str(f), conf=conf, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            best = int(res.boxes.conf.argmax())
            x0, y0, x1, y1 = (float(v) for v in res.boxes.xyxy[best])
            return i, [(int((x0 + x1) / 2), int((y0 + y1) / 2))], [1]
    return None


def cmd_samlabel(args: argparse.Namespace) -> int:
    cv2 = vd.load_cv2()
    from ultralytics import SAM
    from ultralytics.models.sam import SAM2VideoPredictor

    data = _data_dir(args)
    sessions = [data / "raw" / s for s in args.session] if args.session else _sessions(data)
    if not sessions:
        print(f"Error: no sessions under {data / 'raw'}. Run capture --session first.", file=sys.stderr)
        return 2
    sam = SAM(args.sam_model)
    detector = vd.load_model(args.auto_prompt) if args.auto_prompt else None
    manual = dict(parse_prompt(p) for p in (args.prompt or []))
    for session in sessions:
        frames = _images(session)
        out = data / "labels_seg" / session.name
        if not frames:
            continue
        if out.is_dir() and any(out.iterdir()) and not args.redo:
            print(f"{session.name}: already labelled (use --redo)", file=sys.stderr)
            continue
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        if (session / NEGATIVE_MARKER).exists():
            for f in frames:
                (out / f"{f.stem}.txt").write_text("", encoding="utf-8")
            print(f"{session.name}: negative session, {len(frames)} empty labels", file=sys.stderr)
            continue
        prompt = manual.get(session.name)
        if prompt is None and detector is not None:
            prompt = _auto_prompt(detector, frames, args.auto_conf)
        if prompt is not None:
            print(f"{session.name}: auto prompt at frame {prompt[0] + 1}, point {prompt[1][0]}", file=sys.stderr)
        else:
            prompt = _pick_prompt(cv2, sam, frames, session.name)
        if prompt is None:
            print(f"{session.name}: skipped", file=sys.stderr)
            continue
        start, points, labels = prompt
        used = frames[start:]
        first = cv2.imread(str(used[0]))
        h, w = first.shape[:2]
        scale = min(1.0, SAM_WORK_WIDTH / w)
        size = (int(w * scale), int(h * scale))
        video = data / "labels_seg" / f"{session.name}.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
        for f in used:
            frame = cv2.imread(str(f))
            writer.write(cv2.resize(frame, size) if frame.shape[1::-1] != size else frame)
        writer.release()
        predictor = SAM2VideoPredictor(
            overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=args.sam_model, verbose=False)
        )
        pts = [[int(x * scale), int(y * scale)] for x, y in points]
        sheet: list[tuple[Path, np.ndarray | None]] = []
        written = 0
        for f, res in zip(used, predictor(source=str(video), points=[pts], labels=[labels], stream=True)):
            poly = None
            if res.masks is not None and len(res.masks.xyn):
                poly = max(res.masks.xyn, key=len) if len(res.masks.xyn) > 1 else res.masks.xyn[0]
            line = polygon_line(poly) if poly is not None and len(poly) else None
            if line is not None:
                (out / f"{f.stem}.txt").write_text(line + "\n", encoding="utf-8")
                written += 1
            sheet.append((f, np.asarray(poly) if line is not None else None))
        video.unlink(missing_ok=True)
        _seg_sheet(cv2, sheet, data / "labels_seg" / f"{session.name}_preview.jpg")
        print(
            f"{session.name}: {written}/{len(used)} frames segmented "
            f"(preview: {(data / 'labels_seg' / (session.name + '_preview.jpg')).resolve()})",
            file=sys.stderr,
        )
    cv2.destroyAllWindows()
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    data = _data_dir(args)
    val_sessions = set(args.val_session)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        shutil.rmtree(data / sub, ignore_errors=True)
        (data / sub).mkdir(parents=True)
    counts = {"train": 0, "val": 0, "neg_train": 0, "neg_val": 0}
    for session in _sessions(data):
        labels = data / "labels_seg" / session.name
        if not labels.is_dir():
            print(f"{session.name}: no labels, ignored", file=sys.stderr)
            continue
        split = "val" if session.name in val_sessions else "train"
        for lab in sorted(labels.glob("*.txt")):
            img = next((session / f"{lab.stem}{e}" for e in (".jpg", ".png", ".jpeg") if (session / f"{lab.stem}{e}").exists()), None)
            if img is None:
                continue
            stem = f"{session.name}__{lab.stem}"
            shutil.copy2(img, data / "images" / split / f"{stem}{img.suffix}")
            shutil.copy2(lab, data / "labels" / split / f"{stem}.txt")
            counts[split] += 1
            if lab.stat().st_size == 0:
                counts[f"neg_{split}"] += 1
    missing = val_sessions - {s.name for s in _sessions(data)}
    if missing:
        print(f"Error: unknown val session(s): {', '.join(sorted(missing))}", file=sys.stderr)
        return 2
    (data / "data.yaml").write_text(
        f"path: {data.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: {args.name}\n",
        encoding="utf-8",
    )
    print(
        f"train {counts['train']} (negatives {counts['neg_train']}), val {counts['val']} (negatives {counts['neg_val']})\n"
        f"data.yaml: {(data / 'data.yaml').resolve()}",
        file=sys.stderr,
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    data_yaml = Path(args.data_yaml) if args.data_yaml else _data_dir(args) / "data.yaml"
    if not data_yaml.is_file():
        print(f"Error: {data_yaml} not found. Run autolabel first.", file=sys.stderr)
        return 2
    import torch
    from ultralytics import YOLO

    device = args.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    model = YOLO(args.base)
    model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str((out / "runs").resolve()),
        name=args.name,
        exist_ok=True,
        patience=15,
        workers=0,
    )
    best = out / "runs" / args.name / "weights" / "best.pt"
    if not best.is_file():
        print("Error: training finished without best.pt.", file=sys.stderr)
        return 3
    final = out / f"{args.name}.pt"
    shutil.copy2(best, final)
    metrics = YOLO(str(final)).val(data=str(data_yaml.resolve()), device=device, verbose=False, workers=0)
    seg = getattr(metrics, "seg", None)
    seg_text = f"\nmask mAP50 {seg.map50:.3f}   mask mAP50-95 {seg.map:.3f}" if seg is not None else ""
    print(
        f"\nbox mAP50 {metrics.box.map50:.3f}   box mAP50-95 {metrics.box.map:.3f}{seg_text}\n"
        f"model: {final.resolve()}\nrun it live:\n  python vtt_detect.py --model {final.as_posix()} --track",
        file=sys.stderr,
    )
    return 0


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", required=True, help="class name, e.g. deodorant")
        p.add_argument("--data", help="dataset folder (default: datasets/<name>)")

    p = sub.add_parser("capture", help="record images from the camera")
    common(p)
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--interval", type=float, default=0.4)
    p.add_argument("--source", choices=("orbbec", "webcam", "file"), default="orbbec")
    p.add_argument("--webcam-index", type=int, default=0)
    p.add_argument("--input")
    p.add_argument("--preview-width", type=int, default=960, help="preview window width in px (images are saved full size)")
    p.add_argument("--session", help="save under raw/<session>/ (one place/angle per session; used by samlabel)")
    p.add_argument("--negative", action="store_true", help="this session shows NO target object (background examples)")
    p.add_argument("--autostart", action="store_true", help="start recording immediately (no SPACE needed)")
    p.add_argument("--hint", help="instruction text drawn at the bottom of the capture window")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("samlabel", help="click the object once, SAM 2 segments it through the whole session")
    common(p)
    p.add_argument("--session", action="append", help="session(s) under raw/ to label (default: all unlabeled)")
    p.add_argument("--sam-model", default="sam2.1_b.pt")
    p.add_argument("--redo", action="store_true", help="relabel sessions that already have labels")
    p.add_argument("--auto-prompt", help="detector .pt that places the first click automatically (falls back to the click window)")
    p.add_argument("--auto-conf", type=float, default=0.5)
    p.add_argument("--prompt", action="append", help="scripted click, SESSION:FRAME:X,Y[/-X,Y] (frame is 0-based, '-' = background)")
    p.set_defaults(func=cmd_samlabel)

    p = sub.add_parser("build", help="assemble train/val from labelled sessions (val = whole sessions, no leakage)")
    common(p)
    p.add_argument("--val-session", action="append", required=True, help="session(s) used only for validation")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("autolabel", help="bootstrap labels with a pretrained COCO model")
    common(p)
    p.add_argument("--seed-classes", default="bottle,cup")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--min-height", type=float, default=0.0, help="reject boxes shorter than this fraction of the image height")
    p.add_argument("--min-aspect", type=float, default=0.0, help="reject boxes with height/width below this (e.g. 1.5 for upright cans)")
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_autolabel)

    p = sub.add_parser("review", help="step through labels, delete bad ones")
    common(p)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("train", help="fine-tune YOLO on the dataset")
    common(p)
    p.add_argument("--data-yaml")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--base", default="yolo11n.pt", help="yolo11n.pt for boxes, yolo11n-seg.pt for masks")
    p.add_argument("--out", default="models")
    p.set_defaults(func=cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
