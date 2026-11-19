"""YOLO detector for the C++ tracker's detect + track mode.

vtt.exe owns the camera. Every few frames it sends a JPEG over UDP; this server
runs YOLO on it and replies with the best box, which the tracker uses to stop
drifting onto the background.

    frame  vtt -> here   : b"VTF1" + uint32 frame_id + uint16 width + uint16 height + JPEG  (little endian)
    reply  here -> vtt   : "DET <frame> <x> <y> <w> <h> <conf>\\n"  or  "NONE <frame>\\n"
                           box in ORIGINAL frame pixels

Run:  python vtt_detector_server.py --model models/deodorant_seg2.pt
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any

MAGIC = b"VTF1"
HEADER = struct.Struct("<4sIHH")


@dataclass(frozen=True)
class FrameHeader:
    frame_id: int
    width: int
    height: int


def encode_header(header: FrameHeader) -> bytes:
    return HEADER.pack(MAGIC, header.frame_id, header.width, header.height)


def decode_datagram(data: bytes) -> tuple[FrameHeader, bytes] | None:
    """Split a frame datagram into header and JPEG bytes; None if malformed."""
    if len(data) <= HEADER.size:
        return None
    magic, frame_id, width, height = HEADER.unpack_from(data)
    if magic != MAGIC or width == 0 or height == 0:
        return None
    return FrameHeader(frame_id, width, height), data[HEADER.size :]


def scale_box(
    box: tuple[float, float, float, float], sent_size: tuple[int, int], original_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Map an (x, y, w, h) box from the sent (width, height) image to the original frame."""
    sx = original_size[0] / sent_size[0]
    sy = original_size[1] / sent_size[1]
    x, y, w, h = box
    return x * sx, y * sy, w * sx, h * sy


def format_reply(frame_id: int, box: tuple[float, float, float, float] | None, conf: float = 0.0) -> bytes:
    if box is None:
        return f"NONE {frame_id}\n".encode("ascii")
    x, y, w, h = box
    return f"DET {frame_id} {x:.1f} {y:.1f} {w:.1f} {h:.1f} {conf:.3f}\n".encode("ascii")


def best_box(result: Any, allowed: set[int] | None) -> tuple[tuple[float, float, float, float], float] | None:
    """Highest-confidence (x, y, w, h), conf from an ultralytics result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    best: tuple[tuple[float, float, float, float], float] | None = None
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        if allowed is not None and k not in allowed:
            continue
        if best is None or c > best[1]:
            best = ((float(x1), float(y1), float(x2 - x1), float(y2 - y1)), float(c))
    return best


def newest_datagram(sock: socket.socket, first: bytes) -> tuple[bytes, int]:
    """Drain frames that queued up while YOLO was busy and keep only the newest.

    Answering stale frames would make the tracker correct itself towards where the
    object used to be; skipping them keeps the detector on the live image.
    """
    newest, skipped = first, 0
    sock.setblocking(False)
    try:
        while True:
            try:
                newest, _ = sock.recvfrom(65535)
                skipped += 1
            except (BlockingIOError, OSError):
                break
    finally:
        sock.settimeout(0.5)
    return newest, skipped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/deodorant_seg2.pt")
    p.add_argument("--listen", type=int, default=5601, help="UDP port for frames from vtt")
    p.add_argument("--reply-host", default="127.0.0.1")
    p.add_argument("--reply-port", type=int, default=5602)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", default="auto", help="auto, cpu or a CUDA index like 0")
    p.add_argument("--imgsz", type=int, default=640)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import cv2
    import numpy as np
    from ultralytics import YOLO

    device: Any = None if args.device == "auto" else args.device
    model = YOLO(args.model)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", args.listen))
    rx.settimeout(0.5)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"detector ready: {args.model} on udp {args.listen} -> {args.reply_host}:{args.reply_port}", flush=True)

    frames, dropped, infer_ms, last_print = 0, 0, 0.0, time.monotonic()
    try:
        while True:
            try:
                data, _ = rx.recvfrom(65535)
            except socket.timeout:
                continue
            data, skipped = newest_datagram(rx, data)
            dropped += skipped
            parsed = decode_datagram(data)
            if parsed is None:
                continue
            header, jpeg = parsed
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            t0 = time.perf_counter()
            result = model.predict(image, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)[0]
            infer_ms += (time.perf_counter() - t0) * 1000.0
            found = best_box(result, None)
            if found is None:
                reply = format_reply(header.frame_id, None)
            else:
                box = scale_box(found[0], (image.shape[1], image.shape[0]), (header.width, header.height))
                reply = format_reply(header.frame_id, box, found[1])
            tx.sendto(reply, (args.reply_host, args.reply_port))
            frames += 1
            now = time.monotonic()
            if now - last_print >= 1.0:
                print(f"{frames / (now - last_print):5.1f} frames/s  inference {infer_ms / max(frames, 1):5.1f} ms  "
                      f"skipped {dropped}  last: {reply.decode().strip()}", flush=True)
                frames, dropped, infer_ms, last_print = 0, 0, 0.0, now
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
        tx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
