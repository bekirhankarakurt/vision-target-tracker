"""Pure-function tests for vtt_detector_server (no model, no camera, no network)."""

from types import SimpleNamespace

import numpy as np

import vtt_detector_server as server


def test_header_round_trip():
    data = server.encode_header(server.FrameHeader(123456, 1280, 720)) + b"\xff\xd8jpeg"
    parsed = server.decode_datagram(data)
    assert parsed is not None
    header, jpeg = parsed
    assert header == server.FrameHeader(123456, 1280, 720)
    assert jpeg == b"\xff\xd8jpeg"


def test_bad_datagrams_rejected():
    good = server.encode_header(server.FrameHeader(1, 640, 480))
    assert server.decode_datagram(b"XTF1" + good[4:] + b"x") is None
    assert server.decode_datagram(good) is None  # header only, no image
    assert server.decode_datagram(b"VTF") is None
    assert server.decode_datagram(server.encode_header(server.FrameHeader(1, 0, 480)) + b"x") is None


def test_header_matches_cpp_layout():
    # Same bytes as vtt::encode_frame_header: "VTF1", u32 LE, u16 LE, u16 LE.
    assert server.encode_header(server.FrameHeader(0x01020304, 0x0500, 0x02D0)) == (
        b"VTF1" + bytes([4, 3, 2, 1]) + bytes([0, 5]) + bytes([0xD0, 2])
    )


def test_scale_box_back_to_original():
    box = server.scale_box((64.0, 36.0, 32.0, 18.0), (640, 360), (1280, 720))
    assert box == (128.0, 72.0, 64.0, 36.0)


def test_reply_lines():
    assert server.format_reply(7, None) == b"NONE 7\n"
    assert server.format_reply(42, (10.0, 20.5, 30.0, 40.0), 0.8766) == b"DET 42 10.0 20.5 30.0 40.0 0.877\n"


def _result(xyxy, conf, cls):
    class T:
        def __init__(self, a):
            self.a = np.asarray(a, dtype=float)

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    return SimpleNamespace(boxes=_Boxes(T(xyxy), T(conf), T(cls)))


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    def __len__(self):
        return len(self.conf.a)


def test_best_box_picks_highest_confidence():
    result = _result([[0, 0, 10, 10], [100, 50, 140, 150]], [0.4, 0.9], [0, 0])
    found = server.best_box(result, None)
    assert found is not None
    assert found[0] == (100.0, 50.0, 40.0, 100.0)
    assert abs(found[1] - 0.9) < 1e-9
    assert server.best_box(_result([], [], []), None) is None
    assert server.best_box(result, {3}) is None


def test_newest_datagram_skips_backlog():
    import socket
    import time

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(3):
        tx.sendto(f"f{i}".encode(), rx.getsockname())
    time.sleep(0.05)
    rx.settimeout(0.5)
    first, _ = rx.recvfrom(100)
    newest, skipped = server.newest_datagram(rx, first)
    assert (first, newest, skipped) == (b"f0", b"f2", 2)
    rx.close()
    tx.close()
