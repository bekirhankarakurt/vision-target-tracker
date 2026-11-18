"""Unit tests for the pure helpers in vtt_detect. No camera and no model download."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vtt_detect import (  # noqa: E402
    CSV_HEADER,
    Detection,
    box_distance_m,
    build_parser,
    detection_row,
    format_label,
    parse_classes,
    validate_args,
)

NAMES = {0: "person", 39: "bottle", 41: "cup"}


def test_all_zeros_return_none() -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[0, 0] = 4000.0  # outside the central 40%, so it must not count
    assert box_distance_m(depth, (0, 0, 10, 10)) is None


def test_median_ignores_zeros_and_pixels_outside_the_inner_window() -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    # box (0, 0, 10, 10), inner 0.4 -> pixels x in [3, 7), y in [3, 7)
    depth[4, 4] = 1000.0
    depth[4, 5] = 3000.0
    depth[5, 5] = 0.0
    depth[4, 2] = 100.0  # one pixel left of the inner window
    depth[4, 7] = 100.0  # one pixel right of the inner window
    depth[2, 4] = 100.0  # one pixel above the inner window
    depth[0, 0] = 8000.0
    # Valid samples are 1000 and 3000 mm, so the median is 2 m. Pulling in any
    # of the 100 mm neighbours would move the median.
    assert box_distance_m(depth, (0, 0, 10, 10), inner=0.4) == pytest.approx(2.0)
    assert box_distance_m(depth, (0, 0, 10, 10)) == pytest.approx(2.0)


def test_even_count_median_averages_the_middle_pair() -> None:
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[4, 4] = 1000.0
    depth[4, 5] = 3000.0
    assert box_distance_m(depth, (0, 0, 10, 10), inner=0.4) == pytest.approx(2.0)


def test_box_is_clipped_to_image_edges() -> None:
    depth = np.full((8, 8), 100.0, dtype=np.float32)
    depth[0:4, 0:4] = 2000.0
    depth[0, 0] = 0.0
    depth[1, 1] = 4000.0
    # The box hangs off the top-left. inner=1 keeps the whole box, clipped to [0:4, 0:4].
    # Fourteen 2000 mm samples and one 4000 mm sample: the median is 2 m.
    # The 100 mm border, and anything off the image, must not enter the median.
    measured = box_distance_m(depth, (-2.0, -2.0, 6.0, 6.0), inner=1.0)
    assert measured == pytest.approx(2.0)


def test_box_clipped_on_the_far_edge_ignores_the_rest_of_the_image() -> None:
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[8:10, 8:10] = 1500.0
    depth[0, 0] = 9000.0
    assert box_distance_m(depth, (8, 8, 10, 10), inner=1.0) == pytest.approx(1.5)


def test_inner_window_that_falls_outside_the_image_is_none() -> None:
    depth = np.full((20, 20), 2500.0, dtype=np.float32)
    # Central 40% of this box sits entirely left of x=0. The on-image part of the
    # outer box must not be sampled.
    assert box_distance_m(depth, (-10.0, 0.0, 12.0, 10.0), inner=0.4) is None


def test_box_completely_outside_or_empty_is_none() -> None:
    depth = np.full((8, 8), 1000.0, dtype=np.float32)
    assert box_distance_m(depth, (20, 20, 5, 5)) is None
    assert box_distance_m(depth, (0, 0, 0, 10)) is None
    assert box_distance_m(depth, (0, 0, -1, 10)) is None
    assert box_distance_m(depth, (float("nan"), 0, 4, 4)) is None


def test_non_finite_depth_is_ignored() -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[4, 4] = np.nan
    depth[4, 5] = np.inf
    depth[4, 6] = 2000.0
    assert box_distance_m(depth, (0, 0, 10, 10)) == pytest.approx(2.0)
    depth[4, 6] = 0.0
    assert box_distance_m(depth, (0, 0, 10, 10)) is None


def test_inner_fraction_must_be_in_range() -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="inner"):
        box_distance_m(depth, (0, 0, 4, 4), inner=0.0)
    with pytest.raises(ValueError, match="inner"):
        box_distance_m(depth, (0, 0, 4, 4), inner=1.5)


def test_parse_classes_known_names_preserve_order_and_casefold() -> None:
    assert parse_classes(" Person, BOTTLE, person ", NAMES) == [0, 39]
    assert parse_classes("cup,,bottle", NAMES) == [41, 39]


def test_parse_classes_blank_means_no_filter() -> None:
    assert parse_classes(None, NAMES) is None
    assert parse_classes("", NAMES) is None
    assert parse_classes("   ", NAMES) is None


def test_unknown_class_name_raises_a_clear_error() -> None:
    with pytest.raises(ValueError, match=r"Unknown class name\(s\): not_a_class") as caught:
        parse_classes("bottle,not_a_class", NAMES)
    message = str(caught.value)
    assert "bottle" in message
    assert "person" in message
    with pytest.raises(ValueError, match="pers"):
        parse_classes("pers", {0: "person"})


def test_format_label_includes_id_confidence_and_metres() -> None:
    assert format_label("bottle", 1, 0.82) == "bottle #1 0.82"
    assert format_label("bottle", 1, 0.82, 0.62) == "bottle #1 0.82 0.62 m"
    assert format_label("cup", None, 0.4, 1.5) == "cup 0.40 1.50 m"
    assert format_label("person", 0, 1.0) == "person #0 1.00"


def test_parser_defaults_and_usage_errors(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.source == "orbbec"
    assert args.webcam_index == 0
    assert args.model == "yolo11n.pt"
    assert args.conf == pytest.approx(0.4)
    assert args.track is False
    assert args.no_show is False
    assert args.csv is None
    assert validate_args(args) == 0
    assert validate_args(parser.parse_args(["--conf", "0"])) == 2
    assert validate_args(parser.parse_args(["--conf", "1.5"])) == 2
    assert validate_args(parser.parse_args(["--webcam-index", "-1"])) == 2
    assert validate_args(parser.parse_args(["--source", "file"])) == 2
    missing = tmp_path / "missing.mp4"
    assert validate_args(parser.parse_args(["--source", "file", "--input", str(missing)])) == 2
    present = tmp_path / "clip.mp4"
    present.write_bytes(b"not a video")
    assert validate_args(parser.parse_args(["--source", "file", "--input", str(present)])) == 0


def test_csv_row_leaves_track_id_and_distance_empty() -> None:
    assert CSV_HEADER == (
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
    empty = detection_row(
        3,
        1.25,
        Detection(None, "bottle", 0.5, 1.0, 2.0, 3.0, 4.0, None),
    )
    assert empty[0] == 3
    assert empty[2] == ""
    assert empty[-1] == ""
    ranged = detection_row(
        1,
        0.5,
        Detection(7, "person", 0.9, 8.0, 9.0, 10.0, 11.0, 0.62),
    )
    assert ranged[2] == "7"
    assert ranged[-1] == "0.6200"
