"""Unit tests for the pipeline rules. Run: uv run pytest tests/ (pytest is in the dev extra)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import (
    Polygon,
    box_from_visible,
    contrast_stretch_to_uint8,
    exposure_id,
    polygon_visible_mask,
    tile_positions,
    tile_side,
)


def test_tile_side_is_half_the_short_side():
    assert tile_side(717, 6943) == 358
    assert tile_side(6943, 717) == 358  # portrait images use the same rule


def test_tile_positions_half_overlap_flush_and_drop():
    # side 100, stride 50 along a 1000 px axis: regular windows end at 900; flush adds nothing new
    assert tile_positions(1000, 100, 0.5, "flush") == list(range(0, 901, 50))
    # 1030 px: regular grid ends at 950 (window 950..1050 would overrun), flush adds 930
    assert tile_positions(1030, 100, 0.5, "drop") == list(range(0, 901, 50))
    assert tile_positions(1030, 100, 0.5, "flush") == list(range(0, 901, 50)) + [930]
    # axis shorter than the window -> one window at 0
    assert tile_positions(80, 100) == [0]


def test_exposure_id_merges_tjoint_halves():
    assert exposure_id("A_DJ-RT-20220621-10") == "DJ-RT-20220621-10"
    assert exposure_id("B_DJ-RT-20220621-10") == "DJ-RT-20220621-10"
    assert exposure_id("DJ-RT-20220621-10") == "DJ-RT-20220621-10"


def _square(x0, y0, x1, y1, cls=0):
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    return Polygon("气孔", cls, pts, "polygon")


def test_box_fully_inside_tile_is_kept_with_full_visibility():
    poly = _square(10, 10, 30, 30)
    mask = polygon_visible_mask(poly, 0, 0, 100)
    box = box_from_visible(poly, mask, 4, 0.10, 0.005)
    assert box is not None
    assert (box.x0, box.y0, box.x1, box.y1) == (10, 10, 31, 31)
    assert abs(box.visible_frac - 1.0) < 0.15  # rasterisation of a 20x20 square -> 21x21 pixels


def test_polygon_outside_tile_gives_no_mask():
    assert polygon_visible_mask(_square(500, 500, 520, 520), 0, 0, 100) is None


def test_sliver_is_dropped_but_long_defect_piece_is_kept():
    # a 400x8 'lack of penetration' bar crossing a 100 px tile: visible fraction 25 % -> kept by frac rule
    bar = _square(-150, 46, 250, 54, cls=5)
    mask = polygon_visible_mask(bar, 0, 0, 100)
    assert box_from_visible(bar, mask, 4, 0.10, 0.005) is not None
    # the same bar but 4000 px long: visible fraction 2.5 %, yet it covers 8 % of the tile -> kept by tile rule
    long_bar = _square(-1950, 46, 2050, 54, cls=5)
    mask = polygon_visible_mask(long_bar, 0, 0, 100)
    assert box_from_visible(long_bar, mask, 4, 0.10, 0.005) is not None
    # a 2 px sliver of a 20x20 pore at the tile edge: too thin -> dropped
    pore = _square(98, 40, 118, 60)
    mask = polygon_visible_mask(pore, 0, 0, 100)
    assert box_from_visible(pore, mask, 4, 0.10, 0.005) is None


def test_yolo_line_is_normalised():
    poly = _square(0, 0, 50, 50)
    box = box_from_visible(poly, polygon_visible_mask(poly, 0, 0, 100), 4, 0.1, 0.005)
    cls, cx, _cy, w, _h = box.to_yolo(100).split()
    assert cls == "0" and 0.0 < float(cx) < 1.0 and 0.0 < float(w) <= 1.0


def test_contrast_stretch_percentiles_and_minmax():
    tile = np.arange(0, 65536, 64, dtype=np.uint16).reshape(32, 32)
    out = contrast_stretch_to_uint8(tile, 0.0, 100.0)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    clipped = contrast_stretch_to_uint8(tile, 10.0, 90.0)
    assert (clipped == 0).mean() > 0.05 and (clipped == 255).mean() > 0.05  # tails saturate
    flat = contrast_stretch_to_uint8(np.full((8, 8), 1234, dtype=np.uint16), 0.5, 99.5)
    assert flat.max() == 0
