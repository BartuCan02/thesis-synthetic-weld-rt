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
    merge_polygon_pieces,
    polygon_to_yolo_seg,
    polygon_visible_mask,
    simplify_polygon,
    tile_positions,
    tile_side,
    trace_mask_polygon,
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


# ---- polygon labels for the segmentation arm ---------------------------------------------------------


def _filled(points, side=100):
    import cv2

    canvas = np.zeros((side, side), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(points).astype(np.int32)], 1)
    return canvas


def test_trace_round_trips_a_square_and_implies_the_d7_box_minus_one():
    poly = _square(10, 10, 30, 30)
    mask = polygon_visible_mask(poly, 0, 0, 100)
    box = box_from_visible(poly, mask, 4, 0.10, 0.005)
    pts, n_pieces = trace_mask_polygon(mask)
    assert n_pieces == 1
    assert np.array_equal(_filled(pts), mask)
    assert pts.min(axis=0).tolist() == [box.x0, box.y0]
    assert pts.max(axis=0).tolist() == [box.x1 - 1, box.y1 - 1]


def test_trace_of_the_visible_piece_follows_the_window():
    # square 90..110 in image coords, window at (50, 50) of side 50 sees 90..100 -> local 40..49 inclusive
    poly = _square(90, 90, 110, 110)
    mask = polygon_visible_mask(poly, 50, 50, 50)
    pts, _ = trace_mask_polygon(mask)
    assert np.array_equal(_filled(pts, 50), mask)
    assert pts.min(axis=0).tolist() == [40.0, 40.0]
    assert pts.max(axis=0).tolist() == [49.0, 49.0]


def test_trace_of_concave_piece_keeps_both_arms_and_the_gap():
    # a U shape: base at y 80..100, arms at y 100..150 (x 10..30 and x 60..80). The window y 100..200 sees only
    # the two arms: one instance, two pieces, joined by a bridge; the gap between the arms stays empty
    pts_img = np.array(
        [[10, 80], [10, 150], [30, 150], [30, 100], [60, 100], [60, 150], [80, 150], [80, 80]],
        dtype=np.float32,
    )
    poly = Polygon("裂纹", 2, pts_img, "polygon")
    mask = polygon_visible_mask(poly, 0, 100, 100)
    pts, n_pieces = trace_mask_polygon(mask)
    assert n_pieces == 2
    filled = _filled(pts)
    inter = np.logical_and(filled, mask).sum()
    union = np.logical_or(filled, mask).sum()
    assert inter / union > 0.95  # only the one-pixel bridge line is extra
    assert filled[25, 20] == 1 and filled[25, 70] == 1
    assert filled[75, 45] == 0 and filled[10, 45] == 0  # the gap away from the bridge
    assert pts.min(axis=0).tolist() == [10.0, 0.0] and pts.max(axis=0).tolist() == [80.0, 50.0]


def test_merge_pieces_returns_one_closed_path_covering_both():
    a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    b = np.array([[50, 50], [60, 50], [60, 60], [50, 60]], dtype=np.float32)
    merged = merge_polygon_pieces([a, b])
    assert len(merged) == len(a) + len(b) + 2  # bridge out and back
    filled = _filled(merged)
    assert filled[5, 5] == 1 and filled[55, 55] == 1 and filled[5, 55] == 0


def test_simplify_keeps_corners_and_at_least_three_points():
    # 20 points along the edges of a square -> 4 corners
    sq = []
    for i in range(5):
        sq.append([i * 10, 0])
    for i in range(5):
        sq.append([40, i * 10])
    for i in range(5):
        sq.append([40 - i * 10, 40])
    for i in range(5):
        sq.append([0, 40 - i * 10])
    out = simplify_polygon(np.array(sq, dtype=np.float32), 0.5)
    assert len(out) == 4
    tri = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
    assert len(simplify_polygon(tri, 5.0)) == 3


def test_yolo_seg_line_is_normalised_and_clipped():
    pts = np.array([[0, 0], [50, 0], [50, 100.4]], dtype=np.float32)
    line = polygon_to_yolo_seg(3, pts, 100)
    assert line == "3 0.000000 0.000000 0.500000 0.000000 0.500000 1.000000"
