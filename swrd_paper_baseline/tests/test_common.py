"""Unit tests for the pipeline rules. Run: uv run pytest tests/ (pytest is in the dev extra)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import (
    Polygon,
    box_from_visible,
    clip_polygon_to_tile,
    contrast_stretch_to_uint8,
    exposure_id,
    polygon_to_yolo_seg,
    polygon_visible_mask,
    simplify_polygon,
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


# ---- polygon labels for the segmentation arm ---------------------------------------------------------


def test_clip_keeps_a_polygon_fully_inside_the_tile():
    pts = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
    out = clip_polygon_to_tile(pts, 0, 0, 100)
    assert out.shape == (4, 2)
    assert np.allclose(sorted(map(tuple, out)), sorted(map(tuple, pts)))


def test_clip_translates_to_tile_coordinates_and_cuts_at_the_border():
    # square 90..110 in image coords, tile at x=50 of side 50 covers 50..100 -> visible part 90..100 -> local 40..50
    pts = np.array([[90, 90], [110, 90], [110, 110], [90, 110]], dtype=np.float32)
    out = clip_polygon_to_tile(pts, 50, 50, 50)
    assert len(out) == 4
    assert out.min(axis=0).tolist() == [40.0, 40.0]
    assert out.max(axis=0).tolist() == [50.0, 50.0]


def test_clip_of_polygon_outside_the_tile_is_empty():
    pts = np.array([[500, 500], [520, 500], [520, 520]], dtype=np.float32)
    assert len(clip_polygon_to_tile(pts, 0, 0, 100)) == 0


def test_clipped_concave_polygon_rasterises_like_the_d7_mask():
    # a U shape: base at y 80..100, two arms at y 100..150 (x 10..30 and x 60..80). The window y 100..200
    # sees only the two arms, so the clipped polygon must keep both and leave the gap between them empty.
    pts = np.array(
        [[10, 80], [10, 150], [30, 150], [30, 100], [60, 100], [60, 150], [80, 150], [80, 80]],
        dtype=np.float32,
    )
    poly = Polygon("裂纹", 2, pts, "polygon")
    mask = polygon_visible_mask(poly, 0, 100, 100)
    out = clip_polygon_to_tile(pts, 0, 100, 100)
    canvas = np.zeros((100, 100), dtype=np.uint8)
    import cv2

    cv2.fillPoly(canvas, [np.round(out).astype(np.int32)], 1)
    inter = np.logical_and(canvas, mask).sum()
    union = np.logical_or(canvas, mask).sum()
    assert inter / union > 0.95
    assert (
        canvas[25, 20] == 1 and canvas[25, 70] == 1 and canvas[25, 45] == 0
    )  # both arms, not the gap
    assert canvas[75, 20] == 0  # below the arms nothing


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
