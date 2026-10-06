"""Unit tests for the customer-film export helpers (scripts/_customer.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import LABEL_TO_CLASS, OTHER_DEFECT_LABEL, load_polygons
from _customer import (
    crop_rect,
    labelme_document,
    mask_to_shapes,
    seam_rect_with_margin,
)

NAMES = {
    1: "porosity",
    2: "inclusion",
    3: "crack",
    4: "undercut",
    5: "lack_of_fusion",
    6: "lack_of_penetration",
}


def test_mask_to_shapes_one_polygon_per_component_and_class():
    m = np.zeros((40, 60), np.uint8)
    m[5:15, 5:15] = 1  # porosity
    m[20:30, 5:15] = 1  # second porosity
    m[5:35, 30:50] = 6  # lack of penetration
    shapes = mask_to_shapes(m, NAMES)
    labels = sorted(s["label"] for s in shapes)
    assert labels == ["lack_of_penetration", "porosity", "porosity"]
    assert all(s["shape_type"] == "polygon" and len(s["points"]) >= 3 for s in shapes)


def test_shapes_rasterise_back_to_the_mask():
    m = np.zeros((50, 50), np.uint8)
    cv2.circle(m, (25, 25), 12, 3, -1)  # a crack-shaped blob
    (shape,) = mask_to_shapes(m, NAMES)
    back = np.zeros_like(m)
    cv2.fillPoly(back, [np.round(np.array(shape["points"])).astype(np.int32)], 1)
    inter = ((back > 0) & (m > 0)).sum()
    union = ((back > 0) | (m > 0)).sum()
    assert inter / union > 0.9  # boundary pixels only differ


def test_mask_to_shapes_skips_specks_and_background():
    m = np.zeros((20, 20), np.uint8)
    m[3, 3] = 2  # one pixel: no 3-point contour
    assert mask_to_shapes(m, NAMES) == []
    assert mask_to_shapes(np.zeros((20, 20), np.uint8), NAMES) == []


def test_labelme_document_round_trips_through_the_paper_loader(tmp_path):
    m = np.zeros((30, 30), np.uint8)
    m[2:10, 2:10] = 4  # undercut
    other = np.zeros((30, 30), np.uint8)
    other[20:28, 20:28] = 1
    shapes = mask_to_shapes(m, NAMES) + mask_to_shapes(other, {1: OTHER_DEFECT_LABEL})
    doc = labelme_document("oge__abc", 30, 30, shapes)
    import json

    p = tmp_path / "oge__abc.json"
    p.write_text(json.dumps(doc))
    polys, header = load_polygons(p)
    assert header["imageHeight"] == 30 and header["imagePath"] == "oge__abc.tif"
    by_label = {q.label: q for q in polys}
    assert by_label["undercut"].class_id == LABEL_TO_CLASS["undercut"] == 3
    assert by_label[OTHER_DEFECT_LABEL].class_id is None  # excluded from negatives, never a target
    x0, y0, x1, y1 = by_label["undercut"].bbox
    assert (x0, y0) == (2, 2) and 9 <= x1 <= 10 and 9 <= y1 <= 10


def test_seam_rect_pads_by_fraction_of_short_side():
    seam = np.zeros((100, 400), np.uint8)
    seam[40:60, 50:350] = 1  # 20 px tall, 300 px wide -> pad = 0.1 * 20 = 2
    (rect, pad) = seam_rect_with_margin(seam, 0.1, seam.shape)
    assert pad == 2 and rect == (38, 48, 62, 352)


def test_crop_rect_grows_to_keep_labels():
    seam = np.zeros((100, 400), np.uint8)
    seam[40:60, 50:350] = 1
    gt = np.zeros_like(seam)
    gt[45:50, 100:110] = 1  # inside
    rect, status = crop_rect(seam, gt, 0.1, seam.shape)
    assert status == "contained" and rect == (38, 48, 62, 352)
    gt[80:90, 100:110] = 1  # an off-seam defect below the strip
    rect, status = crop_rect(seam, gt, 0.1, seam.shape)
    assert status == "extended" and rect == (38, 48, 92, 352)


def test_crop_rect_without_seam_is_none():
    assert crop_rect(None, None, 0.1, (10, 10)) == (None, "no_seam")
    assert crop_rect(np.zeros((10, 10), np.uint8), None, 0.1, (10, 10)) == (None, "no_seam")


def test_instance_contrast_sign_follows_polarity():
    from _customer import instance_contrast, region_contrast

    img = np.full((60, 60), 30000, np.uint16)
    m = np.zeros((60, 60), np.uint8)
    cv2.circle(m, (20, 20), 6, 1, -1)
    cv2.circle(m, (45, 45), 6, 1, -1)
    img[m > 0] = 20000  # dark voids, SWRD polarity
    d = instance_contrast(img, m)
    assert len(d) == 2 and all(v < 0 for v in d)
    inv = img.max() - img  # wrong polarity
    assert all(v > 0 for v in instance_contrast(inv, m))
    seam = np.zeros((60, 60), np.uint8)
    seam[20:40, :] = 1
    img2 = np.full((60, 60), 10000, np.uint16)
    img2[seam > 0] = 15000
    assert region_contrast(img2, seam) > 0
    assert region_contrast(img2, np.ones_like(seam)) is None
