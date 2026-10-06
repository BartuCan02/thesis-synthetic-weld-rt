"""Pure helpers for export_customer_films.py (no deeplify, Mongo or S3 imports, so they are unit-testable here).

A customer film enters the paper pipeline as a fake SWRD release pair: a 16-bit seam-cropped TIFF and a
LabelMe JSON. These helpers turn an indexed class mask into LabelMe polygons and decide the crop rectangle.
"""

from __future__ import annotations

import cv2
import numpy as np

Rect = tuple[int, int, int, int]  # y1, x1, y2, x2 (exclusive end), the deeplify convention


def mask_to_shapes(mask: np.ndarray, names: dict[int, str], min_area_px: int = 1) -> list[dict]:
    """One LabelMe polygon per connected component of each class in an indexed mask.

    ``names`` maps pixel value -> label string; value 0 and unlisted values are ignored. Holes are
    dropped (RETR_EXTERNAL): the paper pipeline only ever needs the outer boundary, because it
    rasterises the polygon and takes the bounding box of the visible part. Components with fewer than
    three contour points or less than ``min_area_px`` pixels are skipped (they cannot form a box).
    """
    shapes: list[dict] = []
    for value, label in sorted(names.items()):
        if value == 0:
            continue
        binary = (mask == value).astype(np.uint8)
        if not binary.any():
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) < 3 or cv2.contourArea(c) < min_area_px:
                continue
            pts = c.reshape(-1, 2).astype(float).tolist()
            shapes.append({"label": label, "points": pts, "shape_type": "polygon"})
    return shapes


def labelme_document(stem: str, height: int, width: int, shapes: list[dict]) -> dict:
    """A LabelMe file as the paper pipeline's ``load_polygons`` expects it."""
    return {
        "version": "swrd-paper-baseline customer export",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{stem}.tif",
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def bbox_of(mask: np.ndarray) -> Rect | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def seam_rect_with_margin(
    seam: np.ndarray, margin_frac: float, shape: tuple[int, ...]
) -> tuple[Rect, int] | None:
    """Seam bounding box padded by ``margin_frac`` x the seam's short side (deeplify pipeline-v2 rule).

    Returns (rect, pad) or None when the seam mask is empty.
    """
    box = bbox_of(seam > 0)
    if box is None:
        return None
    y1, x1, y2, x2 = box
    pad = round(margin_frac * min(y2 - y1, x2 - x1))
    h, w = shape[:2]
    return (max(0, y1 - pad), max(0, x1 - pad), min(h, y2 + pad), min(w, x2 + pad)), pad


def crop_rect(
    seam: np.ndarray | None, gt: np.ndarray | None, margin_frac: float, shape: tuple[int, ...]
) -> tuple[Rect | None, str]:
    """The weld crop for one film and how it was obtained.

    - no seam           -> (None, "no_seam"): the caller skips the film (a whole film is not a weld strip,
                           and the paper's window rule — side = half the shorter side — would be meaningless)
    - seam, GT inside   -> (rect, "contained")
    - seam, GT outside  -> (rect grown to include the GT box with the same pad, "extended"). SWRD's crops
                           contain every annotation by construction; a training film must not lose labels.
    """
    sr = seam_rect_with_margin(seam, margin_frac, shape) if seam is not None else None
    if sr is None:
        return None, "no_seam"
    rect, pad = sr
    gt_box = bbox_of(gt > 0) if gt is not None else None
    if gt_box is None:
        return rect, "contained"
    h, w = shape[:2]
    gy1, gx1, gy2, gx2 = gt_box
    gt_rect = (max(0, gy1 - pad), max(0, gx1 - pad), min(h, gy2 + pad), min(w, gx2 + pad))
    y1, x1, y2, x2 = rect
    if y1 <= gt_rect[0] and x1 <= gt_rect[1] and y2 >= gt_rect[2] and x2 >= gt_rect[3]:
        return rect, "contained"
    union = (
        min(y1, gt_rect[0]),
        min(x1, gt_rect[1]),
        max(y2, gt_rect[2]),
        max(x2, gt_rect[3]),
    )
    return union, "extended"


def resize_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bring a label map to (height, width) without interpolating class values."""
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def instance_contrast(
    image: np.ndarray, mask: np.ndarray, ring_px: int = 5, min_px: int = 4
) -> list[float]:
    """Per connected component of ``mask``: mean(inside) - mean(ring around it), in raw grey units.

    Polarity check. In SWRD's convention (more metal = brighter) a pore or slag inclusion is a void, so this
    difference is negative. Measured on 300 SWRD films (2026-10-05): negative for 98.4 % of pores and 95 % of
    inclusions, median -741 / -629 grey levels. A source whose exported films do not show the same sign has the
    wrong polarity. Components smaller than ``min_px`` are skipped; the ring is a (2*ring_px+1) dilation minus the component.
    """
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    k = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    img = image.astype(np.float32)
    out: list[float] = []
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        if comp.sum() < min_px:
            continue
        ring = cv2.dilate(comp, k) - comp
        if ring.sum() == 0:
            continue
        out.append(float(img[comp > 0].mean() - img[ring > 0].mean()))
    return out


def region_contrast(image: np.ndarray, region: np.ndarray) -> float | None:
    """mean(inside region) - mean(outside), or None if the region is empty or fills the image.

    For the weld seam this is positive in SWRD's polarity (the reinforcement is extra metal): 91.6 % of 300 SWRD
    films, median +7,611 grey levels.
    """
    r = region > 0
    if not r.any() or r.all():
        return None
    img = image.astype(np.float32)
    return float(img[r].mean() - img[~r].mean())
