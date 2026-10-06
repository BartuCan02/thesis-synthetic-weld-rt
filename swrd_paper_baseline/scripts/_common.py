"""Shared constants and helpers for the SWRD paper-baseline pipeline.

Everything here follows Zhao et al. 2025 (J. Nondestructive Evaluation 44:50). Section numbers in the
docstrings refer to that paper. Values the paper does *not* state are marked UNSPECIFIED and are passed
in as explicit CLI flags by the scripts, so that every choice is visible in the ClearML dataset metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

# ---------------------------------------------------------------------------
# Classes (Sect. 2, Fig. 4, Table 2). Index = YOLO class id.
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = [
    "porosity",
    "inclusion",
    "crack",
    "undercut",
    "lack_of_fusion",
    "lack_of_penetration",
]

# Chinese LabelMe label string -> class id. The release has more label strings than the paper's six
# classes (9 defect labels + seam + pseudo-defect, see CLAUDE.md). Only the ones below become training
# labels; 00_inventory.py prints every other string it meets so the mapping can be checked against Fig. 4.
# UNSPECIFIED: whether 夹钨 (tungsten inclusion) is counted under "inclusion". Fig. 4 says 2,017 inclusions;
# the inventory decides by matching that number.
LABEL_TO_CLASS: dict[str, int] = {
    "气孔": 0,  # porosity
    "夹渣": 1,  # inclusion (slag)
    "夹钨": 1,  # inclusion (tungsten) — see note above
    "裂纹": 2,  # crack
    "咬边": 3,  # undercut
    "未熔合": 4,  # lack of fusion
    "未焊透": 5,  # lack of penetration
    # English canonical names, written by export_customer_films.py for Deeplify's customer films.
    "porosity": 0,
    "inclusion": 1,
    "crack": 2,
    "undercut": 3,
    "lack_of_fusion": 4,
    "lack_of_penetration": 5,
}
#: Label written by export_customer_films.py for weld-defect classes outside the six (burn-through,
#: spatter, excess material, ...). Never a training label; tiles it touches are excluded from the
#: negatives with ``01_tile.py --exclude-from-negatives other_defect``.
OTHER_DEFECT_LABEL = "other_defect"

# Paper's numbers we test ourselves against.
FIG4_INSTANCES_ORIGINAL: dict[str, int] = {  # Fig. 4, on the original films
    "porosity": 25401,
    "inclusion": 2017,
    "crack": 1754,
    "undercut": 213,
    "lack_of_fusion": 802,
    "lack_of_penetration": 745,
}
FIG4_MEAN_AREA_PX2: dict[str, float] = {
    "porosity": 1619.33,
    "inclusion": 3363.07,
    "crack": 13195.21,
    "undercut": 8883.29,
    "lack_of_fusion": 17657.22,
    "lack_of_penetration": 54412.66,
}
TABLE2_INSTANCES_TILES: dict[str, int] = {  # Table 2, counted on tiles (overlaps count twice)
    "porosity": 102927,
    "inclusion": 9141,
    "crack": 13515,
    "undercut": 1015,
    "lack_of_fusion": 6862,
    "lack_of_penetration": 20422,
}
PAPER_N_WELD_IMAGES = 4930  # Sect. 3.1
PAPER_N_TILES_MIN = 380_000  # Sect. 3.2 "over 380,000"
PAPER_N_POSITIVE_TILES = 80_648  # Sect. 3.2
TABLE3_SPLIT = {  # Sect. 4.1
    "train_defect": 72585,
    "train_background": 72646,
    "val_defect": 8099,
    "val_background": 8038,
}

# ---------------------------------------------------------------------------
# Raw release layout (mirror of s3://swdr/cropped/ on the box)
# ---------------------------------------------------------------------------

IMAGES_SUBDIR = "crop_weld_images"
JSONS_SUBDIR = "crop_weld_jsons"
TIF_SUFFIXES = {".tif", ".tiff"}


def exposure_id(stem: str) -> str:
    """Original-film id for a cropped image stem.

    T-joint films were cut into a primary and a secondary weld image (Sect. 3.1), named ``A_<film>`` and
    ``B_<film>`` in the release. Both halves come from one exposure, so a film-level split must keep them
    together. Standard films keep their own stem.
    """
    if stem[:2] in ("A_", "B_"):
        return stem[2:]
    return stem


def sha1_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tif_shape(path: Path) -> tuple[int, int, str]:
    """(height, width, dtype) from the TIFF header without decoding pixels."""
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        h, w = page.shape[0], page.shape[1]
        return int(h), int(w), str(page.dtype)


def read_tif(path: Path) -> np.ndarray:
    """Full 2-D array (uint16 expected). A trailing singleton channel axis is squeezed."""
    try:
        img = tifffile.imread(path)
    except ValueError:  # codec missing (e.g. LZW without imagecodecs): let libtiff via OpenCV try
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D grayscale image, got shape {img.shape}")
    return img


# ---------------------------------------------------------------------------
# LabelMe polygons
# ---------------------------------------------------------------------------


@dataclass
class Polygon:
    label: str  # raw label string from the JSON
    class_id: int | None  # None = not one of the six classes
    points: np.ndarray  # (N, 2) float32, x then y, in cropped-image pixels
    shape_type: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x0, y0 = self.points.min(axis=0)
        x1, y1 = self.points.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    @property
    def area(self) -> float:
        """Geometric polygon area in px^2 (what Fig. 4 calls 'pixel area')."""
        if len(self.points) < 3:
            x0, y0, x1, y1 = self.bbox
            return (x1 - x0) * (y1 - y0)
        return float(cv2.contourArea(self.points.astype(np.float32)))


def load_polygons(json_path: Path) -> tuple[list[Polygon], dict]:
    """Parse a LabelMe file. Returns (polygons, header) where header has imageHeight/imageWidth/imagePath."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    polys: list[Polygon] = []
    for shape in data.get("shapes", []):
        label = (shape.get("label") or "").strip()
        pts = np.asarray(shape.get("points", []), dtype=np.float32)
        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "rectangle" and len(pts) == 2:  # LabelMe stores rectangles as two corners
            (x0, y0), (x1, y1) = pts
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        if len(pts) == 0:
            continue
        polys.append(Polygon(label, LABEL_TO_CLASS.get(label), pts, shape_type))
    header = {k: data.get(k) for k in ("imageHeight", "imageWidth", "imagePath")}
    return polys, header


# ---------------------------------------------------------------------------
# Sliding window (Sect. 3.2)
# ---------------------------------------------------------------------------


def tile_side(height: int, width: int, side_fraction: float = 0.5) -> int:
    """Window side = ``side_fraction`` x the image's shorter side, rounded to the nearest pixel.

    Paper: one half. Rounding is not stated; the 2026-10-02 sweep (results/grid_sweep_2026-10-02.json) shows
    that round-to-nearest for the side, round-to-nearest for the stride and counting a window that ends
    exactly on the border reproduce the paper's tile total (412,726 vs "over 380,000") and its 80,648 defect
    tiles within 1 %, while flooring the side gives 441,304 / 84,874. Python's round() (half to even) is
    what the sweep used.
    """
    return max(1, round(min(height, width) * side_fraction))


def tile_positions(length: int, side: int, overlap: float = 0.5, edge: str = "flush") -> list[int]:
    """Top-left offsets of windows along one axis.

    overlap 0.5 -> stride = side / 2 (paper: "50 % overlap").
    edge: how to treat the remainder when the last regular window does not reach the image border.
      "flush" adds one extra window ending exactly at the border (no pixel is left uncovered);
      "drop" keeps only windows that fit the regular grid.
    UNSPECIFIED in the paper; both are reported by 00_inventory.py so the choice can be made on the count.
    """
    if length <= side:
        return [0]
    stride = max(1, round(side * (1.0 - overlap)))
    positions = list(range(0, length - side + 1, stride))
    if edge == "flush" and positions[-1] != length - side:
        positions.append(length - side)
    elif edge not in ("flush", "drop"):
        raise ValueError(f"edge must be 'flush' or 'drop', got {edge!r}")
    return positions


def count_tiles(height: int, width: int, side_fraction: float, overlap: float, edge: str) -> int:
    side = tile_side(height, width, side_fraction)
    return len(tile_positions(width, side, overlap, edge)) * len(
        tile_positions(height, side, overlap, edge)
    )


# ---------------------------------------------------------------------------
# Polygon -> box inside a tile (UNSPECIFIED in the paper; decision D7)
# ---------------------------------------------------------------------------


@dataclass
class TileBox:
    class_id: int
    x0: int  # tile-local pixel coords, inclusive-exclusive like numpy slices
    y0: int
    x1: int
    y1: int
    visible_frac: float  # visible polygon area / full polygon area

    def to_yolo(self, side: int) -> str:
        cx = (self.x0 + self.x1) / 2.0 / side
        cy = (self.y0 + self.y1) / 2.0 / side
        w = (self.x1 - self.x0) / side
        h = (self.y1 - self.y0) / side
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def polygon_visible_mask(poly: Polygon, x: int, y: int, side: int) -> np.ndarray | None:
    """Rasterise the polygon into a side x side canvas placed at (x, y). None if the bbox misses the tile."""
    bx0, by0, bx1, by1 = poly.bbox
    if bx1 < x or by1 < y or bx0 > x + side or by0 > y + side:
        return None
    canvas = np.zeros((side, side), dtype=np.uint8)
    shifted = np.round(poly.points - np.array([x, y], dtype=np.float32)).astype(np.int32)
    if len(shifted) >= 3:
        cv2.fillPoly(canvas, [shifted], 1)
    else:  # degenerate 1–2 point shapes: mark the clipped bbox
        x0 = int(np.clip(np.floor(bx0 - x), 0, side))
        x1 = int(np.clip(np.ceil(bx1 - x), 0, side))
        y0 = int(np.clip(np.floor(by0 - y), 0, side))
        y1 = int(np.clip(np.ceil(by1 - y), 0, side))
        canvas[y0:y1, x0:x1] = 1
    return canvas


def box_from_visible(
    poly: Polygon,
    mask: np.ndarray,
    min_side_px: int,
    min_visible_frac: float,
    min_tile_frac: float,
) -> TileBox | None:
    """Turn the visible part of a polygon into a box, or None if it fails the keep rule.

    Keep rule (D7, UNSPECIFIED in the paper):
      both box sides >= min_side_px, AND
      (visible_frac >= min_visible_frac  OR  visible box area >= min_tile_frac x tile area).
    The second branch exists for long defects (lack of penetration, lack of fusion) that span many tiles:
    each tile sees a small *fraction* of them but a large *absolute* piece, which is still a real target.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    w, h = x1 - x0, y1 - y0
    if w < min_side_px or h < min_side_px:
        return None
    full_area = max(poly.area, 1.0)
    visible_frac = float(mask.sum()) / full_area
    tile_area = mask.shape[0] * mask.shape[1]
    if visible_frac < min_visible_frac and (w * h) < min_tile_frac * tile_area:
        return None
    return TileBox(poly.class_id, x0, y0, x1, y1, min(visible_frac, 1.0))


# ---------------------------------------------------------------------------
# Preprocessing (Sect. 3.3). Parameters UNSPECIFIED -> flags with D5/D6 defaults.
# ---------------------------------------------------------------------------


def contrast_stretch_to_uint8(tile16: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    """Linear stretch between two percentiles of *this tile*, then 16-bit -> 8-bit.

    p_lo = 0, p_hi = 100 gives a plain min-max stretch (the alternative to ablate).
    """
    lo, hi = np.percentile(tile16, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros(tile16.shape, dtype=np.uint8)
    out = (tile16.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_uint8(tile8: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return clahe.apply(tile8)


def preprocess_tile(
    tile16: np.ndarray, p_lo: float, p_hi: float, clip_limit: float, grid: int
) -> np.ndarray:
    """Sect. 3.3 in order: contrast stretch -> 8-bit -> CLAHE. Returns a single-channel uint8 tile."""
    return clahe_uint8(contrast_stretch_to_uint8(tile16, p_lo, p_hi), clip_limit, grid)


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def dataclass_dict(obj) -> dict:
    d = asdict(obj)
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return d
