"""Insert a real defect into another weld radiograph: physical (additive) vs naive (alpha blend).

Physical arm. Mery & Katsaggelos 2017 (CVPRW, eq. 11-13): in a space Z where attenuation adds,

    Z(film with defect) = Z(film without defect) + Z(defect alone)

The "defect alone" is not imaged on its own (a pore cannot be put in polystyrene). It is obtained by
removal, as Felix proposed on 2026-10-07: inpaint the defect away on the source film, and subtract the
cleaned film from the original in Z. The residual is then added to a defect-free host film.

Which Z. For SWRD the pore-dip test (experiments/wp2_physics/trial_log_compositing) found that the raw
16-bit grey value is already close to additive, so the default is the linear space Z = I. The paper's
space Z = log(I - B) is kept as an option for comparison.

Naive arm (the A/B control). The same source crop is alpha-blended onto the host in raw grey values,
with the same soft mask and at the same place. It carries the source film's background along.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AdditiveSpace:
    """Pixel space in which defects add. offset=None: linear (Z = I). Else Z = log(I - offset)."""

    offset: float | None = None

    @property
    def name(self) -> str:
        return "linear (Z = I)" if self.offset is None else f"log (Z = log(I - {self.offset:g}))"

    def to_z(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float64)
        if self.offset is None:
            return x
        return np.log(np.maximum(x - self.offset, 1.0))

    def from_z(self, z: np.ndarray) -> np.ndarray:
        return z if self.offset is None else np.exp(z) + self.offset


def to_uint16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 65535).astype(np.uint16)


def polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(points).astype(np.int32)], 1)
    return m


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def high_pass(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Film grain estimate: the image minus a Gaussian low-pass."""
    a = img.astype(np.float64)
    return a - cv2.GaussianBlur(a, (0, 0), sigma)


def grain_sigma(img: np.ndarray, exclude: np.ndarray | None = None) -> float:
    """Robust standard deviation of the grain (1.4826 x MAD of the high-pass)."""
    r = high_pass(img)
    v = r[exclude == 0] if exclude is not None else r.ravel()
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


@dataclass
class DefectPatch:
    """A defect cut out of a source film. All arrays share the crop's shape."""

    residual: np.ndarray  # Z(defect alone), float64, zero outside the support
    weight: np.ndarray  # soft support in [0, 1]; 1 on the defect, ramps to 0 over dilate_px
    label_mask: np.ndarray  # the annotated polygon, uint8
    source_crop: np.ndarray  # raw grey values of the source crop (used by the naive arm)
    cleaned_crop: np.ndarray  # source crop with the defect removed (inpainted), raw grey values
    origin: tuple[int, int]  # (x0, y0) of the crop in the source film
    points: np.ndarray  # polygon in crop coordinates
    grain: float  # grain sigma of the source film around the defect (grey levels)


def extract_defect(
    film: np.ndarray,
    points: np.ndarray,
    space: AdditiveSpace,
    dilate_px: int = 6,
    denoise_sigma: float = 1.0,
    sign: int = -1,
    margin: int = 24,
    grain_context: int = 150,
) -> DefectPatch:
    """Remove the defect by inpainting and return what was removed, in the additive space.

    sign=-1 keeps only darkening (missing steel: pores, cracks, slag, lack of fusion/penetration,
    undercut); this is the sign constraint of RQ3. sign=0 switches it off.
    denoise_sigma smooths the residual so that the source film's grain is not added on top of the
    host's own grain.
    The cleaned crop gets real grain back inside the removed area, transplanted from the nearest
    window of the same film beside the defect, so that removal does not leave a smooth patch.
    """
    h, w = film.shape
    x0, y0 = np.floor(points.min(axis=0)).astype(int) - margin
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + margin + 1
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    crop = film[y0:y1, x0:x1]
    pts = points - [x0, y0]
    lab = polygon_mask(crop.shape, pts)
    support = cv2.dilate(lab, _disk(dilate_px))

    z = space.to_z(crop)
    z_bg = cv2.inpaint(z.astype(np.float32), support, dilate_px + 3, cv2.INPAINT_TELEA)
    z_bg = z_bg.astype(np.float64)
    r = z - z_bg
    if denoise_sigma > 0:
        r = cv2.GaussianBlur(r, (0, 0), denoise_sigma)
    if sign < 0:
        r = np.minimum(r, 0.0)
    elif sign > 0:
        r = np.maximum(r, 0.0)

    # weight: 1 on the (slightly grown) label, linear ramp to 0 at the edge of the support. A distance
    # ramp, not a Gaussian blur of the mask, so that 2-px-wide cracks keep their full depth.
    core = cv2.dilate(lab, _disk(1))
    dist = cv2.distanceTransform((core == 0).astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip(1.0 - dist / dilate_px, 0.0, 1.0) * (support > 0)
    r = r * weight

    cw = crop.shape[1]
    grain_fill = np.zeros(crop.shape)
    for dx in (cw, -cw, 2 * cw, -2 * cw):  # first same-size window beside the crop that fits
        if 0 <= x0 + dx and x0 + dx + cw <= w:
            grain_fill = high_pass(film[y0:y1, x0 + dx : x0 + dx + cw])
            break
    cleaned = np.where(support > 0, space.from_z(z_bg) + grain_fill, crop.astype(np.float64))

    gy0, gy1 = max(0, y0 - grain_context), min(h, y1 + grain_context)
    gx0, gx1 = max(0, x0 - grain_context), min(w, x1 + grain_context)
    excl = np.zeros((gy1 - gy0, gx1 - gx0), np.uint8)
    excl[y0 - gy0 : y1 - gy0, x0 - gx0 : x1 - gx0] = support
    grain = grain_sigma(film[gy0:gy1, gx0:gx1], excl)
    return DefectPatch(
        r, weight, lab, crop.copy(), to_uint16(cleaned), (int(x0), int(y0)), pts, grain
    )


def contrast_scale(
    host: np.ndarray, patch: DefectPatch, top_left: tuple[int, int], context: int = 150
) -> float:
    """Host grain / source grain around the two locations.

    Films differ in scanner gain and exposure, so a dip of N grey levels on one film is not a dip of N
    on another. Scaling by the grain ratio keeps the defect's contrast-to-noise ratio, assuming both
    films share film type and gradient. Calibration from the image itself, no metadata.
    """
    x, y = top_left
    ph, pw = patch.residual.shape
    hh, hw = host.shape
    y0, y1 = max(0, y - context), min(hh, y + ph + context)
    x0, x1 = max(0, x - context), min(hw, x + pw + context)
    return grain_sigma(host[y0:y1, x0:x1]) / max(patch.grain, 1e-9)


def composite_physical(
    host: np.ndarray,
    patch: DefectPatch,
    top_left: tuple[int, int],
    space: AdditiveSpace,
    scale: float = 1.0,
) -> np.ndarray:
    """Add the (scaled) defect residual to the host in the additive space."""
    x, y = top_left
    ph, pw = patch.residual.shape
    out = host.copy()
    region = host[y : y + ph, x : x + pw]
    out[y : y + ph, x : x + pw] = to_uint16(
        space.from_z(space.to_z(region) + scale * patch.residual)
    )
    return out


def composite_naive(host: np.ndarray, patch: DefectPatch, top_left: tuple[int, int]) -> np.ndarray:
    """Alpha-blend the raw source crop onto the host (grey values, same soft mask)."""
    x, y = top_left
    ph, pw = patch.residual.shape
    out = host.copy()
    region = host[y : y + ph, x : x + pw].astype(np.float64)
    a = patch.weight
    out[y : y + ph, x : x + pw] = to_uint16((1 - a) * region + a * patch.source_crop)
    return out


def _zscore(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0


def place_by_profile(
    host: np.ndarray,
    keepout: np.ndarray,
    cleaned_source: np.ndarray,
    patch: DefectPatch,
    rng: np.random.Generator,
    n_columns: int = 400,
    half_window: int = 150,
    end_margin: float = 0.05,
    n_slices: int = 3,
) -> tuple[tuple[int, int], float]:
    """Choose where the patch goes on the host.

    The defect keeps its position relative to the weld bead: the host row is the one where the host's
    brightness profile across the weld best matches the source's profile around the defect
    (normalised cross-correlation of z-scored column-median profiles; the source profile is taken on
    the film with the defect already removed). The patch width is cut into n_slices vertical slices and
    the score is the worst slice, so a spot straddling a brightness step along the weld scores low.
    Columns are sampled at random along the weld; positions touching a keep-out zone are rejected.
    Returns ((x, y) top-left of the patch on the host, score).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    ph, pw = patch.residual.shape
    sx0, sy0 = patch.origin
    sh = cleaned_source.shape[0]
    cy_crop = round(float(patch.points[:, 1].mean()))
    cy_src = sy0 + cy_crop
    a, b = max(0, cy_src - half_window), min(sh, cy_src + half_window)
    prof_s = np.median(cleaned_source[:, sx0 : sx0 + pw].astype(np.float64), axis=1)[a:b]
    prof_s = _zscore(prof_s)
    n = len(prof_s)
    off_a = cy_src - a  # defect centre row inside the source window

    hh, hw = host.shape
    xs = rng.integers(int(end_margin * hw), int((1 - end_margin) * hw) - pw, size=n_columns)
    support = patch.weight > 0
    edges = np.linspace(0, pw, n_slices + 1).astype(int)
    best = (-np.inf, None)
    for x in xs:
        score = None
        for k in range(n_slices):
            cols = host[:, x + edges[k] : x + edges[k + 1]].astype(np.float64)
            win = sliding_window_view(np.median(cols, axis=1), n)  # (hh - n + 1, n)
            mu, sd = win.mean(1, keepdims=True), win.std(1, keepdims=True) + 1e-9
            sc = ((win - mu) / sd) @ prof_s / n
            score = sc if score is None else np.minimum(score, sc)
        for start in np.argsort(-score)[:30]:  # window start row -> patch top row
            if score[start] <= best[0]:
                break
            y = start + off_a - cy_crop
            if y < 0 or y + ph > hh:
                continue
            if keepout[y : y + ph, x : x + pw][support].any():
                continue
            best = (float(score[start]), (int(x), int(y)))
            break
    if best[1] is None:
        raise RuntimeError("no valid placement found")
    return best[1], best[0]
