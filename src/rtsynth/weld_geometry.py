"""Where across the weld is a point? A normalised coordinate from the brightness profile.

The weld runs left to right on a landscape film. A *profile across the weld* is the brightness as a
function of the row, taken as the median of a vertical strip. In SWRD (more steel = brighter) the bead
is the bright hump in the middle, the parent metal the darker bands above and below it.

Bead geometry from one profile (the same rule for real and inserted defects, so that the two can be
compared):

  1. smooth the profile (Gaussian, sigma 6 rows);
  2. level = p5 + 0.5 x (p95 - p5); the *bead run* is the longest stretch of rows above that level;
  3. its two ends are the *toes* (where the bead meets the parent metal, at half height), the centre line
     is their midpoint and the half-width is half their distance;
  4. if the run touches one film border (the crop cut the parent metal off on that side), the centre is
     the peak of the smoothed profile inside the run and the half-width is the distance from the peak to
     the toe that is defined; if it touches both borders, or the profile is flat, the geometry is undefined.

The across-weld coordinate of a row y is  u = (y - centre) / half_width:  0 on the centre line, +-1 on the
toes, |u| > 1 in the parent metal. Up/down is a film convention, so distributions are compared on |u| as
well as on u.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BeadGeometry:
    centre: float  # row of the bead centre line
    half_width: float  # rows from the centre line to a toe
    top: int | None  # toe rows (None when that side is cut by the film border)
    bottom: int | None
    level: float  # grey level of the half-height threshold
    contrast: float  # p95 - p5 of the smoothed profile

    def u(self, y: float) -> float:
        return (y - self.centre) / self.half_width

    def row(self, u: float) -> float:
        return self.centre + u * self.half_width


def strip_profile(img: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Median over the columns [x0, x1) of every row, as float64 (length = image height)."""
    x0, x1 = max(0, int(x0)), min(img.shape[1], int(x1))
    if x1 <= x0:
        raise ValueError("empty strip")
    return np.median(img[:, x0:x1].astype(np.float64), axis=1)


def smooth_profile(prof: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    return cv2.GaussianBlur(prof.reshape(-1, 1).astype(np.float64), (0, 0), sigma).ravel()


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """[start, end) of the longest run of True; (0, 0) if none."""
    best, start = (0, 0, 0), None
    for i, v in enumerate(np.concatenate([mask, [False]])):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[0]:
                best = (i - start, start, i)
            start = None
    return best[1], best[2]


def bead_geometry(
    prof: np.ndarray,
    sigma: float = 6.0,
    min_half_width: float = 30.0,
    min_run: int = 40,
    max_run_frac: float = 0.97,
    min_contrast_frac: float = 0.03,
    peak_margin: int = 20,
) -> BeadGeometry | None:
    """Bead centre line and half-width from a profile across the weld (see the module docstring).

    Returns None when the geometry is undefined: a flat profile, a bead run shorter than ``min_run`` rows
    or longer than ``max_run_frac`` of the film, or a bead cut by both film borders.
    """
    sm = smooth_profile(prof, sigma)
    n = len(sm)
    lo, hi = np.percentile(sm, [5, 95])
    contrast = float(hi - lo)
    if hi <= 0 or contrast < min_contrast_frac * hi:
        return None
    level = float(lo + 0.5 * contrast)
    top, bottom = _longest_run(sm > level)
    run = bottom - top
    if run < min_run or run > max_run_frac * n:
        return None
    top_cut, bottom_cut = top == 0, bottom == n
    if top_cut and bottom_cut:
        return None
    if not top_cut and not bottom_cut:
        centre = (top + bottom - 1) / 2.0
        half = (bottom - 1 - top) / 2.0
        t, b = top, bottom - 1
    else:
        peak = top + int(np.argmax(sm[top:bottom]))
        if top_cut:
            if peak < peak_margin:
                return None
            centre, half, t, b = float(peak), float(bottom - 1 - peak), None, bottom - 1
        else:
            if peak > n - 1 - peak_margin:
                return None
            centre, half, t, b = float(peak), float(peak - top), top, None
    if half < min_half_width:
        return None
    return BeadGeometry(centre, half, t, b, level, contrast)


def geometry_at(img: np.ndarray, x0: int, x1: int, **kw) -> BeadGeometry | None:
    """Bead geometry of the strip [x0, x1) of a landscape film."""
    return bead_geometry(strip_profile(img, x0, x1), **kw)


def polygon_u(img: np.ndarray, points: np.ndarray, **kw) -> tuple[float, float, float] | None:
    """(u of the polygon's centroid row, u of its top row, u of its bottom row) or None if undefined.

    The strip is the polygon's own column range, widened to at least 40 columns.
    """
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    if x1 - x0 < 40:
        c = (x0 + x1) / 2
        x0, x1 = c - 20, c + 20
    g = geometry_at(img, int(np.floor(x0)), int(np.ceil(x1)) + 1, **kw)
    if g is None:
        return None
    ys = points[:, 1]
    return g.u(float(ys.mean())), g.u(float(ys.min())), g.u(float(ys.max()))
