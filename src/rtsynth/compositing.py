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

import math
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
    compensation: float = 1.0  # factor that restored the depth lost to denoise_sigma (1 = none)


def extract_defect(
    film: np.ndarray,
    points: np.ndarray,
    space: AdditiveSpace,
    dilate_px: int = 6,
    denoise_sigma: float = 1.0,
    sign: int = -1,
    margin: int = 24,
    grain_context: int = 150,
    compensate: float = 0.0,
) -> DefectPatch:
    """Remove the defect by inpainting and return what was removed, in the additive space.

    sign=-1 keeps only darkening (missing steel: pores, cracks, slag, lack of fusion/penetration,
    undercut); this is the sign constraint of RQ3. sign=0 switches it off.
    denoise_sigma smooths the residual so that the source film's grain is not added on top of the
    host's own grain.
    The cleaned crop gets real grain back inside the removed area, transplanted from the nearest
    window of the same film beside the defect, so that removal does not leave a smooth patch.
    compensate > 0 (version 3): the blur makes thin defects shallower (a 2 px crack keeps about 70 % of
    its peak depth). The deepest decile inside the labelled polygon is measured on a barely smoothed
    residual (sigma 0.5) and on the blurred one, and the residual is multiplied by their ratio, clipped to
    [1, compensate]. Grain inflates the reference by a few percent on faint, wide defects (known bias,
    bounded by the clip). 0 switches it off.
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
    if sign < 0:
        r = np.minimum(r, 0.0)
    elif sign > 0:
        r = np.maximum(r, 0.0)
    comp = 1.0
    if denoise_sigma > 0:
        r_raw = r
        r = cv2.GaussianBlur(r, (0, 0), denoise_sigma)
        if sign < 0:
            r = np.minimum(r, 0.0)
        elif sign > 0:
            r = np.maximum(r, 0.0)
        if compensate > 0 and lab.any():
            # depth = deepest decile inside the polygon (the peak a detector sees), measured on a barely
            # smoothed residual (sigma 0.5: pixel noise halved, a 2 px line keeps > 90 % of its peak) and
            # on the blurred one; the ratio is what the blur took from the peak. The median would not see
            # it (a blur spreads a thin line, it does not lower its median over a wider polygon).
            r_ref = cv2.GaussianBlur(r_raw, (0, 0), 0.5)
            before = float(np.percentile(np.abs(r_ref[lab > 0]), 90))
            after = float(np.percentile(np.abs(r[lab > 0]), 90))
            if after > 0:
                comp = float(np.clip(before / after, 1.0, compensate))
                r = r * comp

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
        r, weight, lab, crop.copy(), to_uint16(cleaned), (int(x0), int(y0)), pts, grain, comp
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


@dataclass(frozen=True)
class SourceProfile:
    """Brightness profile across the weld around a defect, on the source film with the defect removed."""

    values: np.ndarray  # z-scored column-median profile, length <= 2 * half_window
    off: int  # row of the defect centre inside ``values``
    cy_crop: int  # row of the defect centre inside the patch


def source_profile(
    cleaned_source: np.ndarray, patch: DefectPatch, half_window: int = 150
) -> SourceProfile:
    pw = patch.residual.shape[1]
    sx0, sy0 = patch.origin
    sh = cleaned_source.shape[0]
    cy_crop = round(float(patch.points[:, 1].mean()))
    cy_src = sy0 + cy_crop
    a, b = max(0, cy_src - half_window), min(sh, cy_src + half_window)
    prof = np.median(cleaned_source[:, sx0 : sx0 + pw].astype(np.float64), axis=1)[a:b]
    return SourceProfile(_zscore(prof), cy_src - a, cy_crop)


def flip_patch(patch: DefectPatch) -> DefectPatch:
    """Mirror a patch left-right (along the weld). The profile across the weld is unchanged."""
    pw = patch.residual.shape[1]
    pts = patch.points.copy()
    pts[:, 0] = pw - 1 - pts[:, 0]
    return DefectPatch(
        patch.residual[:, ::-1].copy(),
        patch.weight[:, ::-1].copy(),
        patch.label_mask[:, ::-1].copy(),
        patch.source_crop[:, ::-1].copy(),
        patch.cleaned_crop[:, ::-1].copy(),
        patch.origin,
        pts,
        patch.grain,
        patch.compensation,
    )


def place_with_profile(
    host: np.ndarray,
    keepout: np.ndarray,
    prof: SourceProfile,
    patch: DefectPatch,
    rng: np.random.Generator,
    n_columns: int = 400,
    end_margin: float = 0.05,
    n_slices: int = 3,
) -> tuple[tuple[int, int], float]:
    """Choose where the patch goes on the host (see ``place_by_profile``)."""
    from numpy.lib.stride_tricks import sliding_window_view

    ph, pw = patch.residual.shape
    prof_s, off_a, cy_crop = prof.values, prof.off, prof.cy_crop
    n = len(prof_s)
    hh, hw = host.shape
    lo, hi = int(end_margin * hw), int((1 - end_margin) * hw) - pw
    if hi <= lo or hh < n or hh < ph:
        raise RuntimeError("host too small for this patch")
    xs = rng.integers(lo, hi, size=n_columns)
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
    prof = source_profile(cleaned_source, patch, half_window)
    return place_with_profile(host, keepout, prof, patch, rng, n_columns, end_margin, n_slices)


# ---------------------------------------------------------------------------------------------------
# version 3 (2026-10-16): vary the defect, null implants, placement by bead geometry
# ---------------------------------------------------------------------------------------------------


def vflip_patch(patch: DefectPatch) -> DefectPatch:
    """Mirror a patch top-bottom (across the weld): the upper toe becomes the lower toe."""
    ph = patch.residual.shape[0]
    pts = patch.points.copy()
    pts[:, 1] = ph - 1 - pts[:, 1]
    return DefectPatch(
        patch.residual[::-1, :].copy(),
        patch.weight[::-1, :].copy(),
        patch.label_mask[::-1, :].copy(),
        patch.source_crop[::-1, :].copy(),
        patch.cleaned_crop[::-1, :].copy(),
        patch.origin,
        pts,
        patch.grain,
        patch.compensation,
    )


@dataclass(frozen=True)
class Augmentation:
    """One random variation of a cut-out defect (Tyystjarvi 2022: affine + flips; Virkkunen 2021:
    amplitude). The residual is a thickness perturbation, so a scaled, stretched or turned residual is
    still a physically valid defect; additivity and the sign constraint are untouched."""

    scale: float = 1.0  # isotropic size factor
    aspect: float = 1.0  # x is stretched by sqrt(aspect), y squeezed by the same
    angle_deg: float = 0.0  # rotation, counter-clockwise
    amplitude: float = 1.0  # depth factor on the residual
    hflip: bool = False
    vflip: bool = False

    @staticmethod
    def sample(
        rng: np.random.Generator,
        elongated: bool,
        scale: tuple[float, float] = (0.7, 1.4),
        aspect: tuple[float, float] = (0.8, 1.25),
        angle_deg: float = 5.0,
        amplitude: tuple[float, float] = (0.7, 1.3),
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ) -> Augmentation:
        """Elongated classes (cracks, undercut, lack of fusion) turn by at most ``angle_deg`` so they keep
        running along the weld; round ones (inclusions, pores) may take any angle."""
        return Augmentation(
            scale=float(rng.uniform(*scale)),
            aspect=float(rng.uniform(*aspect)),
            angle_deg=float(
                rng.uniform(-angle_deg, angle_deg) if elongated else rng.uniform(0, 360)
            ),
            amplitude=float(rng.uniform(*amplitude)),
            hflip=bool(rng.random() < hflip_prob),
            vflip=bool(rng.random() < vflip_prob),
        )

    def as_dict(self) -> dict:
        return {
            "scale": round(self.scale, 4),
            "aspect": round(self.aspect, 4),
            "angle_deg": round(self.angle_deg, 2),
            "amplitude": round(self.amplitude, 4),
            "hflip": self.hflip,
            "vflip": self.vflip,
        }


def augment_patch(patch: DefectPatch, aug: Augmentation) -> DefectPatch:
    """Apply one Augmentation to every array of the patch and to its polygon, consistently.

    The crop is resampled onto a canvas that holds the whole transformed crop (so a turned crack is not
    cut), the polygon is moved with the same matrix. Residual, weight and the two grey crops use linear
    interpolation; the label mask nearest-neighbour. Outside the original crop the residual and weight are
    0 and the grey crops reflect their border.
    """
    ph, pw = patch.residual.shape
    sx = aug.scale * math.sqrt(aug.aspect)
    sy = aug.scale / math.sqrt(aug.aspect)
    th = math.radians(aug.angle_deg)
    c, s = math.cos(th), math.sin(th)
    lin = np.array([[c * sx, -s * sy], [s * sx, c * sy]], np.float64)  # rotation x scale
    corners = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float64)
    moved = corners @ lin.T
    mn, mx = moved.min(0), moved.max(0)
    nw, nh = math.ceil(mx[0] - mn[0]) + 1, math.ceil(mx[1] - mn[1]) + 1
    M = np.hstack([lin, (-mn).reshape(2, 1)])  # 2x3 affine: out = lin @ in - mn

    def warp(a: np.ndarray, interp: int, border: int, value: float = 0.0) -> np.ndarray:
        return cv2.warpAffine(a, M, (nw, nh), flags=interp, borderMode=border, borderValue=value)

    residual = warp(patch.residual, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT) * aug.amplitude
    weight = np.clip(warp(patch.weight, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT), 0.0, 1.0)
    residual = np.where(weight > 0, residual, 0.0)
    label = warp(patch.label_mask, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT).astype(np.uint8)
    src = warp(patch.source_crop, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
    cleaned = warp(patch.cleaned_crop, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
    pts = patch.points @ lin.T - mn
    out = DefectPatch(
        residual, weight, label, src, cleaned, patch.origin, pts, patch.grain, patch.compensation
    )
    if aug.hflip:
        out = flip_patch(out)
    if aug.vflip:
        out = vflip_patch(out)
    return out


def null_implant(
    host: np.ndarray,
    support: np.ndarray,
    top_left: tuple[int, int],
    space: AdditiveSpace,
    inpaint_radius: int = 9,
) -> np.ndarray:
    """Run the removal step on a clean spot of the host and add nothing (Virkkunen 2021's "virtual
    unflawed" implant). The region under ``support`` is inpainted and refilled with grain from the window
    beside it, exactly as extract_defect does on a source. Tiles of such spots are negatives that carry
    the pipeline's processing traces, so a detector cannot use those traces as the cue for "defect".
    """
    x, y = top_left
    ph, pw = support.shape
    _hh, hw = host.shape
    out = host.copy()
    region = host[y : y + ph, x : x + pw]
    sup = (support > 0).astype(np.uint8)
    z = space.to_z(region)
    z_bg = cv2.inpaint(z.astype(np.float32), sup, inpaint_radius, cv2.INPAINT_TELEA).astype(
        np.float64
    )
    grain_fill = np.zeros(region.shape)
    for dx in (pw, -pw, 2 * pw, -2 * pw):
        if 0 <= x + dx and x + dx + pw <= hw:
            grain_fill = high_pass(host[y : y + ph, x + dx : x + dx + pw])
            break
    filled = np.where(sup > 0, space.from_z(z_bg) + grain_fill, region.astype(np.float64))
    out[y : y + ph, x : x + pw] = to_uint16(filled)
    return out


def profile_score(host: np.ndarray, x: int, pw: int, cy: int, prof: SourceProfile) -> float:
    """Match score (normalised cross-correlation, worst of three slices) of the source profile against
    the host strip [x, x+pw) with the defect centre on row ``cy``. -inf if the window does not fit."""
    n = len(prof.values)
    start = cy - prof.off
    if start < 0 or start + n > host.shape[0]:
        return -math.inf
    edges = np.linspace(0, pw, 4).astype(int)
    score = math.inf
    for k in range(3):
        cols = host[start : start + n, x + edges[k] : x + edges[k + 1]].astype(np.float64)
        w = np.median(cols, axis=1)
        sd = w.std()
        if sd <= 0:
            return -math.inf
        score = min(score, float(((w - w.mean()) / sd) @ prof.values / n))
    return score


def place_by_geometry(
    host: np.ndarray,
    keepout: np.ndarray,
    inside: np.ndarray | None,
    patch: DefectPatch,
    prof: SourceProfile,
    u_target: float,
    rng: np.random.Generator,
    n_columns: int = 200,
    end_margin: float = 0.05,
    min_score: float = 0.5,
    geometry_kw: dict | None = None,
) -> tuple[tuple[int, int], float, float] | None:
    """Version 3 placement: put the defect at the host row whose across-weld coordinate is ``u_target``.

    At each of ``n_columns`` random columns the host's bead geometry is measured on the strip under the
    patch (weld_geometry.bead_geometry); columns without a defined geometry are skipped. The defect
    centre goes to row centre + u_target x half_width. The spot must keep its support out of ``keepout``,
    its centroid inside ``inside`` (the host's main seam polygon) if given, and its profile match score
    must reach ``min_score``; among valid columns the best score wins.
    Returns ((x, y) top-left, score, u realised) or None.
    """
    from rtsynth.weld_geometry import bead_geometry, strip_profile

    ph, pw = patch.residual.shape
    hh, hw = host.shape
    lo, hi = int(end_margin * hw), int((1 - end_margin) * hw) - pw
    if hi <= lo or hh < ph:
        return None
    cy_crop = float(patch.points[:, 1].mean())
    cx_crop = float(patch.points[:, 0].mean())
    support = patch.weight > 0
    best = None
    for x in rng.integers(lo, hi, size=n_columns):
        x = int(x)
        g = bead_geometry(strip_profile(host, x, x + pw), **(geometry_kw or {}))
        if g is None:
            continue
        cy = g.row(u_target)
        y = round(cy - cy_crop)
        if y < 0 or y + ph > hh:
            continue
        if keepout[y : y + ph, x : x + pw][support].any():
            continue
        if (
            inside is not None
            and not inside[min(hh - 1, round(y + cy_crop)), min(hw - 1, round(x + cx_crop))]
        ):
            continue
        sc = profile_score(host, x, pw, round(cy), prof)
        if sc < min_score:
            continue
        if best is None or sc > best[1]:
            best = ((x, y), sc, g.u(y + cy_crop))
    return best
