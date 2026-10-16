"""Unit tests for the version-3 insertion pieces. Run from the repo root: uv run --with pytest pytest tests/"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtsynth.compositing import (
    AdditiveSpace,
    Augmentation,
    SourceProfile,
    augment_patch,
    extract_defect,
    grain_sigma,
    null_implant,
    place_by_geometry,
    vflip_patch,
)
from rtsynth.weld_geometry import bead_geometry, geometry_at, polygon_u

SPACE = AdditiveSpace(None)


def synthetic_film(h=700, w=3000, half=200, centre=350, level=30000, bead=20000, grain=150, seed=0):
    """Landscape film with a bright bead hump across rows centre +- half, parent metal darker."""
    rng = np.random.default_rng(seed)
    rows = np.arange(h)
    hump = np.clip(1 - ((rows - centre) / half) ** 2, 0, 1)  # parabola: 1 at centre, 0 at the toes
    prof = level + bead * hump
    img = prof[:, None] + rng.normal(0, grain, (h, w))
    return np.clip(np.rint(img), 0, 65535).astype(np.uint16)


def add_pore(img, cx, cy, r=12, depth=3000):
    out = img.astype(np.float64)
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    d = np.hypot(xx - cx, yy - cy)
    out -= depth * np.clip(1 - (d / r) ** 2, 0, 1)
    return np.clip(np.rint(out), 0, 65535).astype(np.uint16)


def circle_points(cx, cy, r, n=24):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


# ---- bead geometry --------------------------------------------------------------------------------


def test_bead_geometry_finds_centre_and_toes_of_a_hump():
    img = synthetic_film()
    g = geometry_at(img, 1000, 1100)
    assert g is not None
    # half-height of a parabola is at half / sqrt(2) from the centre
    assert abs(g.centre - 350) < 4
    assert abs(g.half_width - 200 / np.sqrt(2)) < 6
    assert abs(g.u(350)) < 0.03
    assert abs(g.u(350 + 200 / np.sqrt(2)) - 1) < 0.05


def test_bead_geometry_one_side_cut_uses_the_peak():
    img = synthetic_film(h=500, centre=120, half=200)  # top toe cut by the film border
    g = geometry_at(img, 1000, 1100)
    assert g is not None and g.top is None and g.bottom is not None
    assert abs(g.centre - 120) < 8


def test_bead_geometry_undefined_when_flat_or_cut_both_sides():
    flat = synthetic_film(bead=0)
    assert geometry_at(flat, 1000, 1100) is None
    both = synthetic_film(h=200, centre=100, half=400)
    assert geometry_at(both, 1000, 1100) is None


def test_bead_geometry_rejects_short_runs():
    prof = np.full(700, 1000.0)
    prof[300:320] = 5000
    assert bead_geometry(prof) is None


def test_polygon_u_of_a_pore_near_the_toe():
    img = synthetic_film()
    toe = 350 + 200 / np.sqrt(2)
    r = polygon_u(img, circle_points(1500, toe, 10))
    assert r is not None and abs(r[0] - 1) < 0.08


# ---- extraction with compensation ---------------------------------------------------------------


def test_compensation_restores_median_depth_of_a_thin_line():
    img = synthetic_film(grain=20)
    img = img.astype(np.float64)
    img[400:402, 1000:1300] -= 4000  # a 2 px wide crack
    img = np.clip(np.rint(img), 0, 65535).astype(np.uint16)
    pts = np.array([[1000, 400], [1300, 400], [1300, 402], [1000, 402]], np.float64)
    plain = extract_defect(img, pts, SPACE, compensate=0.0)
    comp = extract_defect(img, pts, SPACE, compensate=1.6)
    d_plain = -np.median(plain.residual[plain.label_mask > 0])
    d_comp = -np.median(comp.residual[comp.label_mask > 0])
    assert plain.compensation == 1.0 and 1.1 < comp.compensation <= 1.6
    assert d_comp > d_plain * 1.1
    assert d_comp <= 4000 * 1.05  # never deeper than the raw dip
    assert (comp.residual <= 0).all()


# ---- augmentation ---------------------------------------------------------------------------------


def _pore_patch():
    img = add_pore(synthetic_film(grain=30), 1500, 350)
    return extract_defect(img, circle_points(1500, 350, 12), SPACE)


def test_augment_scales_the_polygon_and_the_mask_together():
    patch = _pore_patch()
    aug = Augmentation(scale=1.4, aspect=1.0, angle_deg=0.0, amplitude=1.0)
    out = augment_patch(patch, aug)
    assert out.residual.shape[0] > patch.residual.shape[0] * 1.3
    assert out.label_mask.sum() > patch.label_mask.sum() * 1.4**2 * 0.85
    # polygon centroid lands on the mask centroid
    m = cv2.moments(out.label_mask)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    assert abs(out.points[:, 0].mean() - cx) < 1.5 and abs(out.points[:, 1].mean() - cy) < 1.5
    assert out.weight.min() >= 0 and out.weight.max() <= 1
    assert (out.residual <= 0).all()


def test_augment_amplitude_and_rotation():
    patch = _pore_patch()
    out = augment_patch(patch, Augmentation(amplitude=1.3, angle_deg=90.0))
    assert abs(out.residual.min() / patch.residual.min() - 1.3) < 0.1
    assert out.residual.shape == patch.residual.shape[::-1]


def test_augment_stretch_changes_aspect_only():
    patch = _pore_patch()
    out = augment_patch(patch, Augmentation(aspect=1.25))
    assert out.residual.shape[1] > out.residual.shape[0]
    assert abs(out.residual.sum() / patch.residual.sum() - 1) < 0.15  # area-preserving stretch


def test_vflip_mirrors_rows_and_polygon():
    patch = _pore_patch()
    out = vflip_patch(patch)
    assert np.array_equal(out.residual, patch.residual[::-1])
    assert (
        abs((out.points[:, 1] + patch.points[:, 1]).mean() - (patch.residual.shape[0] - 1)) < 1e-9
    )


def test_augment_sample_respects_angle_rule():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert abs(Augmentation.sample(rng, elongated=True).angle_deg) <= 5
    angles = [Augmentation.sample(rng, elongated=False).angle_deg for _ in range(50)]
    assert max(angles) > 90


# ---- null implant ---------------------------------------------------------------------------------


def test_null_implant_changes_only_the_support_and_keeps_grain():
    host = synthetic_film(grain=150, seed=3)
    sup = np.zeros((60, 80), bool)
    sup[20:40, 20:60] = True
    out = null_implant(host, sup, (1000, 300), SPACE)
    diff = out.astype(int) != host.astype(int)
    assert diff[300:360, 1000:1080][~sup].sum() == 0
    assert diff.sum() == diff[300:360, 1000:1080][sup].sum()
    assert diff[300:360, 1000:1080][sup].mean() > 0.9
    g_in = grain_sigma(out[320:340, 1020:1060])
    g_out = grain_sigma(host[320:340, 1120:1160])
    assert 0.6 < g_in / g_out < 1.5
    assert abs(float(out[320:340, 1020:1060].mean()) - float(host[320:340, 1020:1060].mean())) < 400


# ---- placement by geometry ------------------------------------------------------------------------


def test_place_by_geometry_lands_on_the_requested_u():
    src = add_pore(synthetic_film(grain=30, seed=1), 1500, 350 + 120)
    patch = extract_defect(src, circle_points(1500, 470, 12), SPACE)
    # source profile on the cleaned source (the pore removed), window of 300 rows
    cleaned = src.copy()
    x0, y0 = patch.origin
    ph, pw = patch.residual.shape
    cleaned[y0 : y0 + ph, x0 : x0 + pw] = patch.cleaned_crop
    cy_src = y0 + patch.points[:, 1].mean()
    a, b = int(cy_src - 150), int(cy_src + 150)
    prof = np.median(cleaned[a:b, x0 : x0 + pw].astype(float), axis=1)
    prof = SourceProfile(
        (prof - prof.mean()) / prof.std(), int(cy_src) - a, round(patch.points[:, 1].mean())
    )
    host = synthetic_film(h=800, centre=420, half=260, grain=100, seed=2)
    keep = np.zeros(host.shape, bool)
    keep[:, :1200] = True
    res = place_by_geometry(
        host, keep, None, patch, prof, u_target=0.8, rng=np.random.default_rng(0)
    )
    assert res is not None
    (x, y), score, u = res
    assert x >= 1200 and score > 0.5
    assert abs(u - 0.8) < 0.05
    g = geometry_at(host, x, x + pw)
    assert abs(g.u(y + patch.points[:, 1].mean()) - 0.8) < 0.05


def test_place_by_geometry_returns_none_without_geometry():
    patch = _pore_patch()
    prof = SourceProfile(np.zeros(300), 150, 10)
    host = synthetic_film(bead=0)
    assert (
        place_by_geometry(
            host, np.zeros(host.shape, bool), None, patch, prof, 0.5, np.random.default_rng(0)
        )
        is None
    )
