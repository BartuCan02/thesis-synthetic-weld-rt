"""Pore-dip test: in which pixel space do SWRD defects add?

Question. A pore removes a little steel. If the raw16 pixel is already linear in log-exposure (film
density read by a density-linear scanner), the pore's dip in grey value does not depend on how bright
the surrounding film is. If the pixel is linear in light through the film, the dip grows in proportion
to (background - B), where B is the scanner offset, and the right space is log(I - B), as in
Mery & Katsaggelos 2017, eq. 13.

Test. For every pore: dip = median(ring around the pore) - mean(inside the polygon). Regress
log(dip) on log(background) and log(area) *within* each film (film fixed effects), so film-level
exposure, film type and scanner settings cancel. The slope on log(background) is the elasticity beta.

    beta ~ 0  -> pixels already additive (linear space)
    beta ~ 1  -> log space with B ~ 0
    other     -> log(I - B) with B = bg * (1 - 1/beta) at the typical background bg

Known confounds (stated, not removed): pore depth may correlate with position across the seam, and
scatter lowers relative contrast in thick regions. Treat the result as a first estimate.

Runs where the raw release is (the data box):
    python pore_dip_test.py --raw ~/swrd_paper_baseline/data/raw \
        --common-dir ~/thesis/swrd_paper_baseline/scripts --out pore_dip
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

POROSITY = "气孔"
SEAM = "焊缝"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="raw release root (crop_weld_*)")
    ap.add_argument(
        "--common-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "swrd_paper_baseline" / "scripts",
    )
    ap.add_argument("--out", type=Path, default=Path("pore_dip"))
    ap.add_argument("--max-films", type=int, default=600)
    ap.add_argument("--min-pores", type=int, default=10)
    ap.add_argument("--min-area", type=float, default=30.0)
    ap.add_argument("--ring-inner", type=int, default=3, help="ring starts this many px outside")
    ap.add_argument("--ring-outer", type=int, default=9, help="ring ends this many px outside")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(args.common_dir))
    from _common import load_polygons, read_tif

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    jsons = sorted((args.raw / "crop_weld_jsons").glob("*/*/*.json"))
    tif_by_stem = {p.stem: p for p in (args.raw / "crop_weld_images").glob("*/*/*.tif")}
    rng.shuffle(jsons)

    rows = []  # (film_idx, log_dip, log_bg, log_area, bg, dip)
    films_used = []
    for jp in jsons:
        if len(films_used) >= args.max_films:
            break
        tif = tif_by_stem.get(jp.stem)
        if tif is None:
            continue
        polys, _ = load_polygons(jp)
        pores = [p for p in polys if p.label == POROSITY and p.area >= args.min_area]
        if len(pores) < args.min_pores:
            continue
        img = read_tif(tif)
        if img.dtype != np.uint16:
            continue
        h, w = img.shape
        # every labelled thing except the seam is excluded from rings
        others = np.zeros((h, w), np.uint8)
        for p in polys:
            if p.label != SEAM:
                cv2.fillPoly(others, [np.round(p.points).astype(np.int32)], 1)
        others = cv2.dilate(others, np.ones((2 * args.ring_inner + 1,) * 2, np.uint8))
        k_in = np.ones((2 * args.ring_inner + 1,) * 2, np.uint8)
        k_out = np.ones((2 * args.ring_outer + 1,) * 2, np.uint8)
        fi = len(films_used)
        n_ok = 0
        for p in pores:
            x0, y0, x1, y1 = (int(v) for v in p.bbox)
            m = args.ring_outer + 2
            cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
            cx1, cy1 = min(w, x1 + m + 1), min(h, y1 + m + 1)
            crop = img[cy0:cy1, cx0:cx1].astype(np.float64)
            pm = np.zeros(crop.shape, np.uint8)
            cv2.fillPoly(pm, [np.round(p.points - [cx0, cy0]).astype(np.int32)], 1)
            if pm.sum() < 5:
                continue
            ring = (cv2.dilate(pm, k_out) > 0) & ~(cv2.dilate(pm, k_in) > 0)
            ring &= others[cy0:cy1, cx0:cx1] == 0
            if ring.sum() < 20:
                continue
            bg = float(np.median(crop[ring]))
            inner = float(crop[pm > 0].mean())
            dip = bg - inner
            if dip <= 0 or bg <= 0:
                continue
            rows.append((fi, np.log(dip), np.log(bg), np.log(p.area), bg, dip))
            n_ok += 1
        if n_ok >= args.min_pores:
            films_used.append(jp.stem)
        else:  # drop this film's rows
            rows = [r for r in rows if r[0] != fi]

    a = np.array(rows)
    film = a[:, 0].astype(int)
    y, x_bg, x_area = a[:, 1], a[:, 2], a[:, 3]

    def demean(v: np.ndarray, f: np.ndarray) -> np.ndarray:
        means = np.bincount(f, v) / np.bincount(f)
        return v - means[f]

    def fit(idx: np.ndarray) -> np.ndarray:
        f = film[idx]
        _, f = np.unique(f, return_inverse=True)
        X = np.stack([demean(x_bg[idx], f), demean(x_area[idx], f)], 1)
        return np.linalg.lstsq(X, demean(y[idx], f), rcond=None)[0]

    all_idx = np.arange(len(a))
    beta_bg, beta_area = fit(all_idx)
    nprng = np.random.default_rng(args.seed)
    films = np.unique(film)
    by_film = {f: np.where(film == f)[0] for f in films}
    boots = []
    for _ in range(args.bootstrap):
        pick = nprng.choice(films, size=len(films), replace=True)
        boots.append(fit(np.concatenate([by_film[f] for f in pick]))[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    bg_med = float(np.median(a[:, 4]))
    b_offset = bg_med * (1 - 1 / beta_bg) if beta_bg > 0.05 else float("-inf")

    result = {
        "films": len(films_used),
        "pores": len(a),
        "beta_background": float(beta_bg),
        "beta_background_ci95": [float(lo), float(hi)],
        "beta_area": float(beta_area),
        "median_background": bg_med,
        "implied_offset_B": b_offset,
        "reading": "beta~0: linear space; beta~1: log space with B~0; else log(I - B)",
        "params": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out / "pore_dip_result.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.out / "pore_dip_rows.npz", rows=a)
    print(json.dumps(result, indent=2))

    # binned plot of within-film log(dip) vs log(bg), area partialled out
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, f = np.unique(film, return_inverse=True)
    xb, ya, xa = demean(x_bg, f), demean(y, f), demean(x_area, f)
    ya = ya - beta_area * xa
    order = np.argsort(xb)
    bins = np.array_split(order, 25)
    bx = [xb[b].mean() for b in bins]
    by = [ya[b].mean() for b in bins]
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=130)
    ax.scatter(xb, ya, s=2, alpha=0.08, color="0.5", label="pores")
    ax.plot(bx, by, "o", color="C0", label="binned mean")
    xs = np.linspace(np.percentile(xb, 1), np.percentile(xb, 99), 10)
    ax.plot(xs, 0 * xs, "--", color="C2", label="beta = 0 (linear space)")
    ax.plot(xs, xs, "--", color="C3", label="beta = 1 (log space, B = 0)")
    ax.plot(
        xs, beta_bg * xs, "-", color="C0", label=f"fit beta = {beta_bg:.2f} [{lo:.2f}, {hi:.2f}]"
    )
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(np.percentile(ya, 1), np.percentile(ya, 99))
    ax.set_xlabel("log background brightness (within film)")
    ax.set_ylabel("log pore dip (within film, size removed)")
    ax.set_title(f"Pore-dip test: {len(a)} pores on {len(films_used)} SWRD films")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out / "pore_dip_test.png")


if __name__ == "__main__":
    main()
