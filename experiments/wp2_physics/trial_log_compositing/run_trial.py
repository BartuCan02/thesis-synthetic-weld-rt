"""Trial: insert one real SWRD defect per class into a defect-free SWRD film, physical vs naive.

Inputs: the films chosen by select_films.py (copied to data/films, git-ignored).
Outputs:
    figures/<class>.png     detail per class: source, removal, extracted defect, host, both arms, label
    figures/overview.png    all classes side by side: host original | physical | naive
    results/trial_run.json  placements, scores, parameters
    data/outputs/           16-bit composites + LabelMe JSON of the moved polygon (git-ignored)

Run from the repo root:
    uv run python experiments/wp2_physics/trial_log_compositing/run_trial.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from rtsynth.compositing import (
    AdditiveSpace,
    composite_naive,
    composite_physical,
    contrast_scale,
    extract_defect,
    place_by_profile,
)

SEAM = "焊缝"
CLASS_TITLE = {
    "porosity": "Porosity",
    "inclusion": "Slag inclusion",
    "crack": "Crack",
    "undercut": "Undercut",
    "lack_of_fusion": "Lack of fusion",
    "lack_of_penetration": "Lack of penetration",
}


def load_labelme(path: Path) -> list[tuple[str, np.ndarray]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return [(s["label"], np.asarray(s["points"], np.float64)) for s in d["shapes"] if s["points"]]


def host_keepout(
    shape: tuple[int, int], shapes: list[tuple[str, np.ndarray]], grow: int
) -> np.ndarray:
    """Everything labelled on the host except its main seam polygon, grown by `grow` px."""
    seams = [p for lab, p in shapes if lab == SEAM]
    main = max(seams, key=lambda p: cv2.contourArea(p.astype(np.float32))) if seams else None
    m = np.zeros(shape, np.uint8)
    for lab, p in shapes:
        if lab == SEAM and p is main:
            continue
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
        else:
            cv2.polylines(m, [np.round(p).astype(np.int32)], False, 1, 3)
    return cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0


def window(center: tuple[float, float], size: tuple[int, int], shape: tuple[int, int]):
    (cx, cy), (ww, wh), (h, w) = center, size, shape
    ww, wh = min(ww, w), min(wh, h)
    x0 = int(np.clip(round(cx - ww / 2), 0, w - ww))
    y0 = int(np.clip(round(cy - wh / 2), 0, h - wh))
    return x0, y0, x0 + ww, y0 + wh


def stretch(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((img.astype(np.float64) - lo) / max(hi - lo, 1.0), 0, 1)


def show(ax, img, title, lo, hi):
    ax.imshow(stretch(img, lo, hi), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--films", type=Path, default=HERE / "data" / "films")
    ap.add_argument("--space", choices=["linear", "log"], default="linear")
    ap.add_argument("--offset", type=float, default=0.0, help="B for --space log")
    ap.add_argument("--dilate", type=int, default=6)
    ap.add_argument("--denoise-sigma", type=float, default=1.0)
    ap.add_argument(
        "--context", type=int, default=90, help="px of context around the patch in zooms"
    )
    ap.add_argument(
        "--no-contrast-scale",
        action="store_true",
        help="add the residual unscaled (default: scale by host grain / source grain)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    space = AdditiveSpace(None if args.space == "linear" else args.offset)
    rng = np.random.default_rng(args.seed)
    manifest = json.loads((args.films / "manifest.json").read_text(encoding="utf-8"))
    hosts = manifest["hosts"]
    fig_dir, res_dir = HERE / "figures", HERE / "results"
    out_dir = HERE / "data" / "outputs"
    panel_dir = fig_dir / "panels"
    panels_meta = {}
    for d in (fig_dir, res_dir, out_dir, panel_dir):
        d.mkdir(parents=True, exist_ok=True)

    run = {"space": space.name, "params": {k: str(v) for k, v in vars(args).items()}, "items": []}
    overview = []
    for i, (cls, src) in enumerate(manifest["sources"].items()):
        host_d = hosts[i % len(hosts)]
        source = tifffile.imread(args.films / src["tif"])
        host = tifffile.imread(args.films / host_d["tif"])
        pts = np.asarray(src["points"], np.float64)

        patch = extract_defect(
            source, pts, space, dilate_px=args.dilate, denoise_sigma=args.denoise_sigma
        )
        ph, pw = patch.residual.shape
        sx, sy = patch.origin
        cleaned_full = source.copy()
        cleaned_full[sy : sy + ph, sx : sx + pw] = patch.cleaned_crop
        keep = host_keepout(host.shape, load_labelme(args.films / host_d["json"]), grow=15)
        (x, y), score = place_by_profile(host, keep, cleaned_full, patch, rng)
        scale = 1.0 if args.no_contrast_scale else contrast_scale(host, patch, (x, y))
        phys = composite_physical(host, patch, (x, y), space, scale)
        naive = composite_naive(host, patch, (x, y))
        new_pts = patch.points + [x, y]

        # 16-bit outputs + LabelMe of the moved polygon (training-ready format, git-ignored)
        tifffile.imwrite(out_dir / f"{cls}_physical.tif", phys)
        tifffile.imwrite(out_dir / f"{cls}_naive.tif", naive)
        (out_dir / f"{cls}_physical.json").write_text(
            json.dumps(
                {
                    "version": "5.0.1",
                    "flags": {},
                    "imagePath": f"{cls}_physical.tif",
                    "imageHeight": int(host.shape[0]),
                    "imageWidth": int(host.shape[1]),
                    "imageData": None,
                    "shapes": [
                        {
                            "label": src["label"],
                            "points": new_pts.tolist(),
                            "shape_type": "polygon",
                            "group_id": None,
                            "flags": {"synthetic": True},
                        }
                    ],
                },
                ensure_ascii=False,
                indent=1,
            )
        )

        wsize = (max(pw + 2 * args.context, 360), max(ph + 2 * args.context, 260))
        hx0, hy0, hx1, hy1 = window((x + pw / 2, y + ph / 2), wsize, host.shape)
        sx0, sy0, sx1, sy1 = window((sx + pw / 2, sy + ph / 2), wsize, source.shape)
        h_win = host[hy0:hy1, hx0:hx1]
        lo, hi = np.percentile(h_win, [0.5, 99.5])
        s_win = source[sy0:sy1, sx0:sx1]
        slo, shi = np.percentile(s_win, [0.5, 99.5])
        dip_src = float(-patch.residual[patch.label_mask > 0].mean())
        item = {
            "class": cls,
            "source": src["stem"],
            "host": host_d["stem"],
            "polygon_area_px": round(src["area"]),
            "patch_hw": [ph, pw],
            "host_top_left": [x, y],
            "profile_match": round(score, 3),
            "source_grain": round(patch.grain, 1),
            "contrast_scale": round(scale, 3),
            "mean_dip_in_polygon": round(dip_src, 4 if space.offset is not None else 1),
        }
        run["items"].append(item)
        print(json.dumps(item, ensure_ascii=False))

        # ---- detail figure ------------------------------------------------------------------------
        aspect = (hy1 - hy0) / (hx1 - hx0)
        col_w = 4.3
        fig = plt.figure(figsize=(4 * col_w, 2 * col_w * aspect + 2.6 + 1.0), dpi=130)
        gs = fig.add_gridspec(3, 4, height_ratios=[col_w * aspect, col_w * aspect, 2.0])
        title = CLASS_TITLE[cls]
        ax = fig.add_subplot(gs[0, 0])
        show(ax, s_win, f"Real {title.lower()} on its source film", slo, shi)
        ax.plot(*np.vstack([pts, pts[:1]]).T - [[sx0], [sy0]], color="#ff3b30", lw=0.9)
        ax = fig.add_subplot(gs[0, 1])
        show(
            ax, cleaned_full[sy0:sy1, sx0:sx1], "Defect removed (inpainted + real grain)", slo, shi
        )
        ax = fig.add_subplot(gs[0, 2])
        res_win = np.zeros_like(s_win, dtype=np.float64)
        ry, rx = sy - sy0, sx - sx0
        res_win[ry : ry + ph, rx : rx + pw] = patch.residual
        vmax = float(np.abs(patch.residual).max()) or 1.0
        im = ax.imshow(res_win, cmap="RdBu", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title("Extracted defect = original - removed", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("grey levels" if space.offset is None else "log units", fontsize=7)
        ax = fig.add_subplot(gs[0, 3])
        ax.axis("off")
        ax.text(
            0.0,
            0.95,
            "\n".join(
                [
                    f"Class: {title}",
                    f"Source film: {src['stem']}",
                    f"Host film: {host_d['stem']}",
                    f"Polygon area: {round(src['area'])} px",
                    f"Additive space: {space.name}",
                    f"Placement profile match: {score:.2f}",
                    f"Contrast scale (grain ratio): {scale:.2f}",
                    "",
                    "Physical: host + scale x extracted defect",
                    "Naive: alpha blend of the raw",
                    "source crop (A/B control arm)",
                ]
            ),
            va="top",
            fontsize=9,
            family="monospace",
        )

        show(fig.add_subplot(gs[1, 0]), h_win, "Host film, defect-free (original)", lo, hi)
        show(
            fig.add_subplot(gs[1, 1]),
            phys[hy0:hy1, hx0:hx1],
            f"Synthetic {title.lower()}: physical",
            lo,
            hi,
        )
        show(
            fig.add_subplot(gs[1, 2]),
            naive[hy0:hy1, hx0:hx1],
            f"Synthetic {title.lower()}: naive",
            lo,
            hi,
        )
        ax = fig.add_subplot(gs[1, 3])
        show(ax, phys[hy0:hy1, hx0:hx1], "Physical + label (exact mask)", lo, hi)
        ax.plot(*np.vstack([new_pts, new_pts[:1]]).T - [[hx0], [hy0]], color="#ff3b30", lw=0.9)

        ax = fig.add_subplot(gs[2, :])
        flo, fhi = np.percentile(phys, [0.5, 99.5])
        show(
            ax,
            phys,
            f"Whole host film with the synthetic {title.lower()} (physical); box = zoom",
            flo,
            fhi,
        )
        ax.add_patch(Rectangle((hx0, hy0), hx1 - hx0, hy1 - hy0, fill=False, ec="#ff3b30", lw=1.2))
        fig.suptitle(f"{title}: real SWRD defect moved into a defect-free SWRD film", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{cls}.png")
        plt.close(fig)

        overview.append((title, h_win, phys[hy0:hy1, hx0:hx1], naive[hy0:hy1, hx0:hx1], lo, hi))

        # ---- individual 8-bit panels for the review page (same stretch within each row) ----------
        def save(name: str, img: np.ndarray, a: float, b: float, cls: str = cls) -> None:
            cv2.imwrite(
                str(panel_dir / f"{cls}_{name}.png"),
                np.rint(stretch(img, a, b) * 255).astype(np.uint8),
            )

        save("host", h_win, lo, hi)
        save("physical", phys[hy0:hy1, hx0:hx1], lo, hi)
        save("naive", naive[hy0:hy1, hx0:hx1], lo, hi)
        save("source", s_win, slo, shi)
        save("removed", cleaned_full[sy0:sy1, sx0:sx1], slo, shi)
        panels_meta[cls] = {
            **item,
            "title": title,
            "host_size": [hx1 - hx0, hy1 - hy0],
            "source_size": [sx1 - sx0, sy1 - sy0],
            "host_polygon": (new_pts - [hx0, hy0]).round(1).tolist(),
            "source_polygon": (pts - [sx0, sy0]).round(1).tolist(),
        }

    # ---- overview sheet ---------------------------------------------------------------------------
    rows = len(overview)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 2.6 * rows), dpi=120)
    for r, (title, a, b, c, lo, hi) in enumerate(overview):
        show(axes[r, 0], a, f"{title}: host original", lo, hi)
        show(axes[r, 1], b, f"{title}: physical", lo, hi)
        show(axes[r, 2], c, f"{title}: naive", lo, hi)
    fig.suptitle("One real defect per class inserted into defect-free SWRD films", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "overview.png")
    plt.close(fig)
    (res_dir / "trial_run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    (panel_dir / "panels.json").write_text(json.dumps(panels_meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
