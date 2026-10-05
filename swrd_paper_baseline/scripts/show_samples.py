"""Visual check of a rendered tile set: tiles per class, and one film with its tiles mapped back.

Figure 1  samples_per_class.png     N rendered tiles per class with their YOLO boxes drawn
Figure 2  film_<stem>.png           top: the raw film (display stretch) with every tile of the paper's grid
                                    outlined and colour-coded by split (train / val / not selected); the
                                    tiles shown in the strip below are marked with thick coloured frames
                                    middle: the film stitched back from per-tile preprocessed tiles (shows
                                    what the model sees, incl. per-tile CLAHE seams)
                                    bottom: the marked tiles as the model sees them, with boxes

Run:
  uv run python scripts/show_samples.py --raw-dir ... --work-dir ... --yolo-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid --out-dir ~/swrd_paper_baseline/logs/figs
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import CLASS_NAMES, contrast_stretch_to_uint8, load_polygons, preprocess_tile, read_tif
from matplotlib.patches import Polygon as MplPoly
from matplotlib.patches import Rectangle

CLASS_COLORS = ["#ffd400", "#ff7f0e", "#e41a1c", "#4daf4a", "#377eb8", "#984ea3"]
SPLIT_COLORS = {"train": "#2ca02c", "val": "#1f77b4", None: "#888888"}


def read_yolo(label_path: Path, side: int) -> list[tuple[int, int, int, int, int]]:
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text().split("\n"):
        if not line.strip():
            continue
        c, cx, cy, w, h = line.split()
        c, cx, cy, w, h = (
            int(c),
            float(cx) * side,
            float(cy) * side,
            float(w) * side,
            float(h) * side,
        )
        boxes.append((c, int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)))
    return boxes


def draw_tile(ax, img: np.ndarray, boxes, title: str, frame: str | None = None) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for c, x0, y0, x1, y1 in boxes:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=CLASS_COLORS[c], lw=1.6))
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    if frame is not None:
        for s in ax.spines.values():
            s.set_edgecolor(frame)
            s.set_linewidth(4)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--yolo-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument(
        "--film", default=None, help="stem of the film for figure 2 (default: most classes)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    split = json.loads((args.work_dir / "split_tiles.json").read_text())
    part_of = {i: "train" for i in split["train"]} | {i: "val" for i in split["val"]}
    tiles_by_film: dict[str, list[dict]] = defaultdict(list)
    by_class: dict[int, list[dict]] = defaultdict(list)
    with open(args.work_dir / "tiles.jsonl") as f:
        for line in f:
            t = json.loads(line)
            t["part"] = part_of.get(t["tile_id"])
            tiles_by_film[t["stem"]].append(t)
            if t["part"] and t["boxes"]:
                classes = {int(b.split()[0]) for b in t["boxes"]}
                for c in classes:
                    by_class[c].append(t)

    # ---------------- figure 1: tiles per class ----------------
    n = args.per_class
    fig, axes = plt.subplots(len(CLASS_NAMES), n, figsize=(2.6 * n, 2.9 * len(CLASS_NAMES)))
    for c, name in enumerate(CLASS_NAMES):
        pool = by_class[c]

        # random tiles containing this class, skipping the thinnest slivers (class box < 12 px either way)
        def ok(t, c=c):
            for b in t["boxes"]:
                cc, _, _, w, h = b.split()
                if int(cc) == c and float(w) * t["side"] >= 12 and float(h) * t["side"] >= 12:
                    return True
            return False

        eligible = [t for t in pool if ok(t)] or pool
        picks = rng.sample(eligible, min(n, len(eligible)))
        for j in range(n):
            ax = axes[c, j]
            if j >= len(picks):
                ax.axis("off")
                continue
            t = picks[j]
            img = cv2.imread(
                str(args.yolo_dir / "images" / t["part"] / f"{t['tile_id']}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            boxes = read_yolo(
                args.yolo_dir / "labels" / t["part"] / f"{t['tile_id']}.txt", t["side"]
            )
            draw_tile(
                ax, img, boxes, f"{t['stem']}\nx{t['x']} y{t['y']} s{t['side']} [{t['part']}]"
            )
        axes[c, 0].set_ylabel(name, fontsize=12, fontweight="bold", color=CLASS_COLORS[c])
    fig.suptitle(
        f"{args.yolo_dir.name}: {n} rendered tiles per class (boxes = YOLO labels; box colour = class)",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(args.out_dir / "samples_per_class.png", dpi=110)
    plt.close(fig)

    # ---------------- figure 2: one film ----------------
    if args.film:
        stem = args.film
    else:

        def n_classes(ts):
            return len({int(b.split()[0]) for t in ts for b in t["boxes"]})

        candidates = [
            s for s, ts in tiles_by_film.items() if 60 <= len(ts) <= 140 and ts[0]["side"] >= 300
        ]
        stem = max(
            candidates,
            key=lambda s: (
                n_classes(tiles_by_film[s]),
                sum(bool(t["boxes"]) for t in tiles_by_film[s]),
            ),
        )
    ts = tiles_by_film[stem]
    rec_img = ts[0]["image"]
    img16 = read_tif(args.raw_dir / rec_img)
    H, W = img16.shape
    side = ts[0]["side"]
    disp = contrast_stretch_to_uint8(img16, 0.5, 99.5)
    rp = json.loads((args.yolo_dir / "render_params.json").read_text())

    # stitched canvas from per-tile preprocessing of EVERY grid tile (what the model would see at inference)
    canvas = np.zeros((H, W), dtype=np.uint8)
    for t in sorted(ts, key=lambda t: (t["y"], t["x"])):
        tile = preprocess_tile(
            img16[t["y"] : t["y"] + side, t["x"] : t["x"] + side],
            rp["p_lo"],
            rp["p_hi"],
            rp["clahe_clip"],
            rp["clahe_grid"],
        )
        canvas[t["y"] : t["y"] + side, t["x"] : t["x"] + side] = tile

    # choose tiles to show: up to 8, covering as many classes as possible, dataset members only
    chosen, seen = [], set()
    for t in sorted([t for t in ts if t["part"] and t["boxes"]], key=lambda t: -len(t["boxes"])):
        cls = {int(b.split()[0]) for b in t["boxes"]}
        if cls - seen or len(chosen) < 4:
            chosen.append(t)
            seen |= cls
        if len(chosen) == 8:
            break
    negs = [t for t in ts if t["part"] and not t["boxes"]]
    if negs and len(chosen) < 8:
        chosen.append(rng.choice(negs))
    frame_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    json_rel = Path(rec_img.replace("crop_weld_images", "crop_weld_jsons")).with_suffix(".json")
    polys = [p for p in load_polygons(args.raw_dir / json_rel)[0] if p.class_id is not None]

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, len(chosen), height_ratios=[1.6, 1.6, 1.3], hspace=0.25, wspace=0.08)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(disp, cmap="gray", vmin=0, vmax=255)
    for t in ts:
        ax1.add_patch(
            Rectangle(
                (t["x"], t["y"]),
                side,
                side,
                fill=False,
                ec=SPLIT_COLORS[t["part"]],
                lw=0.7,
                alpha=0.9,
                ls="-" if t["part"] else ":",
            )
        )
    for p in polys:
        ax1.add_patch(
            MplPoly(p.points, closed=True, fill=False, ec=CLASS_COLORS[p.class_id], lw=1.5)
        )
    for k, t in enumerate(chosen):
        ax1.add_patch(Rectangle((t["x"], t["y"]), side, side, fill=False, ec=frame_colors[k], lw=3))
        ax1.text(
            t["x"] + 8,
            t["y"] + 30,
            str(k + 1),
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox={"fc": frame_colors[k], "lw": 0, "alpha": 0.9},
        )
    n_tr = sum(t["part"] == "train" for t in ts)
    n_va = sum(t["part"] == "val" for t in ts)
    n_no = sum(t["part"] is None for t in ts)
    ax1.set_title(
        f"{stem}  {W}x{H} px, raw film (display stretch). Grid: {len(ts)} tiles of {side} px — green solid = train ({n_tr}), blue solid = val ({n_va}), grey dotted = not selected ({n_no}). Polygons = annotations; numbered frames = tiles below.",
        fontsize=11,
        loc="left",
    )
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[1, :])
    ax2.imshow(canvas, cmap="gray", vmin=0, vmax=255)
    ax2.set_title(
        f"same film stitched from per-tile preprocessed tiles (stretch p{rp['p_lo']}–p{rp['p_hi']} → 8-bit → CLAHE clip {rp['clahe_clip']} grid {rp['clahe_grid']}); black = area no tile covers",
        fontsize=11,
        loc="left",
    )
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    ax2.set_xticks([])
    ax2.set_yticks([])

    for k, t in enumerate(chosen):
        ax = fig.add_subplot(gs[2, k])
        img = cv2.imread(
            str(args.yolo_dir / "images" / t["part"] / f"{t['tile_id']}.png"), cv2.IMREAD_GRAYSCALE
        )
        boxes = read_yolo(args.yolo_dir / "labels" / t["part"] / f"{t['tile_id']}.txt", side)
        names = sorted({CLASS_NAMES[b[0]] for b in boxes}) or ["(negative)"]
        draw_tile(
            ax,
            img,
            boxes,
            f"{k + 1}: x{t['x']} y{t['y']} [{t['part']}]\n" + ", ".join(names),
            frame=frame_colors[k],
        )
    fig.savefig(args.out_dir / f"film_{stem}.png", dpi=100, bbox_inches="tight")
    print("film:", stem, "| tiles:", len(ts), "| chosen:", [(t["x"], t["y"]) for t in chosen])
    print("wrote", args.out_dir / "samples_per_class.png", "and", args.out_dir / f"film_{stem}.png")


if __name__ == "__main__":
    main()
