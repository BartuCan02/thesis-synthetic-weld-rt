"""Integrity and sanity report for a rendered tile set (gate before upload / training).

Checks every tile image and label file:
  files      image <-> label pairing, PNG decodes, square, 8-bit, side equals the tile index
  pixels     mean / std per tile, flat tiles (std < 2), heavily saturated tiles (> 30 % at 0 or 255)
  labels     class ids in range, coordinates inside [0, 1], positive width/height, boxes per tile
  cut boxes  share of boxes touching a tile edge (a defect cut by the window), per class
  negatives  empty label files, split balance vs Table 3
Writes <out>/check_report.md and prints it.

Run:
  uv run python scripts/check_dataset.py --yolo-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid --work-dir ~/swrd_paper_baseline/data/work
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from _common import CLASS_NAMES, TABLE3_SPLIT
from tqdm import tqdm


def check_one(args: tuple) -> dict:
    img_path, lab_path, side_expected = args
    out = {"ok": True, "problems": []}
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"ok": False, "problems": ["unreadable image"]}
    if img.ndim != 2:
        out["problems"].append(f"not single-channel: {img.shape}")
    if img.dtype != np.uint8:
        out["problems"].append(f"dtype {img.dtype}")
    h, w = img.shape[:2]
    if h != w:
        out["problems"].append(f"not square {w}x{h}")
    if side_expected is not None and h != side_expected:
        out["problems"].append(f"side {h} != index {side_expected}")
    out["side"] = h
    out["mean"] = float(img.mean())
    out["std"] = float(img.std())
    out["sat"] = float(((img == 0) | (img == 255)).mean())
    boxes = []
    if lab_path.is_file():
        for line in lab_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                out["problems"].append("bad label line")
                continue
            c = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            if not 0 <= c < len(CLASS_NAMES):
                out["problems"].append(f"class {c}")
            if bw <= 0 or bh <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1):
                out["problems"].append("box outside [0,1] or empty")
            x0, y0, x1, y1 = (
                (cx - bw / 2) * h,
                (cy - bh / 2) * h,
                (cx + bw / 2) * h,
                (cy + bh / 2) * h,
            )
            touches = x0 <= 1 or y0 <= 1 or x1 >= h - 1 or y1 >= h - 1
            boxes.append((c, bw * h, bh * h, touches))
    else:
        out["problems"].append("missing label file")
    out["boxes"] = boxes
    out["ok"] = not out["problems"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--yolo-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    side_of = {}
    with open(args.work_dir / "tiles.jsonl") as f:
        for line in f:
            t = json.loads(line)
            side_of[t["tile_id"]] = t["side"]

    lines = [f"# Dataset check — {args.yolo_dir}\n"]
    for part in ("train", "val"):
        imgs = sorted((args.yolo_dir / "images" / part).glob("*.png"))
        labs = {p.stem for p in (args.yolo_dir / "labels" / part).glob("*.txt")}
        unpaired_imgs = [p.stem for p in imgs if p.stem not in labs]
        unpaired_labs = labs - {p.stem for p in imgs}
        jobs = [
            (p, args.yolo_dir / "labels" / part / f"{p.stem}.txt", side_of.get(p.stem))
            for p in imgs
        ]
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(
                tqdm(ex.map(check_one, jobs, chunksize=64), total=len(jobs), unit="tile", desc=part)
            )
        problems = Counter(pr for r in res for pr in r["problems"])
        sides = np.array([r.get("side", 0) for r in res])
        means = np.array([r.get("mean", 0) for r in res])
        stds = np.array([r.get("std", 0) for r in res])
        sats = np.array([r.get("sat", 0) for r in res])
        n_boxes = np.array([len(r.get("boxes", [])) for r in res])
        n_pos = int((n_boxes > 0).sum())
        n_neg = len(res) - n_pos
        per_class = Counter()
        cut_per_class = Counter()
        box_w, box_h = [], []
        for r in res:
            for c, bw, bh, touches in r.get("boxes", []):
                per_class[c] += 1
                cut_per_class[c] += touches
                box_w.append(bw)
                box_h.append(bh)
        box_w, box_h = np.array(box_w), np.array(box_h)
        paper_pos = TABLE3_SPLIT[f"{part}_defect"]
        paper_neg = TABLE3_SPLIT[f"{part}_background"]
        lines += [
            f"## {part}\n",
            f"- tiles: **{len(res):,}** = {n_pos:,} with boxes (paper {paper_pos:,}) + {n_neg:,} negatives (paper {paper_neg:,})",
            f"- unpaired: {len(unpaired_imgs)} images without label, {len(unpaired_labs)} labels without image",
            f"- problems: {dict(problems) if problems else 'none'}",
            (
                f"- tile side px: min {sides.min()}, median {int(np.median(sides))}, max {sides.max()}; "
                f"< 160 px: {(sides < 160).sum():,} ({(sides < 160).mean():.1%}), > 640 px: {(sides > 640).sum():,} ({(sides > 640).mean():.1%})"
            ),
            (
                f"- pixel mean over tiles: {means.mean():.1f} ± {means.std():.1f}; tile std median {np.median(stds):.1f}; "
                f"flat tiles (std < 2): {(stds < 2).sum()}; saturated > 30 %: {(sats > 0.30).sum()} ({(sats > 0.30).mean():.2%})"
            ),
            f"- boxes: {int(n_boxes.sum()):,} total; per positive tile median {int(np.median(n_boxes[n_boxes > 0]))}, max {n_boxes.max()}",
            (
                f"- box size px: width median {np.median(box_w):.0f} (p5 {np.percentile(box_w, 5):.0f}, p95 {np.percentile(box_w, 95):.0f}), "
                f"height median {np.median(box_h):.0f} (p5 {np.percentile(box_h, 5):.0f}, p95 {np.percentile(box_h, 95):.0f}); "
                f"boxes < 8 px on a side: {((box_w < 8) | (box_h < 8)).sum():,} ({((box_w < 8) | (box_h < 8)).mean():.1%})"
            ),
            "",
            "| class | boxes | touching a tile edge (cut) |\n|---|---:|---:|",
        ]
        for c, name in enumerate(CLASS_NAMES):
            n = per_class[c]
            lines.append(
                f"| {name} | {n:,} | {cut_per_class[c]:,} ({cut_per_class[c] / n if n else 0:.0%}) |"
            )
        lines.append("")
    report = "\n".join(lines)
    (args.yolo_dir / "check_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
