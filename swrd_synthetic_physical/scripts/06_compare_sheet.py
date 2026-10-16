"""Comparison sheet: per class, real defects next to synthetic ones from one or more pipeline versions.

Each row of a class block is one source of images: REAL (SWRD training films), then every --synthetic
entry (name=path to a 02_make_films.py output). Crops are raw 16-bit windows around the defect, stretched
per crop (p1-p99), optional label outline. Same crop rule for every column, so grain, contrast and the
defect's place on the bead can be compared by eye.

    python 06_compare_sheet.py --raw ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work \
        --grey-steps ../results/grey_steps.json \
        --synthetic "run C (v2)=~/swrd_synthetic_physical/data/raw_physical_v2" \
        --synthetic "v3=~/swrd_synthetic_physical/data/raw_physical_v3" --n 4 --out compare_sheet.png
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "swrd_paper_baseline" / "scripts"))
from _common import CLASS_NAMES, LABEL_TO_CLASS, load_polygons, read_tif

CLASSES = ["inclusion", "crack", "undercut", "lack_of_fusion"]
TITLE = {
    "inclusion": "Slag inclusion",
    "crack": "Crack",
    "undercut": "Undercut",
    "lack_of_fusion": "Lack of fusion",
}
TUNGSTEN = "夹钨"


def exposure_of(stem: str) -> str:
    return stem[2:] if stem[:2] in ("A_", "B_") else stem


def crop(img: np.ndarray, pts: np.ndarray, cw: int, ch: int, outline: bool) -> np.ndarray:
    h, w = img.shape
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    ww = max(cw, (x1 - x0) + 120)
    hh = ww * ch / cw
    if hh < (y1 - y0) + 80:
        hh = (y1 - y0) + 80
        ww = hh * cw / ch
    ww, hh = int(min(ww, w)), int(min(hh, h))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a = int(np.clip(cx - ww / 2, 0, w - ww))
    b = int(np.clip(cy - hh / 2, 0, h - hh))
    c = img[b : b + hh, a : a + ww].astype(np.float64)
    lo, hi = np.percentile(c, [1, 99])
    c8 = np.clip((c - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    c8 = cv2.cvtColor(c8, cv2.COLOR_GRAY2BGR)
    if outline:
        cv2.polylines(
            c8, [np.round(pts - [a, b]).astype(np.int32)], True, (40, 60, 255), 1, cv2.LINE_AA
        )
    return cv2.resize(c8, (cw, ch), interpolation=cv2.INTER_AREA)


def label_bar(text: str, width: int, height: int = 30, size: float = 0.62) -> np.ndarray:
    bar = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(
        bar, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, size, (30, 30, 30), 1, cv2.LINE_AA
    )
    return bar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--grey-steps", type=Path, default=HERE.parent / "results" / "grey_steps.json")
    ap.add_argument("--synthetic", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cell", type=int, nargs=2, default=[340, 210])
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cw, ch = a.cell
    rng = random.Random(a.seed)
    steps = json.loads(a.grey_steps.read_text()) if a.grey_steps.is_file() else {}

    split = json.loads((a.work_dir / "split_films.json").read_text())
    val_exp = {exposure_of(t.split("__")[0]) for t in split["val"]}
    inv = json.loads((a.work_dir / "inventory.json").read_text())["records"]
    films = [
        r
        for r in inv
        if r["dtype"] == "uint16"
        and (not steps or steps.get(r["stem"]) == 1)
        and r["exposure"] not in val_exp
        and r["width"] > r["height"]
        and (a.raw / r["image"]).is_file()
    ]
    rng.shuffle(films)
    real = {c: [] for c in CLASSES}
    for r in films:
        if all(len(v) >= a.n for v in real.values()):
            break
        ci = r.get("class_instances") or []
        want = [c for c in CLASSES if len(real[c]) < a.n and ci[CLASS_NAMES.index(c)]]
        if not want:
            continue
        polys, _ = load_polygons(a.raw / r["json"])
        for c in want:
            cand = [
                p
                for p in polys
                if LABEL_TO_CLASS.get(p.label) == CLASS_NAMES.index(c)
                and p.label != TUNGSTEN
                and p.area >= 80
                and len(p.points) >= 3
            ]
            if cand:
                real[c].append((a.raw / r["image"], rng.choice(cand).points.astype(np.float64)))

    synth = []
    for spec in a.synthetic:
        name, path = spec.split("=", 1)
        path = Path(path).expanduser()
        rows = [
            json.loads(x)
            for x in (path / "inserted.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        pick = {}
        for c in CLASSES:
            cand = [r for r in rows if r["class"] == c]
            pick[c] = rng.sample(cand, min(a.n, len(cand)))
        synth.append((name, path, pick))

    blocks = []
    width = a.n * cw + (a.n - 1) * 6
    gap = np.full((ch, 6, 3), 255, np.uint8)

    def row_of(cells: list[np.ndarray]) -> np.ndarray:
        while len(cells) < a.n:
            cells.append(np.full((ch, cw, 3), 235, np.uint8))
        parts = [p for x in cells for p in (x, gap)]
        return np.hstack(parts[:-1])

    for c in CLASSES:
        blocks.append(label_bar(f"{TITLE[c]}: REAL (SWRD training films)", width, 34, 0.7))
        blocks.append(row_of([crop(read_tif(p), pts, cw, ch, a.outline) for p, pts in real[c]]))
        for name, path, pick in synth:
            cells = []
            for r in pick[c]:
                img = read_tif(path / "crop_weld_images" / "S" / "1" / f"{r['film']}.tif")
                cells.append(crop(img, np.asarray(r["points"], np.float64), cw, ch, a.outline))
            tag = ""
            if pick[c] and pick[c][0].get("aug"):
                tag = "  (varied: size, stretch, angle, depth, mirror)"
            blocks.append(label_bar(f"{TITLE[c]}: SYNTHETIC {name}{tag}", width, 34, 0.7))
            blocks.append(row_of(cells))
        blocks.append(np.full((22, width, 3), 255, np.uint8))
    out = np.vstack(blocks)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), out)
    print(f"[sheet] {a.out} {out.shape}")


if __name__ == "__main__":
    main()
