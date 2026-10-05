"""Step 3a — choose the tiles that enter the dataset and split them (Sect. 3.2 negatives, Sect. 4.1 split).

Paper: all tiles with a defect annotation (80,648) plus an equal number of randomly chosen tiles without
any, then a 9:1 random split into train and val (Table 3). Seeds are not given (D8).

Writes <work-dir>/split_tiles.json  — the paper's protocol: random over tiles
   and <work-dir>/split_films.json  — same tiles, but split by original exposure (A_/B_ halves together),
                                      used in step 5 to measure how much the tile-level split inflates mAP.
Both files hold {"train": [tile_id...], "val": [tile_id...], "params": {...}}.

Run:
  uv run python scripts/02_select_split.py --work-dir ~/swrd_paper_baseline/data/work --seed 0 --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from _common import TABLE3_SPLIT, write_json


def read_tiles(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.1, help="paper: 9:1")
    ap.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="negatives per positive (paper: equal number)",
    )
    args = ap.parse_args()

    tiles = read_tiles(args.work_dir / "tiles.jsonl")
    positives = [t for t in tiles if t["has_box"]]
    clean = [t for t in tiles if not t["touches_defect"]]
    rng = random.Random(args.seed)
    n_neg = min(len(clean), round(len(positives) * args.negative_ratio))
    negatives = rng.sample(clean, n_neg)
    selected = positives + negatives
    print(
        f"[select] positives {len(positives):,} | clean pool {len(clean):,} | negatives drawn {n_neg:,}"
    )

    # --- paper protocol: random over tiles ---------------------------------------------------------
    ids = [t["tile_id"] for t in selected]
    rng.shuffle(ids)
    n_val = round(len(ids) * args.val_ratio)
    split_tiles = {"val": sorted(ids[:n_val]), "train": sorted(ids[n_val:])}

    # --- thesis protocol: by exposure --------------------------------------------------------------
    by_exp: dict[str, list[str]] = defaultdict(list)
    for t in selected:
        by_exp[t["exposure"]].append(t["tile_id"])
    exposures = sorted(by_exp)
    rng.shuffle(exposures)
    val_ids, n_acc = [], 0
    target = len(ids) * args.val_ratio
    for e in exposures:  # greedy fill until the val tile budget is met
        if n_acc >= target:
            break
        val_ids.extend(by_exp[e])
        n_acc += len(by_exp[e])
    val_set = set(val_ids)
    split_films = {"val": sorted(val_ids), "train": sorted(i for i in ids if i not in val_set)}

    pos_ids = {t["tile_id"] for t in positives}

    def describe(split: dict) -> dict:
        d = {}
        for part in ("train", "val"):
            n_pos = sum(1 for i in split[part] if i in pos_ids)
            d[f"{part}_defect"] = n_pos
            d[f"{part}_background"] = len(split[part]) - n_pos
        return d

    params = vars(args) | {"work_dir": str(args.work_dir)}
    write_json(
        args.work_dir / "split_tiles.json",
        split_tiles | {"params": params, "counts": describe(split_tiles)},
    )
    write_json(
        args.work_dir / "split_films.json",
        split_films
        | {
            "params": params,
            "counts": describe(split_films),
            "n_val_exposures": len({t["exposure"] for t in selected if t["tile_id"] in val_set}),
        },
    )

    print("| | ours (tile split) | ours (film split) | paper Table 3 |\n|---|---:|---:|---:|")
    a, b = describe(split_tiles), describe(split_films)
    for k in ("train_defect", "train_background", "val_defect", "val_background"):
        print(f"| {k} | {a[k]:,} | {b[k]:,} | {TABLE3_SPLIT[k]:,} |")
    sub = Counter(t["subset"] for t in selected)
    print(f"[select] L/T mix of selected tiles: {dict(sub)}")
    print(f"[select] wrote split_tiles.json and split_films.json in {args.work_dir}")


if __name__ == "__main__":
    main()
