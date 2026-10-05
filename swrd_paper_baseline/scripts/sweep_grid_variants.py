"""Sweep the unstated details of the paper's sliding window and compare against its counts.

Sect. 3.2 says: window side = half the shorter side, 50 % overlap, slide left-to-right and top-to-bottom,
"over 380,000" tiles, 80,648 with a defect; Table 2 gives instances per class on tiles. It does not say how
"half" is rounded, how the stride is rounded, or what happens at the border. This script enumerates those
choices, computes the total tile count for each from the image sizes (instant), and for the combinations
that land near the paper's total it also counts positive tiles and per-class instances with the D7 box rule.

Run:
  uv run python scripts/sweep_grid_variants.py --raw-dir ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from _common import (
    CLASS_NAMES,
    PAPER_N_POSITIVE_TILES,
    TABLE2_INSTANCES_TILES,
    box_from_visible,
    load_polygons,
    polygon_visible_mask,
)

SIDE_ROUND = {"floor": math.floor, "round": round, "ceil": math.ceil}
STRIDE_ROUND = {"floor": math.floor, "round": round, "ceil": math.ceil}


def positions(length: int, side: int, stride: int, border: str) -> list[int]:
    """Window start offsets along one axis for the border rule.

    inclusive  windows that fit entirely, a window ending exactly on the border counts   range(0, L-side+1)
    exclusive  like inclusive but a window ending exactly on the border does not count   range(0, L-side)
    flush      inclusive + one extra window pushed flush with the border if a remainder is left
    pad        keep sliding while the window *starts* inside the image (last ones hang over the border)
    """
    if length <= side:
        return [0]
    if border == "inclusive":
        return list(range(0, length - side + 1, stride))
    if border == "exclusive":
        return list(range(0, length - side, stride))
    if border == "flush":
        p = list(range(0, length - side + 1, stride))
        if p[-1] != length - side:
            p.append(length - side)
        return p
    if border == "pad":
        return list(range(0, length, stride))
    raise ValueError(border)


def grid(
    h: int, w: int, side_round: str, stride_round: str, border: str
) -> tuple[int, list[int], list[int]]:
    side = max(1, SIDE_ROUND[side_round](min(h, w) / 2))
    stride = max(1, STRIDE_ROUND[stride_round](side / 2))
    return side, positions(w, side, stride, border), positions(h, side, stride, border)


def count_positives(args: tuple) -> tuple[str, int, int, Counter]:
    rec, raw_dir, variant = args
    side_round, stride_round, border = variant
    polys, _ = load_polygons(Path(raw_dir) / rec["json"])
    polys = [p for p in polys if p.class_id is not None]
    side, xs, ys = grid(rec["height"], rec["width"], side_round, stride_round, border)
    n_tiles = len(xs) * len(ys)
    n_pos = 0
    per_class = Counter()
    for y in ys:
        for x in xs:
            has = False
            for p in polys:
                mask = polygon_visible_mask(p, x, y, side)
                if mask is None or not mask.any():
                    continue
                b = box_from_visible(p, mask, 4, 0.10, 0.005)
                if b is not None:
                    has = True
                    per_class[b.class_id] += 1
            n_pos += has
    return "|".join(variant), n_tiles, n_pos, per_class


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--target-low", type=int, default=375_000)
    ap.add_argument("--target-high", type=int, default=415_000)
    ap.add_argument("--max-phase2", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    inv = json.loads((args.work_dir / "inventory.json").read_text(encoding="utf-8"))
    dup = {s for g in inv["duplicate_groups"] for s in g[1:]}
    records_all = [r for r in inv["records"] if r["json"]]
    records_dedup = [r for r in records_all if r["stem"] not in dup]

    # ---- phase 1: totals from image sizes only ---------------------------------------------------
    rows = []
    for side_round, stride_round, border in itertools.product(
        SIDE_ROUND, STRIDE_ROUND, ["inclusive", "exclusive", "flush", "pad"]
    ):
        for dedup_name, recs in (("dedup", records_dedup), ("all", records_all)):
            total = 0
            for r in recs:
                _, xs, ys = grid(r["height"], r["width"], side_round, stride_round, border)
                total += len(xs) * len(ys)
            rows.append((total, side_round, stride_round, border, dedup_name))
    rows.sort()
    print("## Phase 1 — total tiles per variant (paper: 'over 380,000')\n")
    print(
        "| total tiles | side rounding | stride rounding | border | images |\n|---:|---|---|---|---|"
    )
    for total, sr, tr, b, d in rows:
        flag = " <--" if args.target_low <= total <= args.target_high else ""
        print(f"| {total:,} | {sr} | {tr} | {b} | {d} |{flag}")

    # ---- phase 2: positives for the candidates ---------------------------------------------------
    cands = [
        (sr, tr, b, d)
        for total, sr, tr, b, d in rows
        if args.target_low <= total <= args.target_high
    ]
    # keep only one images-set per grid variant to save time (dedup preferred)
    seen, chosen = set(), []
    for sr, tr, b, d in sorted(cands, key=lambda c: c[3] != "dedup"):
        if (sr, tr, b) not in seen:
            seen.add((sr, tr, b))
            chosen.append((sr, tr, b, d))
    chosen = chosen[: args.max_phase2]
    print(
        f"\n## Phase 2 — positives and Table 2 for {len(chosen)} candidate grid(s) (D7 defaults)\n"
    )
    print(
        "| variant | images | tiles | with box (paper 80,648) | "
        + " | ".join(f"{n} ({TABLE2_INSTANCES_TILES[n]:,})" for n in CLASS_NAMES)
        + " |"
    )
    print("|---|---|---:|---:|" + "---:|" * len(CLASS_NAMES))
    results = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for sr, tr, b, d in chosen:
            recs = records_dedup if d == "dedup" else records_all
            jobs = [(r, str(args.raw_dir), (sr, tr, b)) for r in recs]
            n_tiles = n_pos = 0
            per_class = Counter()
            for _, t, p, pc in ex.map(count_positives, jobs, chunksize=8):
                n_tiles += t
                n_pos += p
                per_class.update(pc)
            key = f"{sr}/{tr}/{b}"
            results[key] = {
                "images": d,
                "tiles": n_tiles,
                "positives": n_pos,
                "per_class": {CLASS_NAMES[c]: per_class[c] for c in range(len(CLASS_NAMES))},
            }
            cells = " | ".join(
                f"{per_class[c]:,} ({(per_class[c] - TABLE2_INSTANCES_TILES[n]) / TABLE2_INSTANCES_TILES[n]:+.1%})"
                for c, n in enumerate(CLASS_NAMES)
            )
            print(
                f"| {key} | {d} | {n_tiles:,} | {n_pos:,} ({(n_pos - PAPER_N_POSITIVE_TILES) / PAPER_N_POSITIVE_TILES:+.1%}) | {cells} |"
            )
    (args.work_dir / "grid_sweep.json").write_text(
        json.dumps({"phase1": rows, "phase2": results}, indent=1)
    )


if __name__ == "__main__":
    main()
