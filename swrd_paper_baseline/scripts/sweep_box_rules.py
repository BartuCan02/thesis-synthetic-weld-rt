"""Second sweep: the polygon -> box rule, on the grid that matched the paper's tile count.

Candidate rules for a polygon that touches a window:
  d7         our default: rasterised polygon, piece >= 4 px, >= 10 % visible or >= 0.5 % of the tile
  poly_any   rasterised polygon, keep any visible piece >= 1 px
  bbox_any   clip the polygon's *bounding box* to the window, keep if the clipped box is >= 1 px both ways
             (a long sloped line gets boxes in windows its pixels never enter — the naive LabelMe->YOLO path)
  bbox_4px   same, but clipped box >= 4 px both ways
  bbox_10pct same, clipped box area >= 10 % of the full box area

All rules are evaluated in one pass. Run:
  uv run python scripts/sweep_box_rules.py --raw-dir ... --work-dir ... --side-round round --stride-round round --border inclusive
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from _common import (
    CLASS_NAMES,
    PAPER_N_POSITIVE_TILES,
    TABLE2_INSTANCES_TILES,
    box_from_visible,
    load_polygons,
    polygon_visible_mask,
)
from sweep_grid_variants import grid

BBOX_MIN_PX = [1, 4, 6, 8, 10, 12, 16, 20, 24, 32]
BBOX_REL = [0.01, 0.02, 0.03, 0.05, 0.08]  # min clipped side as a fraction of the window side
RULES = (
    ["d7", "poly_any", "bbox_10pct"]
    + [f"bbox_{m}px" for m in BBOX_MIN_PX]
    + [f"bbox_rel{r}" for r in BBOX_REL]
)


def clipped_bbox(poly, x: int, y: int, side: int) -> tuple[int, int, int, int] | None:
    bx0, by0, bx1, by1 = poly.bbox
    x0, y0 = max(bx0, x), max(by0, y)
    x1, y1 = min(bx1, x + side), min(by1, y + side)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(np.floor(x0 - x)), int(np.floor(y0 - y)), int(np.ceil(x1 - x)), int(np.ceil(y1 - y))


def one_image(args: tuple) -> dict:
    rec, raw_dir, variant = args
    polys, _ = load_polygons(Path(raw_dir) / rec["json"])
    polys = [p for p in polys if p.class_id is not None]
    side, xs, ys = grid(rec["height"], rec["width"], *variant)
    pos = Counter()
    inst = {r: Counter() for r in RULES}
    for y in ys:
        for x in xs:
            has = {r: False for r in RULES}
            for p in polys:
                cb = clipped_bbox(p, x, y, side)
                if cb is None:
                    continue
                cx0, cy0, cx1, cy1 = cb
                w, h = cx1 - cx0, cy1 - cy0
                full_w = max(p.bbox[2] - p.bbox[0], 1e-6)
                full_h = max(p.bbox[3] - p.bbox[1], 1e-6)
                for m in BBOX_MIN_PX:
                    if w >= m and h >= m:
                        has[f"bbox_{m}px"] = True
                        inst[f"bbox_{m}px"][p.class_id] += 1
                for rel in BBOX_REL:
                    if w >= rel * side and h >= rel * side:
                        has[f"bbox_rel{rel}"] = True
                        inst[f"bbox_rel{rel}"][p.class_id] += 1
                if (w * h) / (full_w * full_h) >= 0.10:
                    has["bbox_10pct"] = True
                    inst["bbox_10pct"][p.class_id] += 1
                mask = polygon_visible_mask(p, x, y, side)
                if mask is None or not mask.any():
                    continue
                has["poly_any"] = True
                inst["poly_any"][p.class_id] += 1
                if box_from_visible(p, mask, 4, 0.10, 0.005) is not None:
                    has["d7"] = True
                    inst["d7"][p.class_id] += 1
            for r in RULES:
                pos[r] += has[r]
    return {"tiles": len(xs) * len(ys), "pos": pos, "inst": inst}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--side-round", default="round")
    ap.add_argument("--stride-round", default="round")
    ap.add_argument("--border", default="inclusive")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    inv = json.loads((args.work_dir / "inventory.json").read_text(encoding="utf-8"))
    dup = {s for g in inv["duplicate_groups"] for s in g[1:]}
    recs = [r for r in inv["records"] if r["json"] and r["stem"] not in dup]
    variant = (args.side_round, args.stride_round, args.border)
    jobs = [(r, str(args.raw_dir), variant) for r in recs]
    tiles = 0
    pos = Counter()
    inst = {r: Counter() for r in RULES}
    with ProcessPoolExecutor(args.workers) as ex:
        for out in ex.map(one_image, jobs, chunksize=8):
            tiles += out["tiles"]
            pos.update(out["pos"])
            for r in RULES:
                inst[r].update(out["inst"][r])
    print(f"grid {'/'.join(variant)}: {tiles:,} tiles on {len(recs)} images\n")
    print(
        "| rule | with box (80,648) | "
        + " | ".join(f"{n} ({TABLE2_INSTANCES_TILES[n]:,})" for n in CLASS_NAMES)
        + " |"
    )
    print("|---|---:|" + "---:|" * len(CLASS_NAMES))
    for r in RULES:
        cells = " | ".join(
            f"{inst[r][c]:,} ({(inst[r][c] - TABLE2_INSTANCES_TILES[n]) / TABLE2_INSTANCES_TILES[n]:+.1%})"
            for c, n in enumerate(CLASS_NAMES)
        )
        print(
            f"| {r} | {pos[r]:,} ({(pos[r] - PAPER_N_POSITIVE_TILES) / PAPER_N_POSITIVE_TILES:+.1%}) | {cells} |"
        )
    (args.work_dir / "box_rule_sweep.json").write_text(
        json.dumps(
            {
                "grid": variant,
                "tiles": tiles,
                "pos": pos,
                "inst": {r: dict(inst[r]) for r in RULES},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
