"""Step 2 — index every sliding-window tile and its boxes (Sect. 3.2 + polygon->box rule D7).

No pixels are written here. For every image in ``inventory.json`` the script enumerates the windows of the
paper's rule (side = half the shorter side, 50 % overlap), rasterises each polygon into each window it
touches, and keeps a box when the visible part passes the D7 rule. The result is one JSON-lines record per
tile with its film, offset, side, boxes, and two flags used later for negative sampling:

  has_box        at least one kept box of the six classes
  touches_defect any defect polygon (mapped class, or an unmapped defect label listed with
                 --exclude-from-negatives) has visible pixels in the tile, even if no box was kept

A tile is a usable *negative* only if ``touches_defect`` is false — a tile holding a sliver we dropped is not
"defect-free". Table 2 of the paper is re-counted at the end (instances per class on tiles).

Run:
  uv run python scripts/01_tile.py --raw-dir ~/swrd_paper_baseline/data/raw --work-dir ~/swrd_paper_baseline/data/work \
      --edge flush --min-side-px 4 --min-visible-frac 0.10 --min-tile-frac 0.005
"""

from __future__ import annotations

import argparse
import json
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
    tile_positions,
    tile_side,
    write_json,
)
from tqdm import tqdm

# Labels that are *defects in the release but not in the paper's six classes*. Tiles touched by them are
# never used as negatives. Filled from the inventory's "unmapped" list once we have seen it; the default
# covers the known pseudo-defect label, anything else is passed with --exclude-from-negatives.
DEFAULT_EXCLUDE_FROM_NEGATIVES: list[str] = []


def tile_one_image(args: tuple) -> tuple[list[dict], Counter]:
    (
        rec,
        raw_dir,
        side_fraction,
        overlap,
        edge,
        min_side_px,
        min_visible_frac,
        min_tile_frac,
        excl,
    ) = args
    h, w = rec["height"], rec["width"]
    side = tile_side(h, w, side_fraction)
    xs = tile_positions(w, side, overlap, edge)
    ys = tile_positions(h, side, overlap, edge)

    polys = []
    if rec["json"]:
        polys, _ = load_polygons(Path(raw_dir) / rec["json"])
    defect_polys = [p for p in polys if p.class_id is not None or p.label in excl]

    tiles: list[dict] = []
    per_class = Counter()
    for y in ys:
        for x in xs:
            boxes: list[str] = []
            touches = False
            for p in defect_polys:
                mask = polygon_visible_mask(p, x, y, side)
                if mask is None or not mask.any():
                    continue
                touches = True
                if p.class_id is None:
                    continue
                b = box_from_visible(p, mask, min_side_px, min_visible_frac, min_tile_frac)
                if b is not None:
                    boxes.append(b.to_yolo(side))
                    per_class[b.class_id] += 1
            tiles.append(
                {
                    "tile_id": f"{rec['stem']}__x{x}__y{y}__s{side}",
                    "stem": rec["stem"],
                    "exposure": rec["exposure"],
                    "subset": rec["subset"],
                    "image": rec["image"],
                    "x": x,
                    "y": y,
                    "side": side,
                    "boxes": boxes,
                    "has_box": bool(boxes),
                    "touches_defect": touches,
                }
            )
    return tiles, per_class


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument(
        "--side-fraction", type=float, default=0.5, help="window side / shorter side (paper: 0.5)"
    )
    ap.add_argument("--overlap", type=float, default=0.5, help="window overlap (paper: 0.5)")
    ap.add_argument(
        "--edge",
        choices=["flush", "drop"],
        default="flush",
        help="UNSPECIFIED; see _common.tile_positions",
    )
    ap.add_argument("--min-side-px", type=int, default=4, help="D7: drop boxes thinner than this")
    ap.add_argument(
        "--min-visible-frac",
        type=float,
        default=0.10,
        help="D7: keep if >= this fraction of the polygon is visible",
    )
    ap.add_argument(
        "--min-tile-frac",
        type=float,
        default=0.005,
        help="D7: ...or if the visible box covers >= this fraction of the tile",
    )
    ap.add_argument(
        "--exclude-from-negatives",
        nargs="*",
        default=DEFAULT_EXCLUDE_FROM_NEGATIVES,
        help="raw label strings (not in the six classes) whose tiles may not be negatives",
    )
    ap.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="drop all but the first file of each byte-identical group",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    inv = json.loads((args.work_dir / "inventory.json").read_text(encoding="utf-8"))
    records = [r for r in inv["records"] if r["json"] is not None]
    if args.skip_duplicates:
        drop = {s for g in inv["duplicate_groups"] for s in g[1:]}
        records = [r for r in records if r["stem"] not in drop]
        print(f"[tile] dropped {len(drop)} duplicate files")
    print(f"[tile] {len(records)} images with labels")

    jobs = [
        (
            r,
            str(args.raw_dir),
            args.side_fraction,
            args.overlap,
            args.edge,
            args.min_side_px,
            args.min_visible_frac,
            args.min_tile_frac,
            set(args.exclude_from_negatives),
        )
        for r in records
    ]
    total_per_class = Counter()
    n_tiles = n_pos = n_touch = 0
    out_path = args.work_dir / "tiles.jsonl"
    with ProcessPoolExecutor(args.workers) as ex, open(out_path, "w", encoding="utf-8") as out:
        for tiles, per_class in tqdm(
            ex.map(tile_one_image, jobs, chunksize=4), total=len(jobs), unit="img"
        ):
            total_per_class.update(per_class)
            out.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in tiles)
            n_tiles += len(tiles)
            n_pos += sum(t["has_box"] for t in tiles)
            n_touch += sum(t["touches_defect"] for t in tiles)

    summary = {
        "params": {
            k: v for k, v in vars(args).items() if k not in ("raw_dir", "work_dir", "workers")
        },
        "n_images": len(records),
        "n_tiles": n_tiles,
        "n_tiles_with_box": n_pos,
        "n_tiles_touching_defect": n_touch,
        "n_clean_tiles": n_tiles - n_touch,
        "paper_positive_tiles": PAPER_N_POSITIVE_TILES,
        "instances_per_class": {
            CLASS_NAMES[c]: total_per_class[c] for c in range(len(CLASS_NAMES))
        },
        "table2_paper": TABLE2_INSTANCES_TILES,
    }
    write_json(args.work_dir / "tiles_summary.json", summary)

    print(
        f"\n[tile] tiles: {n_tiles:,} | with box: {n_pos:,} (paper {PAPER_N_POSITIVE_TILES:,}) | "
        f"touching a defect: {n_touch:,} | clean: {n_tiles - n_touch:,}"
    )
    print("| class | ours (tiles) | Table 2 | diff |\n|---|---:|---:|---:|")
    for c, name in enumerate(CLASS_NAMES):
        print(
            f"| {name} | {total_per_class[c]} | {TABLE2_INSTANCES_TILES[name]} | {total_per_class[c] - TABLE2_INSTANCES_TILES[name]:+d} |"
        )
    print(f"[tile] wrote {out_path} and tiles_summary.json")


if __name__ == "__main__":
    main()
