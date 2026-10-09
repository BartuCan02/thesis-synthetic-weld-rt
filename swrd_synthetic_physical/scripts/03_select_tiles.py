"""Step 4: keep only the synthetic tiles that hold an inserted defect.

Run after the baseline's 00_inventory.py and 01_tile.py on the synthetic films. 01_tile.py labels every box in
every tile, real or inserted, with the v1.0 rule. This script keeps a tile when at least one *inserted*
polygon (LabelMe flags.synthetic) passes that same keep rule (D7) in it. Other tiles of the synthetic films
are dropped: they only repeat the host film. This matches what oversampling adds: extra positive tiles.

Writes <work-dir>/split_tiles.json ({"train": [...], "val": [], ...}) for the baseline's 03_render.py, and
<work-dir>/selection_report.json.

    python 03_select_tiles.py --raw ~/swrd_synthetic_physical/data/raw_physical_v1 \
        --work-dir ~/swrd_synthetic_physical/data/work_physical_v1 --budget ../results/budget.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "swrd_paper_baseline" / "scripts"))
from _common import (
    CLASS_NAMES,
    LABEL_TO_CLASS,
    Polygon,
    box_from_visible,
    polygon_visible_mask,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--raw", type=Path, required=True, help="synthetic films (02_make_films.py --out)"
    )
    ap.add_argument(
        "--work-dir", type=Path, required=True, help="where 01_tile.py wrote tiles.jsonl"
    )
    ap.add_argument("--budget", type=Path, default=HERE.parent / "results" / "budget.json")
    ap.add_argument("--min-side-px", type=int, default=4)
    ap.add_argument("--min-visible-frac", type=float, default=0.10)
    ap.add_argument("--min-tile-frac", type=float, default=0.005)
    a = ap.parse_args()

    inserted: dict[str, list[Polygon]] = defaultdict(list)
    for js in (a.raw / "crop_weld_jsons").rglob("*.json"):
        d = json.loads(js.read_text(encoding="utf-8"))
        for s in d["shapes"]:
            if (s.get("flags") or {}).get("synthetic"):
                lab = s["label"]
                inserted[js.stem].append(
                    Polygon(
                        lab, LABEL_TO_CLASS[lab], np.asarray(s["points"], np.float32), "polygon"
                    )
                )

    keep, boxes_ins, tiles_ins, boxes_real = [], Counter(), Counter(), Counter()
    n_tiles = 0
    with open(a.work_dir / "tiles.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            n_tiles += 1
            classes = set()
            n_ins = 0
            for p in inserted.get(t["stem"], []):
                m = polygon_visible_mask(p, t["x"], t["y"], t["side"])
                if m is None or not m.any():
                    continue
                b = box_from_visible(p, m, a.min_side_px, a.min_visible_frac, a.min_tile_frac)
                if b is not None:
                    classes.add(p.class_id)
                    boxes_ins[p.class_id] += 1
                    n_ins += 1
            if not classes:
                continue
            keep.append(t["tile_id"])
            for c in classes:
                tiles_ins[c] += 1
            all_boxes = Counter(int(b.split()[0]) for b in t["boxes"])
            # boxes in the tile beyond the inserted ones come from the host film (real defects)
            extra = sum(all_boxes.values()) - n_ins
            if extra > 0:
                boxes_real["any"] += extra

    budget = json.loads(a.budget.read_text())
    target_tiles = {r["class"]: r["extra_tiles_to_match_rfs"] for r in budget["classes"]}
    per_class = {
        CLASS_NAMES[c]: {
            "tiles_with_inserted": tiles_ins[c],
            "inserted_boxes": boxes_ins[c],
            "target_extra_tiles": target_tiles.get(CLASS_NAMES[c]),
        }
        for c in sorted(tiles_ins)
    }
    report = {
        "synthetic_films": len(inserted),
        "tiles_on_synthetic_films": n_tiles,
        "tiles_kept": len(keep),
        "target_total_extra_tiles": budget["total_extra_tiles"],
        "per_class": per_class,
        "real_host_boxes_in_kept_tiles": boxes_real["any"],
        "params": {k: str(v) for k, v in vars(a).items()},
    }
    split = {
        "train": sorted(keep),
        "val": [],
        "params": report["params"],
        "counts": {"train": len(keep)},
    }
    (a.work_dir / "split_tiles.json").write_text(json.dumps(split))
    (a.work_dir / "selection_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
