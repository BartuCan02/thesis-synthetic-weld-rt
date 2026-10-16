"""Step 4: keep only the synthetic tiles that hold an inserted defect.

Run after the baseline's 00_inventory.py and 01_tile.py on the synthetic films. 01_tile.py labels every box in
every tile, real or inserted, with the v1.0 rule. This script keeps a tile when at least one *inserted*
polygon (LabelMe flags.synthetic) passes that same keep rule (D7) in it. Other tiles of the synthetic films
are dropped: they only repeat the host film. This matches what oversampling adds: extra positive tiles.

Version 3 (2026-10-16): null implants (inserted.jsonl records with class "null", no label polygon) make
*negative* tiles that carry the pipeline's processing traces. A tile whose window holds a null implant's centre
and no inserted defect is kept as a null tile; --null-holdout of them (random, seeded) are kept out of the
training list and written to <work-dir>/null_holdout.json for the post-training check (detections on them =
insertion artefacts learned).

Writes <work-dir>/split_tiles.json ({"train": [...], "val": [], ...}) for the baseline's 03_render.py, and
<work-dir>/selection_report.json.

    python 03_select_tiles.py --raw ~/swrd_synthetic_physical/data/raw_physical_v1 \
        --work-dir ~/swrd_synthetic_physical/data/work_physical_v1 --budget ../results/budget.json
"""

from __future__ import annotations

import argparse
import json
import random
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
    ap.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="baseline split_films.json; if given, count inserted defects that touch a baseline tile",
    )
    ap.add_argument("--null-holdout", type=int, default=300, help="null tiles kept out of training")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    nulls: dict[str, list[tuple[float, float]]] = defaultdict(list)
    ins_path = a.raw / "inserted.jsonl"
    if ins_path.is_file():
        for line in ins_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["class"] == "null":
                (x, y), (h, w) = r["top_left"], r["patch_hw"]
                nulls[r["film"]].append((x + w / 2, y + h / 2))

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
    null_tiles, null_tiles_positive = [], 0
    n_tiles = 0
    with open(a.work_dir / "tiles.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            n_tiles += 1
            classes = set()
            n_ins = 0
            has_null = any(
                t["x"] <= cx < t["x"] + t["side"] and t["y"] <= cy < t["y"] + t["side"]
                for cx, cy in nulls.get(t["stem"], [])
            )
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
                if has_null:
                    if t["boxes"]:
                        null_tiles_positive += 1
                    else:
                        null_tiles.append(t["tile_id"])
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
    # version-2 check: no inserted defect may touch a tile the baseline already uses
    touching = None
    if a.split_file:
        split_b = json.loads(a.split_file.read_text())
        win = defaultdict(list)
        for tid in split_b["train"] + split_b["val"]:
            st, xs, ys, ss = tid.split("__")
            win[st].append((int(xs[1:]), int(ys[1:]), int(ss[1:])))
        touching = 0
        for stem, polys in inserted.items():
            host = stem.split("-", 2)[2]
            for p in polys:
                if any(
                    (m := polygon_visible_mask(p, wx, wy, ws)) is not None and m.any()
                    for wx, wy, ws in win.get(host, [])
                ):
                    touching += 1
    rng = random.Random(a.seed)
    null_tiles.sort()
    holdout = sorted(rng.sample(null_tiles, min(a.null_holdout, len(null_tiles))))
    null_train = sorted(set(null_tiles) - set(holdout))
    (a.work_dir / "null_holdout.json").write_text(json.dumps({"tiles": holdout}))
    report = {
        "synthetic_films": len(inserted),
        "tiles_on_synthetic_films": n_tiles,
        "tiles_kept": len(keep) + len(null_train),
        "tiles_with_inserted_defect": len(keep),
        "null_tiles_train": len(null_train),
        "null_tiles_holdout": len(holdout),
        "null_tiles_dropped_positive": null_tiles_positive,
        "target_total_extra_tiles": budget["total_extra_tiles"],
        "per_class": per_class,
        "real_host_boxes_in_kept_tiles": boxes_real["any"],
        "inserted_defects_touching_a_baseline_tile": touching,
        "params": {k: str(v) for k, v in vars(a).items()},
    }
    train = sorted(keep + null_train)
    split = {
        "train": train,
        "val": [],
        "params": report["params"],
        "counts": {"train": len(train), "null_train": len(null_train)},
    }
    (a.work_dir / "split_tiles.json").write_text(json.dumps(split))
    (a.work_dir / "selection_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
