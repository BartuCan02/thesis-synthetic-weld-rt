"""Step 0: how many synthetic defects per class, matched to the oversampling run.

The synthetic run is compared with run A (film split, no oversampling) and run B (film split, repeat factor
sampling t = 0.1). To make "new synthetic defects" and "repeated real tiles" comparable, the synthetic run
adds, per class, as many extra training tiles as RFS adds per epoch. Then B and C differ only in what the
extra tiles show: repeats of the same real tiles (B) or real defects moved to new films and positions (C).

Everything is counted on the training films of the film split only (exposures not in split_films.json's val).
Films that are byte-identical copies of a val film are excluded as sources and hosts (leakage).

Writes <out>/budget.json and prints the table.

Run on the box:
    python 00_budget.py --work-dir ~/swrd_paper_baseline/data/work --out ../results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "swrd_paper_baseline" / "scripts"))
from _common import CLASS_NAMES


def exposure_of(tile_id: str) -> str:
    stem = tile_id.split("__")[0]
    return stem[2:] if stem[:2] in ("A_", "B_") else stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument(
        "--split-file", type=Path, default=None, help="default: <work-dir>/split_films.json"
    )
    ap.add_argument("--rfs-threshold", type=float, default=0.1)
    ap.add_argument(
        "--classes",
        nargs="+",
        default=["inclusion", "crack", "undercut", "lack_of_fusion"],
        help="classes to synthesise (default: the four that RFS t=0.1 boosts)",
    )
    ap.add_argument("--out", type=Path, default=HERE.parent / "results")
    args = ap.parse_args()

    split = json.loads((args.split_file or args.work_dir / "split_films.json").read_text())
    val_exp = {exposure_of(t) for t in split["val"]}
    train_ids = set(split["train"])
    inv = json.loads((args.work_dir / "inventory.json").read_text())
    recs = inv["records"]

    # exposures that are byte-identical copies of a val exposure
    exp_of_stem = {r["stem"]: r["exposure"] for r in recs}
    tainted = set()
    for group in inv["duplicate_groups"]:
        exps = {exp_of_stem[s] for s in group if s in exp_of_stem}
        if exps & val_exp:
            tainted |= exps - val_exp

    # ---- instances per class on training films ----------------------------------------------------
    inst, films, eligible = Counter(), Counter(), Counter()
    for r in recs:
        if r["exposure"] in val_exp:
            continue
        ci = r.get("class_instances") or [0] * len(CLASS_NAMES)
        for c, n in enumerate(ci):
            if n:
                inst[c] += n
                films[c] += 1
                if r["dtype"] == "uint16" and r["exposure"] not in tainted:
                    eligible[c] += n

    # ---- tiles per class in the training list, and what RFS does to them --------------------------
    tiles_with, boxes = Counter(), Counter()
    tile_classes: list[set[int]] = []
    with open(args.work_dir / "tiles.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t["tile_id"] not in train_ids:
                continue
            cs = [int(b.split()[0]) for b in t["boxes"]]
            for c in cs:
                boxes[c] += 1
            s = set(cs)
            for c in s:
                tiles_with[c] += 1
            tile_classes.append(s)
    n_train = len(tile_classes)
    assert n_train == len(train_ids), (n_train, len(train_ids))
    share = {c: tiles_with[c] / n_train for c in range(len(CLASS_NAMES))}
    r_cls = {
        c: max(1.0, math.sqrt(args.rfs_threshold / share[c])) if share[c] else 1.0 for c in share
    }
    seen = defaultdict(float)
    for s in tile_classes:
        r_t = max((r_cls[c] for c in s), default=1.0)
        for c in s:
            seen[c] += r_t

    rows, total_tiles, total_inst = [], 0, 0
    for name in args.classes:
        c = CLASS_NAMES.index(name)
        extra = seen[c] - tiles_with[c]
        tiles_per_inst = boxes[c] / inst[c]
        target = math.ceil(extra / tiles_per_inst)
        rows.append(
            {
                "class": name,
                "train_films": films[c],
                "train_instances": inst[c],
                "eligible_source_instances": eligible[c],
                "train_tiles_with_class": tiles_with[c],
                "rfs_repeat_factor": round(r_cls[c], 3),
                "rfs_tiles_per_epoch": round(seen[c]),
                "extra_tiles_to_match_rfs": round(extra),
                "tiles_per_instance": round(tiles_per_inst, 2),
                "synthetic_instances": target,
                "uses_per_source": round(target / max(eligible[c], 1), 2),
            }
        )
        total_tiles += round(extra)
        total_inst += target

    out = {
        "n_train_tiles": n_train,
        "n_val_exposures": len(val_exp),
        "train_exposures_identical_to_val": sorted(tainted),
        "rfs_threshold": args.rfs_threshold,
        "classes": rows,
        "total_extra_tiles": total_tiles,
        "total_synthetic_instances": total_inst,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "budget.json").write_text(json.dumps(out, indent=2))

    hdr = f"{'class':20s} {'films':>6s} {'inst':>6s} {'elig':>6s} {'tiles':>6s} {'r':>5s} {'+tiles':>7s} {'t/inst':>6s} {'synth':>6s} {'uses':>5s}"
    print(hdr)
    for r in rows:
        print(
            f"{r['class']:20s} {r['train_films']:6d} {r['train_instances']:6d} "
            f"{r['eligible_source_instances']:6d} {r['train_tiles_with_class']:6d} "
            f"{r['rfs_repeat_factor']:5.2f} {r['extra_tiles_to_match_rfs']:7d} "
            f"{r['tiles_per_instance']:6.2f} {r['synthetic_instances']:6d} {r['uses_per_source']:5.2f}"
        )
    print(
        f"total extra tiles {total_tiles:,} (+{total_tiles / n_train:.1%}); "
        f"synthetic instances {total_inst:,}; train exposures identical to a val film: {len(tainted)}"
    )


if __name__ == "__main__":
    main()
