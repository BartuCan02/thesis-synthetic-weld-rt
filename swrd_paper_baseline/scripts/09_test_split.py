"""Step 9: carve a held-out test set out of the film-split validation set.

Cecilia (2026-10-19): keep a small test set for the thesis, about 100 of the 369 validation exposures. Training is not
touched: the train tile list is copied unchanged, so every model trained so far still never saw a test tile.

The unit is the exposure (both halves of a T-joint film stay together, as in the film split). The choice is stratified
by class: for every class, the share of validation exposures holding it that go to test is as close to
n_test / n_val as possible, so rare classes (undercut sits in 16 validation exposures) land on both sides. Method:
iterative stratification for multi-label data (Sechidis, Tsoumakas, Vlahavas, ECML PKDD 2011), applied to groups:

  1. per class, the desired number of exposures in test = (exposures with the class) x n_test / n_val, and in val
     the rest; the desired totals are n_test and n_val - n_test;
  2. take the class with the fewest exposures still unassigned; walk its unassigned groups in seeded random order and
     put each where that class still lacks the most (ties: the side that lacks the most exposures overall, then a coin);
     subtract the group from the desired counts of every class it holds; repeat until every group with a defect is placed;
  3. place the defect-free groups on the side that still lacks the most exposures.

--n-test is a target, not an exact count: whole groups are placed to balance every class, and when every validation
exposure holds a defect there are no defect-free groups left to top the count up (seed 0, target 100: 96 exposures).

Groups: an exposure plus every exposure whose film is byte-identical to one of its films (inventory.json
"duplicate_groups"), so identical films never sit on both sides. A group with a copy in TRAINING is forced to val,
never test (three such train/val copies exist; they are a known small leak of the validation set, not of the test set).

Writes, next to the input split:
  split_films_test.json   {"train": [unchanged], "val": [reduced], "test": [...], "test_exposures": [...], ...}
  split_films_test_report.json   per side: exposures, crops, tiles, positive tiles, exposures / tiles / boxes per class

    python 09_test_split.py --work-dir ~/swrd_paper_baseline/data/work --n-test 100 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from _common import CLASS_NAMES, exposure_id

SIDES = ("val", "test")


def exposure_of_tile(tile_id: str) -> str:
    return exposure_id(tile_id.split("__")[0])


def build_groups(
    val_exposures: set[str], train_exposures: set[str], duplicate_stem_groups: list[list[str]]
) -> list[dict]:
    """Union-find over exposures linked by byte-identical films. Returns the groups that hold a val exposure, each
    as {"exposures": sorted list of its val exposures, "forced": "val" if any member is a train exposure, else None}."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in duplicate_stem_groups:
        exps = [exposure_id(s) for s in group]
        for e in exps[1:]:
            parent[find(e)] = find(exps[0])
    members: dict[str, set[str]] = defaultdict(set)
    for e in val_exposures | train_exposures | set(parent):
        members[find(e)].add(e)
    groups = []
    for m in members.values():
        val_m = sorted(m & val_exposures)
        if not val_m:
            continue
        groups.append({"exposures": val_m, "forced": "val" if m & train_exposures else None})
    groups.sort(key=lambda g: g["exposures"][0])
    return groups


def stratified_split(
    groups: list[dict], labels_of: dict[str, set[int]], n_test: int, seed: int
) -> dict[str, str]:
    """Iterative stratification over groups (see the module docstring). Returns exposure -> "val" | "test"."""
    rng = random.Random(seed)
    n_total = sum(len(g["exposures"]) for g in groups)
    if not 0 < n_test < n_total:
        raise ValueError(f"n_test must be between 1 and {n_total - 1}")
    ratio = {"test": n_test / n_total, "val": 1 - n_test / n_total}
    g_labels = [set().union(*(labels_of.get(e, set()) for e in g["exposures"])) for g in groups]
    g_size = [len(g["exposures"]) for g in groups]
    count = Counter()
    for labs, size in zip(g_labels, g_size):
        for c in labs:
            count[c] += size
    want = {s: {c: count[c] * ratio[s] for c in count} for s in SIDES}
    want_total = {"test": float(n_test), "val": float(n_total - n_test)}
    side_of: dict[int, str] = {}

    def assign(i: int, side: str) -> None:
        side_of[i] = side
        for c in g_labels[i]:
            want[side][c] -= g_size[i]
        want_total[side] -= g_size[i]

    for i, g in enumerate(groups):
        if g["forced"]:
            assign(i, g["forced"])
    while True:
        open_by_label = defaultdict(list)
        for i, labs in enumerate(g_labels):
            if i not in side_of:
                for c in labs:
                    open_by_label[c].append(i)
        if not open_by_label:
            break
        c = min(open_by_label, key=lambda k: (len(open_by_label[k]), k))
        idx = open_by_label[c][:]
        rng.shuffle(idx)
        for i in idx:
            if i in side_of:
                continue
            side = max(SIDES, key=lambda s: (want[s][c], want_total[s], rng.random()))
            assign(i, side)
    rest = [i for i in range(len(groups)) if i not in side_of]
    rng.shuffle(rest)
    for i in rest:
        assign(i, max(SIDES, key=lambda s: (want_total[s], rng.random())))
    return {e: side_of[i] for i, g in enumerate(groups) for e in g["exposures"]}


def summarise(tiles: list[dict], side_of: dict[str, str]) -> dict:
    out = {}
    for side in SIDES:
        ts = [t for t in tiles if side_of[t["exposure"]] == side]
        films_with, boxes, tiles_with = Counter(), Counter(), Counter()
        per_exp: dict[str, set[int]] = defaultdict(set)
        for t in ts:
            cs = [int(b.split()[0]) for b in t["boxes"]]
            for c in cs:
                boxes[c] += 1
            for c in set(cs):
                tiles_with[c] += 1
                per_exp[t["exposure"]].add(c)
        for cs in per_exp.values():
            for c in cs:
                films_with[c] += 1
        out[side] = {
            "exposures": len({t["exposure"] for t in ts}),
            "crops": len({t["stem"] for t in ts}),
            "tiles": len(ts),
            "positive_tiles": sum(1 for t in ts if t["boxes"]),
            "per_class": {
                CLASS_NAMES[c]: {
                    "exposures": films_with[c],
                    "tiles": tiles_with[c],
                    "boxes": boxes[c],
                }
                for c in range(len(CLASS_NAMES))
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--work-dir", type=Path, required=True, help="tiles.jsonl, inventory.json, split_films.json"
    )
    ap.add_argument(
        "--split", default="split_films.json", help="the film split to carve the test set from"
    )
    ap.add_argument("--n-test", type=int, default=100, help="target number of validation exposures that become test (can end a few short)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="split_films_test.json")
    a = ap.parse_args()

    src = a.work_dir / a.split
    split = json.loads(src.read_text())
    val_ids = set(split["val"])
    train_exposures = {exposure_of_tile(t) for t in split["train"]}
    val_tiles = []
    with open(a.work_dir / "tiles.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t["tile_id"] in val_ids:
                t["exposure"] = exposure_of_tile(t["tile_id"])
                val_tiles.append(t)
    assert len(val_tiles) == len(val_ids), (len(val_tiles), len(val_ids))
    val_exposures = {t["exposure"] for t in val_tiles}
    labels_of: dict[str, set[int]] = defaultdict(set)
    for t in val_tiles:
        for b in t["boxes"]:
            labels_of[t["exposure"]].add(int(b.split()[0]))

    inv = json.loads((a.work_dir / "inventory.json").read_text())
    groups = build_groups(val_exposures, train_exposures, inv["duplicate_groups"])
    side_of = stratified_split(groups, labels_of, a.n_test, a.seed)
    test_ids = sorted(t["tile_id"] for t in val_tiles if side_of[t["exposure"]] == "test")
    new_val = sorted(t["tile_id"] for t in val_tiles if side_of[t["exposure"]] == "val")
    test_exposures = sorted(e for e, s in side_of.items() if s == "test")
    report = {
        "source": str(src),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "n_test_requested": a.n_test,
        "seed": a.seed,
        "groups": len(groups),
        "groups_forced_to_val": sum(1 for g in groups if g["forced"]),
        "sides": summarise(val_tiles, side_of),
    }
    out = {
        "train": split["train"],
        "val": new_val,
        "test": test_ids,
        "test_exposures": test_exposures,
        "n_val_exposures": report["sides"]["val"]["exposures"],
        "n_test_exposures": report["sides"]["test"]["exposures"],
        "params": {
            **split.get("params", {}),
            "test_split": {
                "source": a.split,
                "n_test": a.n_test,
                "seed": a.seed,
                "method": "iterative stratification",
            },
        },
        "counts": {"train": len(split["train"]), "val": len(new_val), "test": len(test_ids)},
    }
    (a.work_dir / a.out).write_text(json.dumps(out))
    (a.work_dir / a.out.replace(".json", "_report.json")).write_text(json.dumps(report, indent=2))

    s = report["sides"]
    print(
        f"[test split] {len(groups)} groups, {report['groups_forced_to_val']} forced to val (copy in train)"
    )
    print(f"{'':22s}{'val':>22s}{'test':>22s}")
    for k in ("exposures", "crops", "tiles", "positive_tiles"):
        print(f"{k:22s}{s['val'][k]:>22,}{s['test'][k]:>22,}")
    print(
        f"{'class':22s}{'expos / tiles / boxes':>22s}{'expos / tiles / boxes':>22s}  test share of exposures"
    )
    for c in CLASS_NAMES:
        v, t = s["val"]["per_class"][c], s["test"]["per_class"][c]
        share = t["exposures"] / max(t["exposures"] + v["exposures"], 1)
        print(
            f"{c:22s}{v['exposures']:>6} /{v['tiles']:>6} /{v['boxes']:>6}  "
            f"{t['exposures']:>6} /{t['tiles']:>6} /{t['boxes']:>6}  {share:.0%}"
        )
    print(f"[test split] wrote {a.work_dir / a.out}")


if __name__ == "__main__":
    main()
