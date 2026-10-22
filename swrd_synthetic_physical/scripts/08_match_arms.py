"""Step 5b: make two compared synthetic arms train on exactly the same tile counts (rule of 2026-10-19).

Two arms built by the pipeline (for example physical and naive, run_pipeline.sh without UPLOAD) never come out with the
same counts: defects land in a varying number of tiles and some placements fail in one arm only (runs D/E differed by 108
tiles). This step equalises them before upload. Every synthetic training tile of an arm falls in one bucket:

  "null"    an empty label: a null-implant tile (a clean spot that went through the removal step)
  "<c>"     every box in the tile is of class c (the normal case: kept tiles hold one inserted defect)
  "<c>+<d>" boxes of several classes (should not occur; kept so that nothing is silently mixed)

Per bucket the target is the smaller arm's count. From the larger arm, the surplus tiles are drawn at random with a fixed
seed (from the sorted tile list, so the draw is reproducible) and MOVED, image and label, to <yolo-dir>/trimmed/. Nothing
is deleted. Each arm gets <yolo-dir>/trim_report.json: the other arm, the seed, per-bucket counts before and after, and
the dropped tile ids; 04_merge_upload.py reads it, checks it, and records it in the dataset metadata.

Default is a dry run that prints what would move. --apply moves the files. Run once per pair, after both builds and
before either upload; a second run finds the arms equal and moves nothing.

    python 08_match_arms.py --arm ~/swrd_synthetic_physical/data/yolo_physical_v4 \
        --arm ~/swrd_synthetic_physical/data/yolo_naive_v4 --seed 0 [--apply]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def bucket_of(label_text: str) -> str:
    """The bucket of one YOLO label file's content (see the module docstring)."""
    classes = sorted({int(line.split()[0]) for line in label_text.splitlines() if line.strip()})
    return "null" if not classes else "+".join(str(c) for c in classes)


def read_arm(yolo_dir: Path) -> dict[str, list[str]]:
    """Bucket -> sorted tile ids of the arm's synthetic training tiles (images/train + labels/train)."""
    lab_dir = yolo_dir / "labels" / "train"
    out: dict[str, list[str]] = {}
    for lab in sorted(lab_dir.glob("*.txt")):
        if not (yolo_dir / "images" / "train" / f"{lab.stem}.png").is_file():
            raise FileNotFoundError(f"label without image: {lab}")
        out.setdefault(bucket_of(lab.read_text()), []).append(lab.stem)
    return out


def plan_trim(arms: list[dict[str, list[str]]], seed: int) -> list[dict[str, list[str]]]:
    """Per arm, bucket -> tile ids to drop so that every bucket has the smaller arm's count."""
    rng = random.Random(seed)
    drops: list[dict[str, list[str]]] = [{}, {}]
    for bucket in sorted(set(arms[0]) | set(arms[1])):
        n = [len(arm.get(bucket, [])) for arm in arms]
        target = min(n)
        for k in (0, 1):
            surplus = n[k] - target
            if surplus:
                drops[k][bucket] = sorted(rng.sample(sorted(arms[k][bucket]), surplus))
    return drops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--arm", type=Path, action="append", required=True, help="a yolo dir; give exactly two"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--apply", action="store_true", help="move the surplus tiles (default: dry run)"
    )
    a = ap.parse_args()
    if len(a.arm) != 2:
        ap.error("give exactly two --arm folders")
    dirs = [p.expanduser().resolve() for p in a.arm]
    arms = [read_arm(d) for d in dirs]
    drops = plan_trim(arms, a.seed)

    buckets = sorted(set(arms[0]) | set(arms[1]))
    print(f"{'bucket':10s}{dirs[0].name:>26s}{dirs[1].name:>26s}{'after':>10s}")
    for b in buckets:
        n0, n1 = len(arms[0].get(b, [])), len(arms[1].get(b, []))
        print(f"{b:10s}{n0:>26,}{n1:>26,}{min(n0, n1):>10,}")
    tot = [sum(len(v) for v in arm.values()) for arm in arms]
    after = sum(min(len(arms[0].get(b, [])), len(arms[1].get(b, []))) for b in buckets)
    print(f"{'total':10s}{tot[0]:>26,}{tot[1]:>26,}{after:>10,}")
    for k in (0, 1):
        print(f"[match] {dirs[k].name}: drop {sum(len(v) for v in drops[k].values()):,} tiles")
    if any("+" in b for b in buckets):
        print(
            "[match] WARNING: tiles with boxes of several classes exist; they are matched as their own bucket"
        )
    if not a.apply:
        print(
            "[match] dry run: nothing moved. Add --apply to move the surplus tiles to <arm>/trimmed/."
        )
        return

    for k in (0, 1):
        d, other = dirs[k], dirs[1 - k]
        report_path = d / "trim_report.json"
        prev = json.loads(report_path.read_text()) if report_path.is_file() else None
        for sub in ("images", "labels"):
            (d / "trimmed" / sub).mkdir(parents=True, exist_ok=True)
        dropped = [t for b in sorted(drops[k]) for t in drops[k][b]]
        for t in dropped:
            shutil.move(d / "images" / "train" / f"{t}.png", d / "trimmed" / "images" / f"{t}.png")
            shutil.move(d / "labels" / "train" / f"{t}.txt", d / "trimmed" / "labels" / f"{t}.txt")
        # a repeated run keeps the counts of the original build as "before" and adds up what was dropped
        per_bucket = Counter(prev["dropped_per_bucket"] if prev else {})
        for b, ts in drops[k].items():
            per_bucket[b] += len(ts)
        report = {
            "matched_with": str(other),
            "seed": a.seed,
            "buckets": {
                b: {
                    "before": (
                        prev["buckets"].get(b, {}).get("before", 0)
                        if prev
                        else len(arms[k].get(b, []))
                    ),
                    "after": min(len(arms[0].get(b, [])), len(arms[1].get(b, []))),
                }
                for b in buckets
            },
            "dropped": sorted(set(dropped) | set(prev["dropped"] if prev else [])),
            "dropped_per_bucket": dict(per_bucket),
        }
        report_path.write_text(json.dumps(report, indent=2))

    after_arms = [read_arm(d) for d in dirs]
    counts = [{b: len(v) for b, v in arm.items()} for arm in after_arms]
    assert counts[0] == counts[1], f"arms still differ after trimming: {counts}"
    print(
        f"[match] applied: both arms now hold {sum(counts[0].values()):,} tiles per bucket {counts[0]}"
    )


if __name__ == "__main__":
    main()
