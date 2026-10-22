"""Step 10: register the held-out test set (09_test_split.py) as its own ClearML dataset.

The test tiles are the same 8-bit PNGs and YOLO box labels as in the v1.0 tile dataset (de772ad9...): they are copied
from the local v1.0 folder, not re-rendered, so a model scores the identical pixels whether it reads them from v1.0 with
the split file or from this dataset. The dataset is standalone (no parent): ClearML's content de-duplication against a
parent (see 07_merge_upload.py) cannot touch it.

Layout:
  images/test/*.png, labels/test/*.txt   the test tiles and their labels
  swrd6.yaml                             train, val and test all point at images/test: an evaluation-only dataset
                                         (Ultralytics refuses a data file without a train entry)
  split/split_films_test.json            the full split (train unchanged, reduced val, test), so it can be rebuilt
  split/split_films_test_report.json     per-side counts from 09_test_split.py
  split/test_exposures.txt               the test exposures, one per line

Default is a dry run: the folder is assembled and checked, nothing is uploaded. --upload creates the dataset.

    AWS_PROFILE=data-rw python 10_upload_test_set.py --work-dir ~/swrd_paper_baseline/data/work \
        --yolo-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid [--upload]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

PROJECT = "thesis_wp1_benchmark"
DEFAULT_OUTPUT_URI = "s3://clearml-data-deeplify/clearml-datasets"
PARENT_TILES = "de772ad9363c4067bed5835e13a9be81"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--work-dir", type=Path, required=True, help="holds split_films_test.json and its report"
    )
    ap.add_argument("--split", default="split_films_test.json")
    ap.add_argument(
        "--yolo-dir", type=Path, required=True, help="local copy of the v1.0 tile dataset"
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path.home() / "swrd_paper_baseline/data/yolo_v1.0_test"
    )
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--name", default="swrd-paper-tiles-film-test")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI)
    ap.add_argument(
        "--upload", action="store_true", help="create the ClearML dataset (default: dry run)"
    )
    a = ap.parse_args()

    split_path = a.work_dir / a.split
    report_path = a.work_dir / a.split.replace(".json", "_report.json")
    split = json.loads(split_path.read_text())
    report = json.loads(report_path.read_text())
    test_ids = split["test"]
    assert test_ids, f"{split_path.name} has no test list"
    assert not set(test_ids) & set(split["val"]), "a tile is in both val and test"
    assert not set(test_ids) & set(split["train"]), "a test tile is in train"

    # assemble: copy each test tile and its label from whichever v1.0 folder holds it
    shutil.rmtree(a.out_dir, ignore_errors=True)
    for sub in ("images/test", "labels/test", "split"):
        (a.out_dir / sub).mkdir(parents=True)
    missing, boxes = [], Counter()
    for tid in test_ids:
        part = next(
            (p for p in ("train", "val") if (a.yolo_dir / "images" / p / f"{tid}.png").is_file()),
            None,
        )
        if part is None:
            missing.append(tid)
            continue
        shutil.copy2(
            a.yolo_dir / "images" / part / f"{tid}.png", a.out_dir / "images/test" / f"{tid}.png"
        )
        lab = a.yolo_dir / "labels" / part / f"{tid}.txt"
        shutil.copy2(lab, a.out_dir / "labels/test" / f"{tid}.txt")
        for line in lab.read_text().splitlines():
            if line.strip():
                boxes[int(line.split()[0])] += 1
    assert not missing, f"{len(missing)} test tiles not in {a.yolo_dir}, e.g. {missing[:3]}"

    names = yaml.safe_load((a.yolo_dir / "swrd6.yaml").read_text())["names"]
    expected = {names[c]: report["sides"]["test"]["per_class"][names[c]]["boxes"] for c in names}
    got = {names[c]: boxes[c] for c in names}
    assert got == expected, f"box counts differ from the split report: {got} vs {expected}"
    (a.out_dir / "swrd6.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/test",
                "val": "images/test",
                "test": "images/test",
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )
    shutil.copy2(split_path, a.out_dir / "split" / split_path.name)
    shutil.copy2(report_path, a.out_dir / "split" / report_path.name)
    (a.out_dir / "split/test_exposures.txt").write_text("\n".join(split["test_exposures"]) + "\n")
    render = a.yolo_dir / "render_params.json"
    if render.is_file():
        shutil.copy2(render, a.out_dir / "render_params.json")

    side = report["sides"]["test"]
    print(
        f"[test set] {len(test_ids):,} tiles of {side['exposures']} exposures ({side['crops']} crops), "
        f"{side['positive_tiles']:,} with a box; boxes per class {got}"
    )
    print(f"[test set] assembled in {a.out_dir}")
    if not a.upload:
        print("[test set] dry run: nothing uploaded. Add --upload to create the ClearML dataset.")
        return

    from clearml import Dataset

    ds = Dataset.create(
        dataset_name=a.name,
        dataset_project=a.project,
        dataset_version=a.version,
        output_uri=a.output_uri,
        description=(
            f"Held-out test set of the SWRD film split: {side['exposures']} of the 369 validation exposures, chosen by "
            f"iterative stratification over classes (09_test_split.py, seed {report['seed']}). Same 8-bit tiles and "
            f"labels as the v1.0 tile dataset {PARENT_TILES}. Evaluation only; never train on it. Thesis WP1."
        ),
    )
    ds.add_files(path=str(a.out_dir), verbose=False)
    ds.set_metadata(
        {
            "source_tiles_dataset": PARENT_TILES,
            "source_split": report["source"],
            "source_split_sha256": report["source_sha256"],
            "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "seed": report["seed"],
            "test_tiles": len(test_ids),
            "test_exposures": side["exposures"],
            "per_class_test": side["per_class"],
            "per_class_val_after": report["sides"]["val"]["per_class"],
        },
        metadata_name="test_split",
    )
    ds.upload(show_progress=True)
    ds.finalize()
    n = Counter(p.split("/")[0] for p in ds.list_files())
    print(f"[test set] dataset {ds.id} ({a.name} {a.version}): {dict(n)}")
    assert sum(1 for p in ds.list_files() if p.startswith("images/test/")) == len(test_ids)


if __name__ == "__main__":
    main()
