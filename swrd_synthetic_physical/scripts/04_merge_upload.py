"""Step 6: register "SWRD v1.0 tiles + synthetic train tiles" as a ClearML child dataset of v1.0.

Same construction as swrd_paper_baseline/scripts/07_merge_upload.py (the customer side run): every file of
the parent is kept, the synthetic tiles are added to images/train and labels/train only, and the val folder
stays byte-identical. Under the film split (05_train.py --split film) every synthetic tile lands in train,
because no synthetic exposure (SYN-...) is in split_films.json's val list.

Refuses to run if the synthetic folder has val tiles or a tile id that already exists in the parent.

    AWS_PROFILE=data-rw python 04_merge_upload.py --parent-id de772ad9363c4067bed5835e13a9be81 \
        --yolo-dir ~/swrd_synthetic_physical/data/yolo_physical_v1 \
        --work-dir ~/swrd_synthetic_physical/data/work_physical_v1 \
        --raw-dir ~/swrd_synthetic_physical/data/raw_physical_v1 --arm physical
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from clearml import Dataset

HERE = Path(__file__).resolve().parent
PROJECT = "thesis_wp1_benchmark"
DEFAULT_OUTPUT_URI = "s3://clearml-data-deeplify/clearml-datasets"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--parent-id", required=True, help="the SWRD v1.0 tile dataset")
    ap.add_argument("--yolo-dir", type=Path, required=True, help="output of 03_render.py")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True, help="output of 02_make_films.py")
    ap.add_argument("--budget", type=Path, default=HERE.parent / "results" / "budget.json")
    ap.add_argument("--arm", choices=["physical", "naive"], required=True)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--name", default=None, help="default: swrd-paper-tiles-plus-synthetic-<arm>")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI)
    ap.add_argument("--dry-run", action="store_true", help="run every check, create nothing")
    a = ap.parse_args()
    name = a.name or f"swrd-paper-tiles-plus-synthetic-{a.arm}"

    img_dir, lab_dir = a.yolo_dir / "images" / "train", a.yolo_dir / "labels" / "train"
    imgs = sorted(p.name for p in img_dir.glob("*.png"))
    labs = sorted(p.name for p in lab_dir.glob("*.txt"))
    assert imgs, f"no synthetic train tiles under {img_dir}"
    assert [Path(i).stem for i in imgs] == [Path(x).stem for x in labs], "image/label mismatch"
    assert all(i.startswith("SYN-") for i in imgs), (
        "a non-synthetic tile is in the synthetic folder"
    )
    val_dir = a.yolo_dir / "images" / "val"
    assert not val_dir.exists() or not any(val_dir.iterdir()), "synthetic folder has val tiles"

    parent = Dataset.get(dataset_id=a.parent_id)
    parent_files = set(parent.list_files())
    clashes = [
        i for i in imgs if f"images/train/{i}" in parent_files or f"images/val/{i}" in parent_files
    ]
    assert not clashes, f"{len(clashes)} tile ids already exist in the parent, e.g. {clashes[:3]}"
    n_parent = Counter(p.split("/")[1] for p in parent_files if p.startswith("images/"))
    n_pos = sum(1 for x in labs if (lab_dir / x).stat().st_size > 0)
    assert n_pos == len(labs), "every synthetic tile must hold at least one box"

    budget = json.loads(a.budget.read_text())
    selection = json.loads((a.work_dir / "selection_report.json").read_text())
    make = json.loads((a.raw_dir / "make_films_report.json").read_text())
    tiling = json.loads((a.work_dir / "tiles_summary.json").read_text())
    render = json.loads((a.yolo_dir / "render_params.json").read_text())
    print(
        f"[merge] parent {parent.name} {parent.version}: {dict(n_parent)}; "
        f"adding {len(imgs):,} synthetic train tiles ({a.arm})"
    )
    if a.dry_run:
        print("[merge] dry run OK: no clash with the parent; nothing created")
        return

    ds = Dataset.create(
        dataset_name=name,
        dataset_project=a.project,
        dataset_version=a.version,
        parent_datasets=[a.parent_id],
        output_uri=a.output_uri,
        description=(
            f"SWRD paper-baseline tiles v1.0 (parent, unchanged incl. val) + synthetic train tiles: real "
            f"SWRD defects of the rare classes moved into other SWRD training films ({a.arm} insertion), "
            f"budget matched to repeat factor sampling t={budget['rfs_threshold']}. Thesis WP2."
        ),
    )
    ds.add_files(path=str(img_dir), dataset_path="images/train", verbose=False)
    ds.add_files(path=str(lab_dir), dataset_path="labels/train", verbose=False)
    for p in (
        a.budget,
        a.work_dir / "selection_report.json",
        a.raw_dir / "make_films_report.json",
        a.raw_dir / "inserted.jsonl",
        a.yolo_dir / "render_params.json",
        a.work_dir / "tiles_summary.json",
    ):
        ds.add_files(path=str(p), dataset_path="synthetic")
    ds.set_metadata(
        {
            "parent": a.parent_id,
            "arm": a.arm,
            "synthetic_train_tiles": len(imgs),
            "synthetic_films": make["films"],
            "per_class_insertions": make["per_class"],
            "per_class_tiles": selection["per_class"],
            "budget": {r["class"]: r["synthetic_instances"] for r in budget["classes"]},
            "tiling": tiling.get("params"),
            "render": {
                k: render[k] for k in ("p_lo", "p_hi", "clahe_clip", "clahe_grid", "three_channel")
            },
            "val": "identical to parent; film split puts every synthetic tile in train",
        },
        metadata_name="pipeline",
    )
    ds.upload(show_progress=True)
    ds.finalize()
    n_child = Counter(p.split("/")[1] for p in ds.list_files() if p.startswith("images/"))
    print(f"[merge] child dataset {ds.id}: {dict(n_child)} (parent {dict(n_parent)})")
    assert n_child["val"] == n_parent["val"], "val changed: must not happen"
    assert n_child["train"] == n_parent["train"] + len(imgs), "train count mismatch"
    print(f"[merge] dataset id {ds.id}  -> use with 05_train.py --dataset-id {ds.id}")


if __name__ == "__main__":
    main()
