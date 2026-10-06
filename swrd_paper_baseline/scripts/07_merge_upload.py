"""Register "SWRD v1.0 tiles + customer train tiles" as a ClearML Dataset that inherits from v1.0.

The child dataset keeps every file of the parent (train and val tiles, swrd6.yaml) and adds the customer
tiles to ``images/train`` and ``labels/train`` only. The val folder is byte-identical to the parent's, so a
model trained on the child is scored on exactly the SWRD val tiles the baseline was scored on.

Refuses to run if the customer folder has a val split or a tile id that already exists in the parent.

Run on the box (AWS_PROFILE=data-rw; datasets must live on S3, see 04_upload_clearml.py):
  AWS_PROFILE=data-rw uv run python scripts/07_merge_upload.py --parent-id de772ad9363c4067bed5835e13a9be81 \\
      --customer-yolo-dir ~/swrd_paper_baseline/data/yolo_customer_v1 --customer-work-dir ~/swrd_paper_baseline/data/work_customer \\
      --raw-customer-dir ~/swrd_paper_baseline/data/raw_customer --name swrd-paper-tiles-plus-customer --version 1.0.0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from clearml import Dataset

PROJECT = "thesis_wp1_benchmark"
DEFAULT_OUTPUT_URI = "s3://clearml-data-deeplify/clearml-datasets"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parent-id", required=True, help="the SWRD v1.0 tile dataset")
    ap.add_argument("--customer-yolo-dir", type=Path, required=True, help="output of 03_render.py")
    ap.add_argument(
        "--customer-work-dir", type=Path, required=True, help="tiles_summary.json, split_tiles.json"
    )
    ap.add_argument("--raw-customer-dir", type=Path, required=True, help="holds export_report.json")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--name", default="swrd-paper-tiles-plus-customer")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI)
    ap.add_argument("--dry-run", action="store_true", help="run every check, create nothing")
    args = ap.parse_args()

    img_dir = args.customer_yolo_dir / "images" / "train"
    lab_dir = args.customer_yolo_dir / "labels" / "train"
    imgs = sorted(p.name for p in img_dir.glob("*.png"))
    labs = sorted(p.name for p in lab_dir.glob("*.txt"))
    assert imgs, f"no customer train tiles under {img_dir}"
    assert [Path(i).stem for i in imgs] == [Path(l).stem for l in labs], "image/label mismatch"
    val_dir = args.customer_yolo_dir / "images" / "val"
    assert not val_dir.exists() or not any(val_dir.iterdir()), (
        "customer folder has val tiles; 02_select_split.py must run with --val-ratio 0"
    )

    parent = Dataset.get(dataset_id=args.parent_id)
    parent_files = set(parent.list_files())
    clashes = [
        i for i in imgs if f"images/train/{i}" in parent_files or f"images/val/{i}" in parent_files
    ]
    assert not clashes, (
        f"{len(clashes)} customer tile ids already exist in the parent, e.g. {clashes[:3]}"
    )
    n_parent = Counter(p.split("/")[1] for p in parent_files if p.startswith("images/"))
    print(
        f"[merge] parent {parent.name} {parent.version}: {dict(n_parent)}; adding {len(imgs):,} customer train tiles"
    )

    per_source = Counter(i.split("__", 1)[0] for i in imgs)
    n_pos = sum(1 for l in labs if (lab_dir / l).stat().st_size > 0)
    tiles_summary = json.loads((args.customer_work_dir / "tiles_summary.json").read_text())
    split = json.loads((args.customer_work_dir / "split_tiles.json").read_text())
    export_report = json.loads((args.raw_customer_dir / "export_report.json").read_text())
    render_params = json.loads((args.customer_yolo_dir / "render_params.json").read_text())
    if args.dry_run:
        print(
            f"[merge] dry run OK: {len(imgs):,} tiles ({n_pos:,} with a box) from {dict(per_source)}; "
            f"no clash with the parent; nothing created"
        )
        return

    ds = Dataset.create(
        dataset_name=args.name,
        dataset_project=args.project,
        dataset_version=args.version,
        parent_datasets=[args.parent_id],
        output_uri=args.output_uri,
        description=(
            "SWRD paper-baseline tiles v1.0 (parent, unchanged incl. val) + Deeplify customer films "
            "(oge/aramco/maroca) tiled with the same pipeline, train split only. Felix's add-our-data question."
        ),
    )
    ds.add_files(path=str(img_dir), dataset_path="images/train", verbose=False)
    ds.add_files(path=str(lab_dir), dataset_path="labels/train", verbose=False)
    ds.add_files(path=str(args.customer_yolo_dir / "render_params.json"), dataset_path="customer")
    for extra in ("tiles_summary.json", "split_tiles.json", "inventory_report.md"):
        p = args.customer_work_dir / extra
        if p.is_file():
            ds.add_files(path=str(p), dataset_path="customer")
    ds.add_files(path=str(args.raw_customer_dir / "export_report.json"), dataset_path="customer")

    ds.set_metadata(
        {
            "parent": args.parent_id,
            "customer_train_tiles": len(imgs),
            "customer_train_tiles_with_box": n_pos,
            "customer_tiles_per_source": dict(per_source),
            "customer_films_per_source": {
                s: d["exported"] for s, d in export_report["per_source"].items()
            },
            "customer_export_params": export_report["params"],
            "customer_tiling": tiles_summary["params"],
            "customer_instances_per_class_on_tiles": tiles_summary["instances_per_class"],
            "customer_render": {
                k: render_params[k]
                for k in ("p_lo", "p_hi", "clahe_clip", "clahe_grid", "three_channel")
            },
            "customer_split_params": split.get("params"),
            "val": "identical to parent (SWRD v1.0 tile-level val)",
        },
        metadata_name="pipeline",
    )
    ds.upload(show_progress=True)
    ds.finalize()
    files = ds.list_files()
    n_child = Counter(p.split("/")[1] for p in files if p.startswith("images/"))
    print(f"[merge] child dataset {ds.id}: {dict(n_child)} (parent {dict(n_parent)})")
    assert n_child["val"] == n_parent["val"], "val changed — must not happen"
    assert n_child["train"] == n_parent["train"] + len(imgs), "train count mismatch"
    print(f"[merge] dataset id {ds.id}  -> use with 05_train.py --dataset-id {ds.id}")


if __name__ == "__main__":
    main()
