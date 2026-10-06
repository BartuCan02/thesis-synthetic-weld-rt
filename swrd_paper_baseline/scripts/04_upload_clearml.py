"""Step 3c — register the rendered tile set as a ClearML Dataset.

Creates (or reuses) the ClearML project given by --project and uploads the YOLO folder produced by
03_render.py. Every parameter of the pipeline (tiling, D7 box rule, D5/D6 preprocessing, seed, split
counts) is attached as dataset metadata so a training task can be traced back to its exact recipe.

Storage goes to S3, not the ClearML file server: datasets on the file server are unreachable from the
training agents (memory: clearml-gotchas). Run on the box with AWS_PROFILE=data-rw.

Run:
  AWS_PROFILE=data-rw uv run python scripts/04_upload_clearml.py --yolo-dir ~/swrd_paper_baseline/data/yolo_tilesplit \
      --work-dir ~/swrd_paper_baseline/data/work --project thesis_wp1_benchmark --name swrd-paper-tiles --version 1.0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearml import Dataset

DEFAULT_OUTPUT_URI = "s3://clearml-data-deeplify/clearml-datasets"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--yolo-dir", type=Path, required=True, help="output of 03_render.py")
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="holds inventory_report.md, tiles_summary.json, split_*.json",
    )
    ap.add_argument("--project", default="thesis_benchmark", help="ClearML project (the 'folder')")
    ap.add_argument("--name", default="swrd-paper-tiles")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI)
    ap.add_argument(
        "--description",
        default="SWRD paper-baseline tiles (Zhao et al. 2025, Sect. 3.2-3.3, Table 3 split)",
    )
    args = ap.parse_args()

    params_path = args.yolo_dir / "render_params.json"
    render_params = json.loads(params_path.read_text(encoding="utf-8"))
    tiles_summary = json.loads((args.work_dir / "tiles_summary.json").read_text(encoding="utf-8"))

    n_img = {p: len(list((args.yolo_dir / "images" / p).glob("*.png"))) for p in ("train", "val")}
    n_lab = {p: len(list((args.yolo_dir / "labels" / p).glob("*.txt"))) for p in ("train", "val")}
    assert n_img == n_lab, f"image/label count mismatch: {n_img} vs {n_lab}"
    print(f"[upload] {n_img} tiles in {args.yolo_dir}")

    ds = Dataset.create(
        dataset_name=args.name,
        dataset_project=args.project,
        dataset_version=args.version,
        output_uri=args.output_uri,
        description=args.description,
    )
    ds.add_files(path=str(args.yolo_dir / "images"), dataset_path="images", verbose=False)
    ds.add_files(path=str(args.yolo_dir / "labels"), dataset_path="labels", verbose=False)
    ds.add_files(path=str(args.yolo_dir / "swrd6.yaml"))
    ds.add_files(path=str(args.yolo_dir / "render_params.json"))
    for extra in (
        "inventory_report.md",
        "tiles_summary.json",
        "split_tiles.json",
        "split_films.json",
    ):
        p = args.work_dir / extra
        if p.is_file():
            ds.add_files(path=str(p), dataset_path="pipeline")

    ds.set_metadata(
        {
            "paper": "Zhao et al. 2025, J. Nondestructive Evaluation 44:50, doi:10.1007/s10921-025-01186-w",
            "tiling": tiles_summary["params"],
            "tile_counts": {
                k: tiles_summary[k]
                for k in ("n_images", "n_tiles", "n_tiles_with_box", "n_clean_tiles")
            },
            "instances_per_class_on_tiles": tiles_summary["instances_per_class"],
            "render": {
                k: render_params[k]
                for k in ("p_lo", "p_hi", "clahe_clip", "clahe_grid", "three_channel", "split")
            },
            "split_counts": render_params.get("split_counts"),
            "split_params": render_params.get("split_params"),
            "n_files": {"images": n_img, "labels": n_lab},
        },
        metadata_name="pipeline",
    )
    ds.upload(show_progress=True)
    ds.finalize()
    print(f"[upload] dataset id {ds.id}  ({args.project} / {args.name} {args.version})")
    print("[upload] paste the id into experiments/wp1_benchmark/README.md")


if __name__ == "__main__":
    main()
