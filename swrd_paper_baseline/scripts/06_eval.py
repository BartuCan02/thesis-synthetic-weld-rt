"""Step 5 — validate saved YOLO weights on the tile set and report mAP and per-class AP.

Use it (a) to recover the result of a multi-GPU run that hung after saving its weights, (b) to score any
weights on the paper's tile-level val split, and (c) with --split-file to score the same tiles under the
film-level split (step 5 of the plan: how much does the tile-level split inflate the number?).

Weights come from a ClearML model id (the output model Ultralytics registers) or a local .pt path.
Data comes from a ClearML dataset id or a local YOLO folder. Results go to a ClearML task and a JSON file.

Run:
  uv run python scripts/06_eval.py --dataset-id de772ad9363c4067bed5835e13a9be81 --model-id <id> --name eval-v1.0-yolov8m
  uv run python scripts/06_eval.py --data-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid --weights runs/x/weights/best.pt --name eval-local
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PROJECT = "thesis_wp1_benchmark"
OUTPUT_URI = "s3://clearml-data-deeplify/clearml-artifacts"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-id")
    src.add_argument("--data-dir", type=Path)
    w = ap.add_mutually_exclusive_group(required=True)
    w.add_argument("--model-id", help="ClearML model id")
    w.add_argument("--weights", type=Path, help="local .pt file")
    ap.add_argument(
        "--split-file",
        type=Path,
        help="optional split json (train/val tile ids) to re-split the val set",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()

    from clearml import Dataset, InputModel, Task

    task = Task.init(
        project_name=PROJECT,
        task_name=args.name,
        task_type=Task.TaskTypes.testing,
        output_uri=OUTPUT_URI,
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch": False, "matplotlib": True, "tensorboard": False},
    )
    task.connect(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, name="eval"
    )

    root = (
        args.data_dir.resolve()
        if args.data_dir
        else Path(Dataset.get(dataset_id=args.dataset_id).get_local_copy())
    )
    weights = (
        str(args.weights) if args.weights else InputModel(model_id=args.model_id).get_local_copy()
    )
    data = yaml.safe_load((root / "swrd6.yaml").read_text())
    data["path"] = str(root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_file:
        # the images stay where they are; Ultralytics accepts a txt list of image paths as the val set
        split = json.loads(args.split_file.read_text())
        val_ids = set(split["val"])
        paths = [
            p
            for part in ("train", "val")
            for p in (root / "images" / part).glob("*.png")
            if p.stem in val_ids
        ]
        lst = args.out_dir / f"{args.name}_val.txt"
        lst.write_text("\n".join(str(p) for p in sorted(paths)) + "\n")
        data["val"] = str(lst)
        print(f"[eval] val set from {args.split_file.name}: {len(paths):,} tiles")
    data_yaml = args.out_dir / f"{args.name}_data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    from ultralytics import YOLO

    model = YOLO(weights)
    res = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=int(args.device),
        project=str(args.out_dir),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    names = res.names
    idx = res.box.ap_class_index.tolist()
    out = {
        "weights": weights,
        "model_id": args.model_id,
        "dataset": args.dataset_id or str(root),
        "split_file": str(args.split_file)
        if args.split_file
        else "tile-level (dataset val folder)",
        "n_val_images": int(getattr(res, "nt_per_image", [0]).sum())
        if hasattr(res, "nt_per_image")
        else None,
        "mAP50": float(res.box.map50),
        "mAP50-95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "per_class_AP50": {names[i]: float(v) for i, v in zip(idx, res.box.ap50.tolist())},
        "per_class_AP50-95": {names[i]: float(v) for i, v in zip(idx, res.box.ap.tolist())},
    }
    logger = task.get_logger()
    logger.report_single_value("mAP50", out["mAP50"])
    logger.report_single_value("mAP50-95", out["mAP50-95"])
    for cname, v in out["per_class_AP50"].items():
        logger.report_single_value(f"AP50/{cname}", v)
    for cname, v in out["per_class_AP50-95"].items():
        logger.report_single_value(f"AP50-95/{cname}", v)
    summary = args.out_dir / args.name / "eval_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(out, indent=2))
    task.upload_artifact("eval_summary", artifact_object=summary)
    print(json.dumps(out, indent=2))
    task.close()


if __name__ == "__main__":
    main()
