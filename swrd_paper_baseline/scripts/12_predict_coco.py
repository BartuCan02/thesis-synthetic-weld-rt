"""Step 12 — run Ultralytics weights (YOLO or RT-DETR) over the val or test tiles and write COCO predictions.

The reference-model comparison scores every model with 13_score_coco.py on the same ground truth. This script
produces that file for the Ultralytics arms (bm1 YOLOv8m runs, the RT-DETR arm). DEIM and RF-DETR write theirs
inside their own training wrappers (14_train_deim.py, 15_train_rfdetr.py).

Weights: a local .pt, or the best.pt / last.pt artifact of a 05_train.py task (--task-id). The model class is read
from the checkpoint: an RTDETRDetectionModel is run through RTDETR (score cut, no NMS), anything else through YOLO
(NMS at --iou). Low thresholds (--conf 0.001, --max-det 300) so that the PR curve is complete.

Output: <out-dir>/<name>/preds_<part>.json with image_id = tile stem and category_id = YOLO class index (0-based),
plus score_<part>.json when --gt is given (13_score_coco.py on the spot).

  uv run python scripts/12_predict_coco.py --data-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid \
      --split-file ~/swrd_paper_baseline/data/work/split_films_test.json --part test \
      --task-id b8d70cdb8fa54a72bd91a2ffd931ed1d --checkpoint best --imgsz 352 \
      --gt ~/swrd_paper_baseline/data/coco_v1.0/coco/instances_test.json --name coco-bm1-real-yolov8m-best
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def load_sibling(name: str):
    """Import a sibling script by file name (the scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tile_paths(root: Path, split: dict, part: str) -> list[Path]:
    """The tiles of one split part, wherever they sit in the v1.0 folders."""
    ids = set(split[part])
    paths = sorted(
        p
        for folder in ("train", "val")
        for p in (root / "images" / folder).glob("*.png")
        if p.stem in ids
    )
    if len(paths) != len(ids):
        raise SystemExit(
            f"{len(ids) - len(paths)} {part} tiles of the split are missing under {root}"
        )
    return paths


def model_for_weights(weights: str):
    """RTDETR for an RT-DETR checkpoint, YOLO otherwise."""
    from ultralytics import RTDETR, YOLO
    from ultralytics.nn.tasks import attempt_load_one_weight

    net, _ = attempt_load_one_weight(weights)
    if type(net).__name__ == "RTDETRDetectionModel":
        return RTDETR(weights), "rtdetr"
    return YOLO(weights), "yolo"


def results_to_coco(result, stem: str) -> list[dict]:
    """One Ultralytics Results object -> COCO result rows (boxes in original tile pixels)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    rows = []
    for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
        rows.append(
            {
                "image_id": stem,
                "category_id": int(c),
                "bbox": [
                    round(float(x1), 3),
                    round(float(y1), 3),
                    round(float(x2 - x1), 3),
                    round(float(y2 - y1), 3),
                ],
                "score": round(float(s), 5),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, required=True, help="local v1.0 tile folder")
    ap.add_argument("--split-file", type=Path, required=True)
    ap.add_argument("--part", choices=["val", "test"], default="test")
    w = ap.add_mutually_exclusive_group(required=True)
    w.add_argument("--weights", type=Path, help="local .pt")
    w.add_argument("--task-id", help="05_train.py task whose checkpoint artifact to run")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="best")
    ap.add_argument("--imgsz", type=int, default=352)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU for YOLO; ignored by RT-DETR")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument(
        "--gt", type=Path, help="instances_<part>.json: score the predictions right away"
    )
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()

    if args.weights:
        weights = str(args.weights)
    else:
        from clearml import Task

        weights = (
            Task.get_task(task_id=args.task_id).artifacts[f"{args.checkpoint}.pt"].get_local_copy()
        )
    split = json.loads(args.split_file.read_text())
    paths = tile_paths(args.data_dir.resolve(), split, args.part)
    model, arch = model_for_weights(weights)
    print(
        f"[predict] {arch} weights {weights}; {len(paths):,} {args.part} tiles at {args.imgsz} px"
    )

    rows: list[dict] = []
    for i in range(0, len(paths), args.batch):
        chunk = [str(p) for p in paths[i : i + args.batch]]
        results = model.predict(
            source=chunk,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
        for p, r in zip(paths[i : i + args.batch], results):
            rows.extend(results_to_coco(r, p.stem))
    out_dir = args.out_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"preds_{args.part}.json"
    pred_path.write_text(json.dumps(rows))
    print(f"[predict] {len(rows):,} boxes -> {pred_path}")
    (out_dir / "predict_args.json").write_text(
        json.dumps(
            {k: str(v) for k, v in vars(args).items()} | {"weights": weights, "arch": arch},
            indent=2,
        )
    )
    if args.gt:
        scorer = load_sibling("13_score_coco")
        result = scorer.score(args.gt, pred_path)
        (out_dir / f"score_{args.part}.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps({k: v for k, v in result.items() if not k.startswith("per_class")}, indent=2)
        )
        print(json.dumps(result["per_class_AP50"], indent=2))


if __name__ == "__main__":
    main()
