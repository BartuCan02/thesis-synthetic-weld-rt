"""Step 13 — score a COCO predictions file against a COCO ground-truth file: the one scorer for every model.

Ultralytics (YOLO, RT-DETR), DEIM and RF-DETR each report mAP with their own evaluator (Ultralytics interpolates the
PR curve differently from pycocotools). The reference-model comparison therefore scores all of them here, with
pycocotools, on the same ground truth (11_export_coco.py) and the same prediction format:

  predictions.json  [{"image_id": <int id or tile stem>, "category_id": <0..5>, "bbox": [x, y, w, h], "score": s}, ...]

image_id may be the tile stem (12_predict_coco.py writes stems); it is mapped to the ground-truth id here.
--pred-cat-base 1 shifts 1-based category ids (Ultralytics' own save_json uses 1..nc for custom datasets).

Reported: mAP50, mAP50-95, AR@100, per-class AP50 / AP50-95 / AR50@100, counts. Optionally logged to a ClearML
testing task (--clearml-name), like 06_eval.py.

  uv run python scripts/13_score_coco.py --gt ~/swrd_paper_baseline/data/coco_v1.0/coco/instances_test.json \
      --pred runs/bm1-real-rtdetr-l-352-50ep/preds_test_best.json --out runs/.../score_test_best.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

PROJECT = "thesis_wp1_benchmark"
OUTPUT_URI = "s3://clearml-data-deeplify/clearml-artifacts"


def load_predictions(path: Path, gt: dict, cat_base: int = 0) -> list[dict]:
    """Predictions with integer ground-truth image ids and 0-based category ids.

    Predictions for images that are not in the ground truth are dropped (and counted by the caller).
    """
    stem_to_id = {img.get("stem", Path(img["file_name"]).stem): img["id"] for img in gt["images"]}
    gt_ids = {img["id"] for img in gt["images"]}
    out = []
    for p in json.loads(Path(path).read_text()):
        image_id = p["image_id"]
        if isinstance(image_id, str):
            if image_id not in stem_to_id:
                continue
            image_id = stem_to_id[image_id]
        elif image_id not in gt_ids:
            continue
        out.append(
            {
                "image_id": int(image_id),
                "category_id": int(p["category_id"]) - cat_base,
                "bbox": [float(v) for v in p["bbox"]],
                "score": float(p["score"]),
            }
        )
    return out


def _per_class(eval_: dict, cat_ids: list[int], names: dict[int, str]) -> dict:
    """AP50, AP50-95 and AR50@100 per class from pycocotools' accumulated precision / recall arrays.

    precision: [T iou thresholds, R recall points, K classes, A area ranges, M max-det]; -1 marks undefined entries.
    """
    precision = eval_["precision"]  # T x R x K x A x M
    recall = eval_["recall"]  # T x K x A x M
    ap50, ap, ar50 = {}, {}, {}
    for k, cat in enumerate(cat_ids):
        name = names.get(cat, str(cat))
        p50 = precision[0, :, k, 0, -1]
        p50 = p50[p50 > -1]
        pall = precision[:, :, k, 0, -1]
        pall = pall[pall > -1]
        r50 = recall[0, k, 0, -1]
        ap50[name] = float(p50.mean()) if p50.size else 0.0
        ap[name] = float(pall.mean()) if pall.size else 0.0
        ar50[name] = float(r50) if r50 > -1 else 0.0
    return {"per_class_AP50": ap50, "per_class_AP50-95": ap, "per_class_AR50@100": ar50}


def score(gt_path: Path, pred_path: Path, cat_base: int = 0) -> dict:
    """COCO box metrics of ``pred_path`` against ``gt_path`` (both files on disk)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = json.loads(Path(gt_path).read_text())
    names = {c["id"]: c["name"] for c in gt["categories"]}
    preds = load_predictions(pred_path, gt, cat_base)
    n_raw = len(json.loads(Path(pred_path).read_text()))
    out = {
        "gt": str(gt_path),
        "pred": str(pred_path),
        "images": len(gt["images"]),
        "gt_boxes": len(gt["annotations"]),
        "predictions": len(preds),
        "predictions_dropped_unknown_image": n_raw - len(preds),
    }
    zero = {n: 0.0 for n in names.values()}
    if not preds:  # pycocotools cannot load an empty result list
        return out | {
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "AR@100": 0.0,
            "per_class_AP50": zero,
            "per_class_AP50-95": dict(zero),
            "per_class_AR50@100": dict(zero),
        }
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints its own table
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = ev.stats.tolist()
    out |= {
        "mAP50-95": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "AR@100": float(stats[8]),
    }
    out |= _per_class(ev.eval, list(ev.params.catIds), names)
    return out


def report_to_clearml(task, result: dict, prefix: str = "") -> None:
    """Single values on a ClearML task, named like 05_train.py's ("mAP50", "AP50/<class>")."""
    log = task.get_logger()
    for key in ("mAP50", "mAP50-95", "AR@100"):
        log.report_single_value(f"{prefix}{key}", result[key])
    for cname, v in result["per_class_AP50"].items():
        log.report_single_value(f"{prefix}AP50/{cname}", v)
    for cname, v in result["per_class_AP50-95"].items():
        log.report_single_value(f"{prefix}AP50-95/{cname}", v)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--gt", type=Path, required=True, help="instances_<part>.json from 11_export_coco.py"
    )
    ap.add_argument("--pred", type=Path, required=True, help="COCO results json")
    ap.add_argument("--pred-cat-base", type=int, default=0, choices=[0, 1])
    ap.add_argument("--out", type=Path, help="write the result json here (default: next to --pred)")
    ap.add_argument("--clearml-name", help="also log to a ClearML testing task with this name")
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    result = score(args.gt, args.pred, args.pred_cat_base)
    out = args.out or args.pred.with_name(f"score_{args.pred.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if args.clearml_name:
        from clearml import Task

        task = Task.init(
            project_name=args.project,
            task_name=args.clearml_name,
            task_type=Task.TaskTypes.testing,
            output_uri=OUTPUT_URI,
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
        )
        task.connect({k: str(v) for k, v in vars(args).items()}, name="score")
        report_to_clearml(task, result)
        task.upload_artifact("score", artifact_object=out)
        task.close()


if __name__ == "__main__":
    main()
