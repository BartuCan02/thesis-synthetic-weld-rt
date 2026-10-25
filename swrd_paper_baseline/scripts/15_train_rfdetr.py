"""Step 15 — train RF-DETR (Robinson et al., arXiv 2511.09554; Roboflow) on the SWRD tiles as a ClearML task.

Third arm of the reference-model comparison (experiments/wp1_benchmark/reference_models.md). RF-DETR Large (the
2026 Apache-2.0 release: DINOv2-small windowed backbone, LW-DETR-style decoder with 4 layers, 300 queries, 33.9 M
parameters) is fine-tuned through the ``rfdetr`` package, pinned to RFDETR_VERSION. Its venue claim (ICLR 2026)
is Roboflow's and was not verified: treat this arm as the strong-but-unreviewed candidate.

What this wrapper does (standalone file, like 05_train.py: the agent gets nothing else from the repo):
  1. downloads the COCO child dataset (11_export_coco.py --upload) and lays it out the way rfdetr wants it,
     ``rf/train/_annotations.coco.json`` + images and ``rf/valid/...``: symlinks to the parent's PNGs, file_name
     reduced to the basename. No ``test/`` folder: the held-out tiles are never visible to the training run;
  2. runs the training in a child process under torchrun (rfdetr trains with PyTorch Lightning; the ClearML task
     process must not become a Lightning rank), from the published COCO weights, at --imgsz (RF-DETR Large needs a
     multiple of 32, so 352 is allowed) with per-GPU --batch and --grad-accum; model hyperparameters stay the
     package defaults (AdamW, lr 1e-4, encoder lr 1.5e-4, weight decay 1e-4, EMA);
  3. after training, replays rfdetr's TensorBoard scalars into ClearML, predicts the val and test tiles with
     checkpoint_best_total.pth, writes COCO predictions (image_id = ground-truth id, category_id = class index via
     the class names rfdetr stored) and scores them with pycocotools, the same arithmetic as 13_score_coco.py.

Run (box, with AWS_PROFILE=data-rw):
  uv run python scripts/15_train_rfdetr.py --dataset-id <coco child id> --epochs 50 --imgsz 352 --batch 4 \
      --grad-accum 1 --devices 0,1,2,3 --seed 0 --queue multi-gpu --tags benchmark-v1 reference-models \
      --name bm1-real-rfdetr-l-352-50ep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = "thesis_wp1_benchmark"
OUTPUT_URI = "s3://clearml-data-deeplify/clearml-artifacts"
DOCKER_IMAGE = "862264091922.dkr.ecr.eu-central-1.amazonaws.com/model-training:latest"
DOCKER_ARGS = "--gpus all --network host --shm-size=16g -e AWS_PROFILE=clearml-s3 -v /home/ec2-user/.aws:/root/.aws:ro"
RFDETR_VERSION = "1.11.2"  # released 2026-10-04
PACKAGES = [
    f"rfdetr[train]=={RFDETR_VERSION}",
    "tensorboard==2.21.0",
    "pycocotools==2.0.11",
    "clearml==2.1.12",
]
VARIANTS = {"large": "RFDETRLarge", "medium": "RFDETRMedium", "small": "RFDETRSmall"}
BLOCK_SIZE = (
    32  # patch_size 16 x num_windows 2 for the detection variants: imgsz must be a multiple
)


def project_from_argv(default: str = PROJECT) -> str:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--project" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--project="):
            return a.split("=", 1)[1]
    return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset-id",
        required=True,
        help="ClearML COCO child dataset (11_export_coco.py --upload)",
    )
    ap.add_argument("--variant", default="large", choices=sorted(VARIANTS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=352, help=f"multiple of {BLOCK_SIZE}")
    ap.add_argument("--batch", type=int, default=4, help="per-GPU batch (rfdetr default 4)")
    ap.add_argument(
        "--grad-accum", type=int, default=1, help="effective batch = batch x grad-accum x GPUs"
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-interval", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.001, help="score cut for the prediction files")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--queue", default="", help="ClearML queue; empty = run here")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()
    if args.imgsz % BLOCK_SIZE:
        ap.error(f"--imgsz must be a multiple of {BLOCK_SIZE} for RF-DETR (patch 16 x 2 windows)")
    return args


def build_rf_layout(data_root: Path, out: Path) -> Path:
    """rfdetr's COCO layout from a 11_export_coco.py dataset: train/ and valid/ only, images as symlinks.

    file_name becomes the basename because rfdetr reads ``<split>/<file_name>``. The test json is not laid out:
    the held-out tiles stay invisible to the training run and are read from ``data_root`` only for scoring.
    """
    if out.exists():
        shutil.rmtree(out)  # holds only symlinks and two json copies
    for part, folder in (("train", "train"), ("val", "valid")):
        coco = json.loads((data_root / "coco" / f"instances_{part}.json").read_text())
        dest = out / folder
        dest.mkdir(parents=True)
        for im in coco["images"]:
            src = data_root / im["file_name"]
            if not src.is_file():
                raise FileNotFoundError(src)
            name = Path(im["file_name"]).name
            if (dest / name).exists():
                raise FileExistsError(f"duplicate tile name {name} in {part}")
            (dest / name).symlink_to(src)
            im["file_name"] = name
        (dest / "_annotations.coco.json").write_text(json.dumps(coco))
    return out


def worker_main(argv: list[str]) -> None:
    """Training process (one per GPU under torchrun). Reads its arguments from the json the task wrote."""
    spec = json.loads(Path(argv[argv.index("--worker") + 1]).read_text())
    import rfdetr

    model = getattr(rfdetr, spec["model_class"])()
    kwargs = {
        "dataset_dir": spec["dataset_dir"],
        "epochs": spec["epochs"],
        "batch_size": spec["batch"],
        "grad_accum_steps": spec["grad_accum"],
        "lr": spec["lr"],
        "lr_encoder": spec["lr_encoder"],
        "weight_decay": spec["weight_decay"],
        "resolution": spec["imgsz"],
        "output_dir": spec["output_dir"],
        "seed": spec["seed"],
        "num_workers": spec["workers"],
        "tensorboard": True,
        "checkpoint_interval": spec["checkpoint_interval"],
        "early_stopping": False,
    }
    if spec["n_devices"] > 1:
        kwargs["devices"] = spec[
            "n_devices"
        ]  # rfdetr's build_trainer defaults to one device even under torchrun
    model.train(**kwargs)


def tensorboard_scalars(root: Path) -> dict[str, list[tuple[int, float]]]:
    """Every scalar series rfdetr logged, from the event files under ``root``."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    series: dict[str, list[tuple[int, float]]] = {}
    for ev_file in sorted(root.rglob("events.out.tfevents*")):
        acc = EventAccumulator(str(ev_file), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            series.setdefault(tag, []).extend((e.step, float(e.value)) for e in acc.Scalars(tag))
    return series


def coco_score(gt_path: Path, preds: list[dict]) -> dict:
    """pycocotools box metrics with per-class AP50 / AP50-95. Same arithmetic as 13_score_coco.py (standalone copy)."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = json.loads(Path(gt_path).read_text())
    names = {c["id"]: c["name"] for c in gt["categories"]}
    out = {
        "images": len(gt["images"]),
        "gt_boxes": len(gt["annotations"]),
        "predictions": len(preds),
    }
    if not preds:
        zero = {n: 0.0 for n in names.values()}
        return out | {
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "per_class_AP50": zero,
            "per_class_AP50-95": dict(zero),
        }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        ev = COCOeval(coco_gt, coco_gt.loadRes(preds), iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    precision = ev.eval["precision"]
    ap50, ap = {}, {}
    for k, cat in enumerate(ev.params.catIds):
        p50 = precision[0, :, k, 0, -1]
        p50 = p50[p50 > -1]
        pall = precision[:, :, k, 0, -1]
        pall = pall[pall > -1]
        ap50[names[cat]] = float(p50.mean()) if p50.size else 0.0
        ap[names[cat]] = float(pall.mean()) if pall.size else 0.0
    return out | {
        "mAP50-95": float(ev.stats[0]),
        "mAP50": float(ev.stats[1]),
        "AR@100": float(ev.stats[8]),
        "per_class_AP50": ap50,
        "per_class_AP50-95": ap,
    }


def label_to_category(class_names: list[str], gt_categories: list[dict]) -> dict[int, int]:
    """rfdetr label index -> our category id, matched by class name (rfdetr relabels categories contiguously)."""
    by_name = {c["name"]: c["id"] for c in gt_categories}
    missing = [n for n in class_names if n not in by_name]
    if missing:
        raise ValueError(f"rfdetr class names not in the ground truth: {missing}")
    return {i: by_name[n] for i, n in enumerate(class_names)}


def predict_coco(
    model, data_root: Path, gt_path: Path, label_map: dict[int, int], conf: float
) -> list[dict]:
    """COCO predictions of a loaded rfdetr model on every image of a ground-truth json."""
    from PIL import Image

    gt = json.loads(Path(gt_path).read_text())
    rows = []
    for im in gt["images"]:
        det = model.predict(Image.open(data_root / im["file_name"]).convert("RGB"), threshold=conf)
        for (x1, y1, x2, y2), s, c in zip(
            det.xyxy.tolist(), det.confidence.tolist(), det.class_id.tolist()
        ):
            rows.append(
                {
                    "image_id": im["id"],
                    "category_id": label_map[int(c)],
                    "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                    "score": round(float(s), 5),
                }
            )
    return rows


def main() -> None:
    from clearml import Task

    Task.force_store_standalone_script()
    task = Task.init(
        project_name=project_from_argv(),
        task_name="rfdetr (naming…)",
        task_type=Task.TaskTypes.training,
        output_uri=OUTPUT_URI,
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch": False, "matplotlib": False, "tensorboard": False},
    )
    args = parse_args()
    task.set_name(args.name)
    if args.tags:
        task.add_tags(args.tags)
    task.set_base_docker(DOCKER_IMAGE, docker_arguments=DOCKER_ARGS)
    task.set_packages(PACKAGES)
    n_devices = len(args.devices.split(","))
    task.set_user_properties(
        model=f"RF-DETR {args.variant} (rfdetr {RFDETR_VERSION})",
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch_per_gpu=args.batch,
        grad_accum=args.grad_accum,
        effective_batch=args.batch * args.grad_accum * n_devices,
        lr=args.lr,
        task_kind="detect-rfdetr",
    )
    if args.queue:
        task.execute_remotely(queue_name=args.queue)  # everything below runs on the agent

    from clearml import Dataset

    runs_dir = Path(args.runs_dir).resolve()
    out_dir = runs_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(Dataset.get(dataset_id=args.dataset_id).get_local_copy())
    for part in ("train", "val", "test"):
        if not (data_root / "coco" / f"instances_{part}.json").is_file():
            raise FileNotFoundError(
                f"{data_root}/coco/instances_{part}.json: not a 11_export_coco.py dataset"
            )
    rf_dir = build_rf_layout(data_root, runs_dir / f"{args.name}_rf")

    spec = {
        "model_class": VARIANTS[args.variant],
        "dataset_dir": str(rf_dir),
        "epochs": args.epochs,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "imgsz": args.imgsz,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "workers": args.workers,
        "checkpoint_interval": args.checkpoint_interval,
        "n_devices": n_devices,
    }
    spec_path = out_dir / "worker_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    task.connect_configuration(spec, name="rfdetr_train")
    cmd = [sys.executable]
    if n_devices > 1:
        cmd += [
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={n_devices}",
            "--master_port=29518",
        ]
    cmd += [os.path.abspath(__file__), "--worker", str(spec_path)]
    env = os.environ | {"CUDA_VISIBLE_DEVICES": args.devices, "PYTHONUNBUFFERED": "1"}
    print("[rfdetr]", " ".join(cmd))
    subprocess.run(cmd, cwd=out_dir, env=env, check=True)

    # ---- TensorBoard scalars -> ClearML ------------------------------------------------------------------------
    log = task.get_logger()
    series = tensorboard_scalars(out_dir)
    for tag, points in series.items():
        title, _, name = tag.partition("/")
        for step, value in points:
            log.report_scalar(title, name or title, value, int(step))
    print(f"[rfdetr] replayed {len(series)} TensorBoard series into ClearML")

    # ---- predictions and scores on val and test with the best checkpoint ------------------------------------
    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "rfdetr": RFDETR_VERSION,
        "spec": spec,
    }
    best = out_dir / "checkpoint_best_total.pth"
    if not best.is_file():
        raise FileNotFoundError(f"{best}: rfdetr did not write its best checkpoint")
    for name in (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "training_config.json",
    ):
        if (out_dir / name).is_file():
            task.upload_artifact(name, artifact_object=out_dir / name)
    train_cfg = json.loads((out_dir / "training_config.json").read_text())
    gt_val = json.loads((data_root / "coco" / "instances_val.json").read_text())
    label_map = label_to_category(train_cfg["class_names"], gt_val["categories"])
    summary["label_to_category"] = label_map

    from rfdetr import RFDETR

    model = RFDETR.from_checkpoint(str(best))
    for part in ("val", "test"):
        gt_path = data_root / "coco" / f"instances_{part}.json"
        preds = predict_coco(model, data_root, gt_path, label_map, args.conf)
        pred_path = out_dir / f"preds_{part}_best.json"
        pred_path.write_text(json.dumps(preds))
        task.upload_artifact(f"preds_{part}_best", artifact_object=pred_path)
        result = coco_score(gt_path, preds)
        summary[f"{part}_best"] = result
        log.report_single_value(f"{part} best mAP50", result["mAP50"])
        log.report_single_value(f"{part} best mAP50-95", result["mAP50-95"])
        for cname, v in result["per_class_AP50"].items():
            log.report_single_value(f"{part} best AP50/{cname}", v)
        print(
            f"[rfdetr] {part} best: mAP50 {result['mAP50']:.4f} mAP50-95 {result['mAP50-95']:.4f}"
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    task.upload_artifact("summary", artifact_object=out_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    task.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if "--worker" in sys.argv:
        worker_main(sys.argv)  # a torchrun rank: no ClearML task of its own
    else:
        main()
