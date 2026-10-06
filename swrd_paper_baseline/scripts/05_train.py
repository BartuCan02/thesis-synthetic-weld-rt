"""Step 4 — train the paper's baseline: Ultralytics YOLOv8 detect on the tile set (Sect. 4.2, Table 4).

What the paper fixes:          YOLOv8 n/s/m/l/x, COCO-pretrained weights, 100 epochs, batch 480/480/480/320/240,
                               "all unspecified parameters default" -> image size 640, Ultralytics' default
                               augmentation (mosaic, HSV, flips, scale, translate), optimizer "auto".
What batch size really does:   Ultralytics steps the optimizer once per ``nbs`` images (default 64), accumulating
                               smaller batches, and scales weight decay by batch x accumulate / nbs. The paper's
                               batch 480 therefore meant ~300 optimizer steps per epoch and weight decay x 7.5.
                               ``--emulate-paper-batch`` reproduces exactly that on a T4: nbs = the paper's batch,
                               weight decay = 0.0005 x paper_batch / 64, per-GPU batch a divisor of it (accumulate
                               = paper_batch / batch). Only BatchNorm statistics still see the smaller batch.
What we add:                   ClearML task (scalars, per-class AP, best.pt as output model), a results JSON.

Data: a ClearML Dataset id (downloaded on whichever machine runs this; the data yaml path is rewritten to the
local copy) or a local folder with ``swrd6.yaml``. Remote execution: ``--queue multi-gpu`` enqueues the task
and exits; the agent runs the same script. The multi-gpu agent has 4x T4 and 48 vCPU; the 4 vCPUs of the data
box are too few for Ultralytics' CPU-side augmentation at 144k images/epoch.

Run (examples):
  uv run python scripts/05_train.py --data-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid --model yolov8n --epochs 1 --fraction 0.02 --name smoke
  uv run python scripts/05_train.py --dataset-id <clearml id> --model yolov8n --queue multi-gpu --devices 0 --name v1.0-yolov8n
  uv run python scripts/05_train.py --dataset-id <clearml id> --model yolov8m --queue multi-gpu --devices 0,1,2,3 --name v1.0-yolov8m
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

PROJECT = "thesis_wp1_benchmark"
OUTPUT_URI = "s3://clearml-data-deeplify/clearml-artifacts"
DOCKER_IMAGE = "862264091922.dkr.ecr.eu-central-1.amazonaws.com/model-training:latest"
DOCKER_ARGS = "--gpus all --network host --shm-size=16g -e AWS_PROFILE=clearml-s3 -v /home/ec2-user/.aws:/root/.aws:ro"
PAPER_BATCH = {"yolov8n": 480, "yolov8s": 480, "yolov8m": 480, "yolov8l": 320, "yolov8x": 240}
ULTRALYTICS_NBS = (
    64  # nominal batch size: optimizer steps every nbs images; weight decay scales by batch/nbs
)
ULTRALYTICS_WD = 0.0005
# a per-GPU batch that divides the paper's batch and fits a 16 GB T4 at 640 px
EMULATE_BATCH = {
    "yolov8n": 60,
    "yolov8s": 48,
    "yolov8m": 24,
    "yolov8l": 20,
    "yolov8x": 16,
}  # x8 / x10 / x20 / x16 / x15 = paper batch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-id", help="ClearML Dataset id (output of 04_upload_clearml.py)")
    src.add_argument(
        "--data-dir", type=Path, help="local YOLO folder with swrd6.yaml (output of 03_render.py)"
    )
    ap.add_argument("--model", default="yolov8n", choices=list(PAPER_BATCH))
    ap.add_argument("--epochs", type=int, default=100, help="paper: 100")
    ap.add_argument(
        "--imgsz", type=int, default=640, help="Ultralytics default; paper: unspecified -> default"
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="-1 = AutoBatch (largest that fits ~60 %% of GPU memory); paper: 480",
    )
    ap.add_argument("--devices", default="0", help="GPU ids, e.g. '0' or '0,1,2,3' (DDP)")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers per GPU")
    ap.add_argument(
        "--fraction", type=float, default=1.0, help="fraction of the train set (smoke tests)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--emulate-paper-batch",
        action="store_true",
        help="nbs = paper batch, weight decay scaled like the paper, batch = EMULATE_BATCH unless --batch given",
    )
    ap.add_argument(
        "--nbs",
        type=int,
        default=None,
        help="override nominal batch size (optimizer step interval)",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="override weight decay (Ultralytics default 0.0005)",
    )
    ap.add_argument(
        "--name", required=True, help="run name (ClearML task name and local run folder)"
    )
    ap.add_argument(
        "--queue", default="", help="ClearML queue for remote execution; empty = run here"
    )
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return ap.parse_args()


def resolve_data_yaml(args: argparse.Namespace) -> Path:
    """Return a data yaml whose paths point at a local copy of the tiles."""
    if args.data_dir:
        root = args.data_dir.resolve()
    else:
        from clearml import Dataset

        root = Path(Dataset.get(dataset_id=args.dataset_id).get_local_copy())
    src = root / "swrd6.yaml"
    data = yaml.safe_load(src.read_text())
    data["path"] = str(root)
    out = args.runs_dir / f"{args.name}_data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return out


def main() -> None:
    from clearml import Task

    # Task.init BEFORE parse_args: ClearML patches argparse so that, on the agent, the arguments stored
    # in the task's "Args" section are injected (the agent passes no command line of its own).
    task = Task.init(
        project_name=PROJECT,
        task_name="swrd-paper-baseline (naming…)",
        task_type=Task.TaskTypes.training,
        output_uri=OUTPUT_URI,
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch": False, "matplotlib": True, "tensorboard": False},
    )
    args = parse_args()
    task.set_name(args.name)
    task.set_base_docker(DOCKER_IMAGE, docker_arguments=DOCKER_ARGS)
    # The image (probed 2026-10-03) has torch 2.9.1+cu128, opencv 5.0, numpy 2.5 but no ultralytics; pin only
    # what is missing so pip does not pull a second torch.
    task.set_packages(["ultralytics==8.4.171", "clearml==2.1.12", "pyyaml==6.0.3"])
    task.set_user_properties(
        paper_batch=PAPER_BATCH[args.model], paper_epochs=100, paper="Zhao 2025 Table 4"
    )
    if args.queue:
        task.execute_remotely(queue_name=args.queue)  # everything below runs on the agent

    from ultralytics import YOLO, settings

    # keep only the ClearML integration; the key set differs between Ultralytics versions
    wanted = {
        "clearml": True,
        "wandb": False,
        "comet": False,
        "mlflow": False,
        "tensorboard": False,
        "dvc": False,
        "neptune": False,
        "raytune": False,
        "hub": False,
    }
    settings.update({k: v for k, v in wanted.items() if k in settings})
    data_yaml = resolve_data_yaml(args)
    devices = [int(d) for d in str(args.devices).split(",")]

    nbs = args.nbs or ULTRALYTICS_NBS
    weight_decay = args.weight_decay if args.weight_decay is not None else ULTRALYTICS_WD
    batch = args.batch
    if args.emulate_paper_batch:
        paper = PAPER_BATCH[args.model]
        nbs = args.nbs or paper
        weight_decay = (
            args.weight_decay
            if args.weight_decay is not None
            else ULTRALYTICS_WD * paper / ULTRALYTICS_NBS
        )
        if batch == -1:
            batch = EMULATE_BATCH[args.model]
    accumulate = max(round(nbs / batch), 1) if batch > 0 else None
    n_train = sum(
        1
        for _ in (Path(yaml.safe_load(data_yaml.read_text())["path"]) / "images" / "train").glob(
            "*.png"
        )
    )
    task.set_user_properties(
        nbs=nbs,
        batch=batch,
        accumulate=accumulate,
        effective_batch=(batch * accumulate) if accumulate else None,
        weight_decay=weight_decay,
        optimizer_steps_per_epoch=round(n_train * args.fraction / (batch * accumulate))
        if accumulate
        else None,
        paper_steps_per_epoch=round((72585 + 72646) / PAPER_BATCH[args.model]),  # Table 3 train set
    )
    print(
        f"[train] batch {batch} x accumulate {accumulate} = effective {batch * accumulate if accumulate else '?'}; "
        f"nbs {nbs}; weight_decay {weight_decay:.5f}; paper: batch {PAPER_BATCH[args.model]}"
    )

    model = YOLO(f"{args.model}.pt")  # COCO-pretrained, downloaded by Ultralytics on first use
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        nbs=nbs,
        weight_decay=weight_decay,
        device=devices if len(devices) > 1 else devices[0],
        workers=args.workers,
        fraction=args.fraction,
        seed=args.seed,
        project=str(args.runs_dir),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        plots=True,
        verbose=True,
    )

    # ---- summary: best epoch metrics, per-class AP, where the weights are -----------------------------
    # Under DDP, Ultralytics returns a save_dir relative to the agent's cwd (runs/detect/<runs_dir>/<name>);
    # resolve it from the trainer, fall back to a search, and never let the summary step fail the task.
    run_dir = None
    for cand in (
        getattr(getattr(model, "trainer", None), "save_dir", None),
        getattr(results, "save_dir", None) if results is not None else None,
        args.runs_dir / args.name,
        Path("runs") / "detect" / args.runs_dir / args.name,
    ):
        if cand and (Path(cand) / "weights").is_dir():
            run_dir = Path(cand)
            break
    if run_dir is None:
        hits = sorted(Path(".").glob(f"**/{args.name}/weights/best.pt"))
        run_dir = hits[0].parents[1] if hits else args.runs_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run_dir resolved to {run_dir}")
    summary = {
        "run_dir": str(run_dir),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    csv_path = run_dir / "results.csv"
    if csv_path.is_file():
        import csv

        with open(csv_path) as fh:
            raw_rows = list(csv.DictReader(fh))
        rows = []
        for r in raw_rows:
            row = {}
            for k, v in r.items():
                try:
                    row[k.strip()] = float(v)
                except (TypeError, ValueError):
                    row[k.strip()] = v
            rows.append(row)
        key50, key5095 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        best50 = max(rows, key=lambda r: r.get(key50, 0))
        best5095 = max(rows, key=lambda r: r.get(key5095, 0))
        summary["best_mAP50"] = {"value": best50[key50], "epoch": int(best50["epoch"])}
        summary["best_mAP50-95"] = {"value": best5095[key5095], "epoch": int(best5095["epoch"])}
        summary["last"] = {key50: rows[-1][key50], key5095: rows[-1][key5095]}
    try:
        val = YOLO(str(run_dir / "weights" / "best.pt")).val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=16,
            device=devices[0],
            plots=False,
            verbose=False,
        )
        names = val.names
        summary["best_pt_val"] = {
            "mAP50": float(val.box.map50),
            "mAP50-95": float(val.box.map),
            "per_class_AP50": {
                names[i]: float(v)
                for i, v in zip(val.box.ap_class_index.tolist(), val.box.ap50.tolist())
            },
            "per_class_AP50-95": {
                names[i]: float(v)
                for i, v in zip(val.box.ap_class_index.tolist(), val.box.maps.tolist())
                if i in names
            },
        }
        for cname, v in summary["best_pt_val"]["per_class_AP50"].items():
            task.get_logger().report_single_value(f"AP50/{cname}", v)
        task.get_logger().report_single_value("best.pt mAP50", summary["best_pt_val"]["mAP50"])
        task.get_logger().report_single_value(
            "best.pt mAP50-95", summary["best_pt_val"]["mAP50-95"]
        )
    except Exception as e:  # noqa: BLE001 — the summary must never kill a finished run
        summary["best_pt_val_error"] = repr(e)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    task.upload_artifact("summary", artifact_object=run_dir / "summary.json")
    for w in ("best.pt", "last.pt"):
        p = run_dir / "weights" / w
        if p.is_file():
            task.upload_artifact(w, artifact_object=p)
    print(json.dumps(summary, indent=2))
    task.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
