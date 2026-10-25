"""Step 14 — train DEIM-D-FINE-L (Huang et al., CVPR 2025) on the SWRD tiles as a ClearML task.

Second arm of the reference-model comparison (experiments/wp1_benchmark/reference_models.md). DEIM is a training
recipe (dense one-to-one matching with Mosaic/Mixup, matchability-aware loss) on top of D-FINE (Peng et al., ICLR
2025: HGNetv2-B4 backbone, hybrid encoder, 6 deformable decoder layers, 300 queries, fine-grained distribution
box refinement). Code: github.com/Intellindust-AI-Lab/DEIM, pinned to DEIM_SHA and fetched as a tarball on the agent.

What this wrapper does (the agent gets this file alone, like 05_train.py, so nothing is imported from the repo):
  1. downloads the COCO child dataset (11_export_coco.py --upload): the parent's PNGs plus coco/instances_*.json;
  2. writes a derived DEIM config that includes the repo's deim_hgnetv2_l_coco.yml and overrides only what the
     data demands: 6 classes, no COCO remap, our json files, 352 px everywhere (Resize, Mosaic canvas, multi-scale
     base, eval size), the epoch schedule rescaled to --epochs (see deim_schedule), batch and workers;
  3. runs the repo's train.py under torchrun, fine-tuning from --init-checkpoint (the DEIM-D-FINE-L COCO weights,
     downloaded once by hand from the model zoo's Google Drive link and uploaded here as a task artifact); without
     a checkpoint it trains from the ImageNet HGNetv2-B4 weights, which are pre-downloaded on the main process
     because the repo's own download is not multi-process safe;
  4. parses log.txt into ClearML scalars (val mAP50 / mAP50-95 per epoch), picks best (best_stg2.pth if the no-aug
     phase produced one, else best_stg1.pth) and last.pth, predicts the val and test tiles with each, writes COCO
     predictions (image_id = ground-truth id, category_id = class index) and scores them with pycocotools, the
     same scorer as 13_score_coco.py, so the numbers are comparable with 12_predict_coco.py on the YOLO runs.

DEIM's val set during training is our 273-film val (checkpoint selection); the 96-film test set is read only in
step 4. Model-side hyperparameters (AdamW, lr 5e-4 with backbone 2.5e-5, weight decay 1.25e-4, EMA, AMP, 2000
warm-up iterations, Mosaic/Mixup policy) stay the repo's; --lr scales both learning rates together if needed.

Run (box, with AWS_PROFILE=data-rw; ~/deim_dfine_l_coco.pth = the model-zoo checkpoint):
  uv run python scripts/14_train_deim.py --dataset-id <coco child id> --init-checkpoint ~/deim_dfine_l_coco.pth \
      --epochs 50 --imgsz 352 --total-batch 32 --devices 0,1,2,3 --seed 0 --queue multi-gpu \
      --tags benchmark-v1 reference-models --name bm1-real-deim-dfine-l-352-50ep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml

PROJECT = "thesis_wp1_benchmark"
OUTPUT_URI = "s3://clearml-data-deeplify/clearml-artifacts"
DOCKER_IMAGE = "862264091922.dkr.ecr.eu-central-1.amazonaws.com/model-training:latest"
DOCKER_ARGS = "--gpus all --network host --shm-size=16g -e AWS_PROFILE=clearml-s3 -v /home/ec2-user/.aws:/root/.aws:ro"
# torch 2.9.1+cu128 comes with the image; DEIM's requirements.txt minus torch, pinned on 2026-10-25
PACKAGES = [
    "faster-coco-eval==1.8.0",
    "pycocotools==2.0.11",
    "calflops==0.3.2",
    "transformers==5.19.0",
    "scipy==1.18.1",
    "tensorboard==2.21.0",
    "pyyaml==6.0.3",
    "clearml==2.1.12",
]
DEIM_REPO = "Intellindust-AI-Lab/DEIM"
DEIM_SHA = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"  # main on 2026-10-25
DEIM_CONFIG = "configs/deim_dfine/deim_hgnetv2_l_coco.yml"
HGNETV2_B4_URL = (
    "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth"
)
NUM_CLASSES = 6
# DEIM-D-FINE-L defaults (deim_hgnetv2_l_coco.yml): global lr and backbone lr, ratio 20
DEIM_LR, DEIM_BACKBONE_LR = 0.0005, 0.000025


def project_from_argv(default: str = PROJECT) -> str:
    """``--project NAME`` read before Task.init (same trick as 05_train.py)."""
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
    ap.add_argument(
        "--init-checkpoint", type=Path, help="DEIM-D-FINE-L COCO .pth (local; uploaded as artifact)"
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument(
        "--no-aug-epochs",
        type=int,
        default=8,
        help="DEIM's final phase without augmentation (repo: 8 of 58)",
    )
    ap.add_argument(
        "--imgsz", type=int, default=352, help="multiple of 32 (DEIM multi-scale steps are 32 px)"
    )
    ap.add_argument(
        "--total-batch", type=int, default=32, help="train batch over all GPUs (repo: 32)"
    )
    ap.add_argument("--val-batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers per process")
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--lr", type=float, default=None, help="override DEIM's 5e-4; backbone lr scales with it"
    )
    ap.add_argument("--checkpoint-freq", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.001, help="score cut for the prediction files")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--queue", default="", help="ClearML queue; empty = run here")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()
    if args.imgsz % 32:
        ap.error("--imgsz must be a multiple of 32")
    if args.epochs - args.no_aug_epochs < 8:
        ap.error("--epochs must leave at least 8 augmented epochs before the no-aug phase")
    if args.init_checkpoint and not args.init_checkpoint.is_file():
        ap.error(f"--init-checkpoint {args.init_checkpoint} does not exist here")
    return args


def deim_schedule(epochs: int, no_aug_epochs: int) -> dict:
    """DEIM's epoch schedule rescaled to a run of ``epochs``.

    The repo trains 58 epochs = 50 with augmentation + 8 without, flat lr until 4 + 50 // 2 = 29, Mosaic/Mixup
    policy switches at [4, 29, 50]. Same shape here: T = epochs - no_aug_epochs augmented epochs, flat until
    4 + T // 2, policy at [4, flat, T], multi-scale and Mixup stop at T, EMA restarts at T.
    """
    aug = epochs - no_aug_epochs
    flat = 4 + aug // 2
    return {
        "epoches": epochs,
        "flat_epoch": flat,
        "no_aug_epoch": no_aug_epochs,
        "policy_epochs": [4, flat, aug],
        "mixup_epochs": [4, flat],
        "stop_epoch": aug,
    }


def train_ops(imgsz: int) -> list[dict]:
    """configs/base/deim.yml's train transforms with the two sizes that depend on the input size."""
    return [
        {
            "type": "Mosaic",
            "output_size": imgsz
            // 2,  # four tiles of imgsz/2 make one imgsz canvas (repo: 320 for 640)
            "rotation_range": 10,
            "translation_range": [0.1, 0.1],
            "scaling_range": [0.5, 1.5],
            "probability": 1.0,
            "fill_value": 0,
            "use_cache": False,
            "max_cached_images": 50,
            "random_pop": True,
        },
        {"type": "RandomPhotometricDistort", "p": 0.5},
        {"type": "RandomZoomOut", "fill": 0},
        {"type": "RandomIoUCrop", "p": 0.8},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "RandomHorizontalFlip"},
        {"type": "Resize", "size": [imgsz, imgsz]},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]


def val_ops(imgsz: int) -> list[dict]:
    return [
        {"type": "Resize", "size": [imgsz, imgsz]},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
    ]


def derived_config(
    deim_dir: Path, data_root: Path, out_dir: Path, args: argparse.Namespace
) -> dict:
    """The run's DEIM config: the repo's L config plus our data, size and schedule overrides.

    DEIM merges configs key by key but replaces lists whole, so the transform op lists are given in full.
    """
    sched = deim_schedule(args.epochs, args.no_aug_epochs)
    cfg = {
        "__include__": [str((deim_dir / DEIM_CONFIG).resolve())],
        "output_dir": str(out_dir),
        "task": "detection",
        "num_classes": NUM_CLASSES,
        "remap_mscoco_category": False,
        "epoches": sched["epoches"],
        "flat_epoch": sched["flat_epoch"],
        "no_aug_epoch": sched["no_aug_epoch"],
        "eval_spatial_size": [args.imgsz, args.imgsz],
        "checkpoint_freq": args.checkpoint_freq,
        "print_freq": 100,
        "HGNetv2": {"local_model_dir": str(deim_dir / "weight" / "hgnetv2") + "/"},
        "train_dataloader": {
            "dataset": {
                "img_folder": str(data_root),
                "ann_file": str(data_root / "coco" / "instances_train.json"),
                "transforms": {
                    "ops": train_ops(args.imgsz),
                    "policy": {"epoch": sched["policy_epochs"]},
                },
            },
            "collate_fn": {
                "base_size": args.imgsz,
                "mixup_epochs": sched["mixup_epochs"],
                "stop_epoch": sched["stop_epoch"],
            },
            "total_batch_size": args.total_batch,
            "num_workers": args.workers,
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": str(data_root),
                "ann_file": str(data_root / "coco" / "instances_val.json"),
                "transforms": {"ops": val_ops(args.imgsz)},
            },
            "total_batch_size": args.val_batch,
            "num_workers": args.workers,
        },
    }
    if args.lr is not None:
        scale = args.lr / DEIM_LR
        cfg["optimizer"] = {
            "type": "AdamW",
            "params": [
                {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": DEIM_BACKBONE_LR * scale},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", "weight_decay": 0.0},
            ],
            "lr": args.lr,
            "betas": [0.9, 0.999],
            "weight_decay": 0.000125,
        }
    return cfg


def fetch_deim(dest: Path) -> Path:
    """The DEIM repo at DEIM_SHA, from GitHub's tarball (no git needed on the agent)."""
    repo_dir = dest / f"DEIM-{DEIM_SHA}"
    if (repo_dir / "train.py").is_file():
        return repo_dir
    dest.mkdir(parents=True, exist_ok=True)
    tgz = dest / f"deim-{DEIM_SHA[:8]}.tar.gz"
    urllib.request.urlretrieve(f"https://github.com/{DEIM_REPO}/archive/{DEIM_SHA}.tar.gz", tgz)
    with tarfile.open(tgz) as tar:
        tar.extractall(dest, filter="data")
    if not (repo_dir / "train.py").is_file():
        raise RuntimeError(f"DEIM tarball did not unpack to {repo_dir}")
    return repo_dir


def ensure_hgnetv2_weights(deim_dir: Path) -> Path:
    """ImageNet HGNetv2-B4 weights where the derived config points; downloaded once, on the main process."""
    target = deim_dir / "weight" / "hgnetv2" / "PPHGNetV2_B4_stage1.pth"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HGNETV2_B4_URL, target)
    return target


def parse_log(log_txt: Path) -> list[dict]:
    """DEIM's log.txt: one json line per epoch with train_* and test_coco_eval_bbox (the 12 COCO stats)."""
    rows = []
    if not log_txt.is_file():
        return rows
    for line in log_txt.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


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


def predict_coco(
    deim_dir: Path,
    cfg_path: Path,
    ckpt: Path,
    data_root: Path,
    gt_path: Path,
    imgsz: int,
    conf: float,
):
    """COCO predictions of one checkpoint on every image of a ground-truth json (tools/inference/torch_inf.py recipe)."""
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torch import nn

    sys.path.insert(0, str(deim_dir))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(cfg_path), resume=str(ckpt))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(str(ckpt), map_location="cpu")
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    cfg.model.load_state_dict(weights)

    class Deployed(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Deployed().to(device).eval()
    tf = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])
    gt = json.loads(Path(gt_path).read_text())
    rows = []
    batch = 32
    with torch.no_grad():
        for i in range(0, len(gt["images"]), batch):
            imgs = gt["images"][i : i + batch]
            pil = [Image.open(data_root / im["file_name"]).convert("RGB") for im in imgs]
            sizes = torch.tensor([[p.size[0], p.size[1]] for p in pil], device=device)
            x = torch.stack([tf(p) for p in pil]).to(device)
            labels, boxes, scores = model(x, sizes)
            for im, lab, box, sc in zip(imgs, labels, boxes, scores):
                keep = sc > conf
                for c, b, s in zip(lab[keep].tolist(), box[keep].tolist(), sc[keep].tolist()):
                    x1, y1, x2, y2 = b
                    rows.append(
                        {
                            "image_id": im["id"],
                            "category_id": int(c),
                            "bbox": [
                                round(x1, 3),
                                round(y1, 3),
                                round(x2 - x1, 3),
                                round(y2 - y1, 3),
                            ],
                            "score": round(s, 5),
                        }
                    )
    return rows


def main() -> None:
    from clearml import Task

    Task.force_store_standalone_script()
    task = Task.init(
        project_name=project_from_argv(),
        task_name="deim (naming…)",
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
    sched = deim_schedule(args.epochs, args.no_aug_epochs)
    task.set_user_properties(
        model="DEIM-D-FINE-L (HGNetv2-B4)",
        deim_sha=DEIM_SHA,
        imgsz=args.imgsz,
        epochs=args.epochs,
        schedule=json.dumps(sched),
        total_batch=args.total_batch,
        init_checkpoint=str(args.init_checkpoint)
        if args.init_checkpoint
        else "ImageNet HGNetv2-B4 only",
        task_kind="detect-deim",
    )
    if args.init_checkpoint:
        # uploaded now, where the file exists; the agent fetches it back from the task's artifacts
        task.upload_artifact(
            "init_checkpoint", artifact_object=str(args.init_checkpoint), wait_on_upload=True
        )
    if args.queue:
        task.execute_remotely(queue_name=args.queue)  # everything below runs on the agent

    from clearml import Dataset

    runs_dir = Path(args.runs_dir).resolve()
    out_dir = runs_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    deim_dir = fetch_deim(runs_dir / "third_party")
    data_root = Path(Dataset.get(dataset_id=args.dataset_id).get_local_copy())
    for part in ("train", "val", "test"):
        if not (data_root / "coco" / f"instances_{part}.json").is_file():
            raise FileNotFoundError(
                f"{data_root}/coco/instances_{part}.json: not a 11_export_coco.py dataset"
            )
    if not (data_root / "images").is_dir():
        raise FileNotFoundError(
            f"{data_root}/images: the dataset does not carry the parent's tiles"
        )

    cfg = derived_config(deim_dir, data_root, out_dir, args)
    cfg_path = out_dir / "deim_config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    task.connect_configuration(str(cfg_path), name="deim_config")
    task.upload_artifact("deim_config", artifact_object=cfg_path)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(args.devices.split(','))}",
        "--master_port=29517",
        "train.py",
        "-c",
        str(cfg_path),
        "--use-amp",
        "--seed",
        str(args.seed),
        "--output-dir",
        str(out_dir),
        "--summary-dir",
        str(out_dir / "summary"),
    ]
    if args.init_checkpoint:
        ckpt = task.artifacts["init_checkpoint"].get_local_copy()
        cmd += ["-t", str(ckpt)]
    else:
        ensure_hgnetv2_weights(deim_dir)
    env = os.environ | {"CUDA_VISIBLE_DEVICES": args.devices, "PYTHONUNBUFFERED": "1"}
    print("[deim]", " ".join(cmd))
    subprocess.run(cmd, cwd=deim_dir, env=env, check=True)

    # ---- metrics from log.txt -> ClearML scalars ------------------------------------------------------------
    log = task.get_logger()
    rows = parse_log(out_dir / "log.txt")
    for r in rows:
        epoch = int(r.get("epoch", 0))
        stats = r.get("test_coco_eval_bbox")
        if stats:
            log.report_scalar("val", "mAP50-95", float(stats[0]), epoch)
            log.report_scalar("val", "mAP50", float(stats[1]), epoch)
        for k, v in r.items():
            if k.startswith("train_") and isinstance(v, (int, float)):
                log.report_scalar("train", k.removeprefix("train_"), float(v), epoch)
    task.upload_artifact("log_txt", artifact_object=out_dir / "log.txt")

    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "schedule": sched,
        "deim_sha": DEIM_SHA,
        "epochs_logged": len(rows),
        "best_epoch_by_val_mAP50-95": max(
            (r for r in rows if r.get("test_coco_eval_bbox")),
            key=lambda r: r["test_coco_eval_bbox"][0],
            default={},
        ).get("epoch"),
    }
    best = out_dir / "best_stg2.pth"
    if not best.is_file():
        best = out_dir / "best_stg1.pth"
    checkpoints = {"best": best, "last": out_dir / "last.pth"}
    for tag, ckpt in checkpoints.items():
        if not ckpt.is_file():
            summary[f"{tag}_missing"] = str(ckpt)
            continue
        task.upload_artifact(f"{tag}.pth", artifact_object=ckpt)
        for part in ("val", "test"):
            gt_path = data_root / "coco" / f"instances_{part}.json"
            preds = predict_coco(
                deim_dir, cfg_path, ckpt, data_root, gt_path, args.imgsz, args.conf
            )
            pred_path = out_dir / f"preds_{part}_{tag}.json"
            pred_path.write_text(json.dumps(preds))
            task.upload_artifact(f"preds_{part}_{tag}", artifact_object=pred_path)
            result = coco_score(gt_path, preds)
            summary[f"{part}_{tag}"] = result
            log.report_single_value(f"{part} {tag} mAP50", result["mAP50"])
            log.report_single_value(f"{part} {tag} mAP50-95", result["mAP50-95"])
            for cname, v in result["per_class_AP50"].items():
                log.report_single_value(f"{part} {tag} AP50/{cname}", v)
            print(
                f"[deim] {part} {tag}: mAP50 {result['mAP50']:.4f} mAP50-95 {result['mAP50-95']:.4f}"
            )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    task.upload_artifact("summary", artifact_object=out_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    task.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
