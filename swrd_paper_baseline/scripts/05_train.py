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
                               With a split file that has a test list (09_test_split.py): after the last epoch,
                               best.pt and last.pt are scored on the held-out test tiles, which sit in a folder of
                               their own that training never reads; the scores land in the same task ("test
                               best.pt mAP50" ...) and in summary.json under "test".

Split and sampling (WP1 oversampling experiment, experiments/wp1_benchmark/oversampling_rfs.md):
  --split film         (default, the thesis protocol) the tiles split by original exposure: val = every tile of
                       the exposures in split_films.json (02_select_split.py), train = all other tiles. Needs
                       --split-file. The launching machine reads the file; the list of val exposures travels to
                       the agent in the task's configuration.
  --split tile         the dataset's own train/val folders = the paper's random 9:1 over tiles. Use it to repeat
                       the paper runs.
  --rfs-threshold t    repeat factor sampling (Gupta, Dollar, Girshick, LVIS, CVPR 2019): a class found in fewer
                       than a fraction t of the train tiles gets its tiles listed r = sqrt(t / share) times per
                       epoch. 0 = off (default).
  --equal-sampling     (2026-10-23, Cecilia's reference arm for run B: "all classes the same number") every class is
                       seen in the same number of training tiles per epoch, --equal-target (max | second | N; default
                       second = the second most common class's tile count, 15,042 on the real film-split tiles, which
                       keeps the epoch length). Rarer classes are repeated up to it; classes above it keep a share of
                       their tiles (the rest is left out of that run). Defect-free tiles once. Classes that share many
                       tiles with rarer ones can stay above the target: porosity ends 6 % above on the real tiles (see
                       equal_factors; the run warns). Not combinable with --rfs-threshold.
Only --split tile without oversampling trains from the dataset folder as it is. Every other choice trains from a
folder of symlinks next to the run (<runs-dir>/<name>_view), so the downloaded dataset is never written to and
the paper runs' label cache is left alone.

Segmentation arm (2026-10-16): ``--model yolov8m-seg`` (any size) trains the Ultralytics segmentation variant on a
dataset whose labels are polygons (03_render.py --labels polygon). The paper batch / emulation values are those of
the detection model of the same size. Ultralytics reports box metrics "(B)" and mask metrics "(M)"; the summary
and the ClearML single values carry both, so the box mAP50 is directly comparable with the detection runs.

RT-DETR arm (2026-10-25, reference-model comparison, experiments/wp1_benchmark/reference_models.md):
``--model rtdetr-l`` (or rtdetr-x) trains Ultralytics' RT-DETR (Zhao et al., CVPR 2024: HGNetv2 backbone, hybrid
encoder with one attention layer on the stride-32 map, 300 queries, 6 deformable decoder layers, Hungarian
matching, no NMS) on the same tiles, split, imgsz and schedule as the YOLOv8m runs, from the COCO checkpoint
``rtdetr-l.pt``. mAP50 / mAP50-95 and the per-class AP come from Ultralytics' RTDETRValidator on the identical val
and test tiles, so the rows are directly comparable with the bm1 YOLOv8m numbers. --emulate-paper-batch maps
rtdetr-* to yolov8m's paper batch, so optimizer steps per epoch and weight decay match the bm1 reference.
RT-DETR's deformable attention has no deterministic CUDA backward: --deterministic defaults to false for it.

Data: a ClearML Dataset id (downloaded on whichever machine runs this; the data yaml path is rewritten to the
local copy) or a local folder with ``swrd6.yaml``. Remote execution: ``--queue multi-gpu`` enqueues the task
and exits; the agent runs the same script. The multi-gpu agent has 4x T4 and 48 vCPU; the 4 vCPUs of the data
box are too few for Ultralytics' CPU-side augmentation at 144k images/epoch.

Run (examples; the first three are the paper protocol, so they pass --split tile):
  uv run python scripts/05_train.py --data-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid --split tile --model yolov8n --epochs 1 --fraction 0.02 --name smoke
  uv run python scripts/05_train.py --dataset-id <clearml id> --split tile --model yolov8n --queue multi-gpu --devices 0 --name v1.0-yolov8n
  uv run python scripts/05_train.py --dataset-id <clearml id> --split tile --model yolov8m --queue multi-gpu --devices 0,1,2,3 --name v1.0-yolov8m
  uv run python scripts/05_train.py --dataset-id <clearml id> --split film --split-file ~/swrd_paper_baseline/data/work/split_films.json \
      --rfs-threshold 0.1 --model yolov8m --queue multi-gpu --devices 0,1,2,3 --name v1.0-film-rfs0.1-yolov8m
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT = "thesis_wp1_benchmark"


def project_from_argv(default: str = PROJECT) -> str:
    """``--project NAME`` read before Task.init (which needs the project before argparse runs).

    Bartu, 2026-10-13: the SWRD+customer side run for Felix belongs in Deeplify's ``weld_defect_segmentation``
    project, the thesis runs stay in ``thesis_wp1_benchmark``. On the agent argv is empty and the task already
    lives in its project, so the default is never used there.
    """
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--project" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--project="):
            return a.split("=", 1)[1]
    return default


OUTPUT_URI = "s3://clearml-data-deeplify/clearml-artifacts"
DOCKER_IMAGE = "862264091922.dkr.ecr.eu-central-1.amazonaws.com/model-training:latest"
DOCKER_ARGS = "--gpus all --network host --shm-size=16g -e AWS_PROFILE=clearml-s3 -v /home/ec2-user/.aws:/root/.aws:ro"
PAPER_BATCH = {"yolov8n": 480, "yolov8s": 480, "yolov8m": 480, "yolov8l": 320, "yolov8x": 240}
# RT-DETR is not in the paper. Its entries copy yolov8m's, so that --emulate-paper-batch gives the RT-DETR arm the
# same effective batch, nbs and weight decay as the bm1 YOLOv8m reference: the comparison then changes only the model.
PAPER_BATCH |= {"rtdetr-l": PAPER_BATCH["yolov8m"], "rtdetr-x": PAPER_BATCH["yolov8m"]}
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
    # RT-DETR: same per-GPU batch as yolov8m (x20 = 480). RT-DETR-L at 352 px should fit a 16 GB T4 at 24; watch
    # the first epoch's memory and pass --batch 12 (accumulate 40) if it does not.
    "rtdetr-l": 24,
    "rtdetr-x": 16,
}  # x8 / x10 / x20 / x16 / x15 = paper batch
MODEL_CHOICES = list(PAPER_BATCH) + [f"{m}-seg" for m in PAPER_BATCH if m.startswith("yolov8")]


def detect_base(model: str) -> str:
    """'yolov8m-seg' -> 'yolov8m': the detection model whose paper batch and emulation values apply."""
    return model.removesuffix("-seg")


def is_rtdetr(model: str) -> bool:
    """'rtdetr-l' / 'rtdetr-x': the Ultralytics RT-DETR arm of the reference-model comparison."""
    return model.startswith("rtdetr")


def task_kind(model: str) -> str:
    """ClearML user property: what kind of model the run trains."""
    if model.endswith("-seg"):
        return "segment"
    if is_rtdetr(model):
        return "detect-rtdetr"
    return "detect"


def model_class(model: str):
    """The Ultralytics model class for a --model name: RTDETR for rtdetr-*, YOLO otherwise.

    Imported lazily: the agent installs ultralytics only when the task runs, and the unit tests never need it.
    RTDETR carries its own trainer, validator and predictor (300 queries, score threshold instead of NMS). Loading
    an RT-DETR checkpoint through YOLO() would validate it with the YOLO postprocessing and NMS.
    """
    if is_rtdetr(model):
        from ultralytics import RTDETR

        return RTDETR
    from ultralytics import YOLO

    return YOLO


def deterministic_for(model: str, choice: str) -> bool:
    """--deterministic auto: True for YOLO (Ultralytics' default), False for RT-DETR.

    RT-DETR's deformable attention (F.grid_sample) has no deterministic CUDA backward; Ultralytics documents
    deterministic=False for it. Two RT-DETR runs with the same seed therefore differ slightly.
    """
    if choice == "auto":
        return not is_rtdetr(model)
    return choice == "true"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-id", help="ClearML Dataset id (output of 04_upload_clearml.py)")
    src.add_argument(
        "--data-dir", type=Path, help="local YOLO folder with swrd6.yaml (output of 03_render.py)"
    )
    ap.add_argument(
        "--model",
        default="yolov8n",
        choices=MODEL_CHOICES,
        help="detection (yolov8n..x) or segmentation (yolov8n-seg..x-seg, needs polygon labels)",
    )
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
        "--deterministic",
        default="auto",
        choices=["auto", "true", "false"],
        help="torch deterministic algorithms; auto = true for YOLO, false for RT-DETR (its deformable "
        "attention has no deterministic CUDA backward)",
    )
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
        "--split",
        default="film",
        choices=["tile", "film"],
        help="film (default, thesis protocol) = by exposure, from --split-file; "
        "tile = the dataset's own folders (paper protocol)",
    )
    ap.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="split_films.json from 02_select_split.py, needed with --split film; read where it exists "
        "(the box), not on the agent",
    )
    ap.add_argument(
        "--rfs-threshold",
        type=float,
        default=0.0,
        help="repeat factor sampling threshold t (LVIS); classes in fewer than t of the train tiles "
        "are oversampled; 0 = off",
    )
    ap.add_argument(
        "--equal-sampling",
        action="store_true",
        help="sample tiles so every class is seen in the same number of tiles per epoch (--equal-target)",
    )
    ap.add_argument(
        "--equal-target",
        default="second",
        help="with --equal-sampling: tile views per class per epoch: max, second (default) or a number",
    )
    ap.add_argument(
        "--tags", nargs="*", default=[], help="ClearML tags for the task, e.g. benchmark-v1"
    )
    ap.add_argument(
        "--name", required=True, help="run name (ClearML task name and local run folder)"
    )
    ap.add_argument(
        "--project", default=PROJECT, help="ClearML project of the task (read before Task.init)"
    )
    ap.add_argument(
        "--queue", default="", help="ClearML queue for remote execution; empty = run here"
    )
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()
    if args.split_file and args.split != "film":
        ap.error("--split-file only applies to --split film")
    if not 0.0 <= args.rfs_threshold < 1.0:
        ap.error("--rfs-threshold is a fraction of tiles: 0 <= t < 1")
    if args.equal_sampling and args.rfs_threshold:
        ap.error("--equal-sampling and --rfs-threshold are two different samplers: choose one")
    if args.equal_target not in ("max", "second") and not (
        args.equal_target.isdigit() and int(args.equal_target) > 0
    ):
        ap.error("--equal-target must be max, second or a positive whole number")
    return args


def exposure_of(tile_id: str) -> str:
    """Original exposure of a tile: the image stem before '__', T-joint halves A_/B_ merged.

    Same rule as ``_common.exposure_id``, repeated here because the agent receives this file alone.
    """
    stem = tile_id.split("__")[0]
    return stem[2:] if stem[:2] in ("A_", "B_") else stem


def connect_film_split(task, split_file: Path | None) -> dict:
    """Store the film split in the task, so the agent can rebuild it without the split file.

    Where ``split_file`` exists (the box, at launch) its val exposures go into the task's configuration
    "film_split". On the agent the file does not exist, and ClearML returns the stored copy instead.
    A split with a "test" list (09_test_split.py) also stores the test exposures: their tiles are held out of
    training and validation alike. A split without one gives an empty test list, as before.
    """
    local = {
        "val_exposures": [],
        "expected_val_tiles": 0,
        "test_exposures": [],
        "expected_test_tiles": 0,
        "source": "",
    }
    if split_file and Path(split_file).is_file():
        split = json.loads(Path(split_file).read_text())
        local = {
            "val_exposures": sorted({exposure_of(t) for t in split["val"]}),
            "expected_val_tiles": len(split["val"]),
            "test_exposures": sorted({exposure_of(t) for t in split.get("test", [])}),
            "expected_test_tiles": len(split.get("test", [])),
            "source": str(split_file),
        }
    cfg = task.connect_configuration(local, name="film_split")
    if not cfg.get("val_exposures"):
        raise SystemExit("--split film needs --split-file pointing at an existing split_films.json")
    if set(cfg.get("test_exposures") or []) & set(cfg["val_exposures"]):
        raise SystemExit("film split: an exposure is in both val and test")
    return {
        "val_exposures": cfg["val_exposures"],
        "expected_val_tiles": cfg["expected_val_tiles"],
        "test_exposures": cfg.get("test_exposures") or [],
        "expected_test_tiles": cfg.get("expected_test_tiles") or 0,
        "source": cfg["source"],
    }


def assign_parts(
    root: Path, val_exposures: set[str] | None, test_exposures: set[str] | None = None
) -> dict[str, list[Path]]:
    """Every tile image of the dataset, grouped into train, val and (held out) test.

    ``None`` keeps the dataset's own folders. A set of exposures puts every tile of those exposures into
    val, whichever folder it sits in; tiles of ``test_exposures`` go to test, which no model ever sees in
    training or validation; every other tile goes to train.
    """
    images = sorted(p for part in ("train", "val") for p in (root / "images" / part).glob("*.png"))
    if val_exposures is None:
        parts = {part: [p for p in images if p.parent.name == part] for part in ("train", "val")}
        return parts | {"test": []}
    test_exposures = test_exposures or set()
    return {
        "train": [
            p
            for p in images
            if exposure_of(p.stem) not in val_exposures
            and exposure_of(p.stem) not in test_exposures
        ],
        "val": [p for p in images if exposure_of(p.stem) in val_exposures],
        "test": [p for p in images if exposure_of(p.stem) in test_exposures],
    }


def label_of(root: Path, image: Path) -> Path:
    return root / "labels" / image.parent.name / f"{image.stem}.txt"


def tile_classes(label_file: Path) -> set[int]:
    """Class ids present in one YOLO label file (an empty file = a defect-free tile)."""
    return {int(line.split()[0]) for line in label_file.read_text().splitlines() if line.strip()}


def repeat_factors(
    classes_per_tile: list[set[int]], threshold: float
) -> tuple[list[float], dict[int, float]]:
    """Repeat factor sampling (LVIS, Gupta et al. 2019).

    share_c  = fraction of train tiles that contain class c
    r_c      = max(1, sqrt(threshold / share_c))      a class above the threshold keeps r_c = 1
    r_tile   = max of r_c over the classes in the tile (1 for a defect-free tile)
    """
    n = len(classes_per_tile)
    tiles_with = Counter(c for cs in classes_per_tile for c in cs)
    r_class = {c: max(1.0, math.sqrt(threshold / (k / n))) for c, k in sorted(tiles_with.items())}
    r_tile = [max((r_class[c] for c in cs), default=1.0) for cs in classes_per_tile]
    return r_tile, r_class


def equal_target(spec: str, classes_per_tile: list[set[int]]) -> int:
    """The tile views per class per epoch that --equal-target asks for.

    "max"    the tile count of the most common class (porosity in SWRD)
    "second" the tile count of the second most common class (lack of penetration in SWRD, 15,042 on the film
             split's training tiles): the epoch then stays about as long as without rebalancing
    a number that number
    """
    counts = sorted(Counter(c for cs in classes_per_tile for c in cs).values(), reverse=True)
    if spec == "max":
        return counts[0]
    if spec == "second":
        return counts[1] if len(counts) > 1 else counts[0]
    if spec.isdigit() and int(spec) > 0:
        return int(spec)
    raise ValueError(f"--equal-target must be max, second or a positive whole number, not {spec!r}")


def equal_factors(
    classes_per_tile: list[set[int]], target: float | None = None
) -> tuple[list[float], dict[int, float]]:
    """Equal sampling: every class is seen in ``target`` tiles per epoch (default: the most common class's count).

    Each class c gets a factor r_c; a tile holding defects is listed max(r_c over its classes) times, a defect-free
    tile once. Start with r_c = target / n_c (n_c = tiles holding c) and rescale every round
        r_c <- r_c x target / (tile views of c this round)
    until every class sees ``target`` (at most 500 rounds). The rescaling is needed because classes share tiles: a
    tile repeated for its rarer class also shows its other classes again. A factor below 1 means its tiles are kept
    at that rate (round_repeats gives each 0 or 1 copy): this is how a class above the target is brought down.

    A class can stay above the target when it sits in too many tiles of rarer classes: its own factor then sinks
    towards 0 (all its tiles that hold it alone are left out) and the shared tiles still show it more than
    ``target`` times. On the SWRD film split's real tiles porosity ends 6 % above any target for this reason;
    05_train.py warns when the classes end more than 2 % apart.
    """
    tiles_with = Counter(c for cs in classes_per_tile for c in cs)
    if not tiles_with:
        return [1.0] * len(classes_per_tile), {}
    if target is None:
        target = max(tiles_with.values())
    r_class = {c: target / k for c, k in sorted(tiles_with.items())}
    for _ in range(500):
        views = Counter()
        for cs in classes_per_tile:
            if cs:
                w = max(r_class[c] for c in cs)
                for c in cs:
                    views[c] += w
        if all(abs(views[c] - target) <= 1e-9 * target for c in r_class):
            break
        r_class = {c: r * target / views[c] for c, r in r_class.items()}
    r_tile = [max(r_class[c] for c in cs) if cs else 1.0 for cs in classes_per_tile]
    return r_tile, r_class


def equal_spread(classes_per_tile: list[set[int]], copies: list[int]) -> float:
    """(most - least) tile views per epoch over the classes present, relative to the most: 0 = all equal."""
    seen = Counter()
    for cs, k in zip(classes_per_tile, copies):
        for c in cs:
            seen[c] += k
    if not seen:
        return 0.0
    return (max(seen.values()) - min(seen.values())) / max(seen.values())


def round_repeats(r_tile: list[float], seed: int) -> list[int]:
    """Whole copies per tile: floor(r), plus one more with probability r - floor(r).

    LVIS redraws this every epoch. A file list is fixed, so it is drawn once, with a fixed seed.
    """
    rng = random.Random(seed)
    return [int(r) + int(rng.random() < r - int(r)) for r in r_tile]


def class_table(
    classes_per_tile: list[set[int]], copies: list[int], r_class: dict[int, float], names: dict
) -> list[dict]:
    """Per class: train tiles, their share, the repeat factor, and how often the class is seen per epoch."""
    n = len(classes_per_tile)
    rows = []
    for c, name in names.items():
        tiles = sum(1 for cs in classes_per_tile if c in cs)
        per_epoch = sum(k for cs, k in zip(classes_per_tile, copies) if c in cs)
        rows.append(
            {
                "class": name,
                "train_tiles": tiles,
                "share_of_tiles": round(tiles / n, 4),
                "r_class": round(r_class.get(c, 1.0), 3),
                "tiles_per_epoch": per_epoch,
                "x_vs_no_oversampling": round(per_epoch / tiles, 3) if tiles else None,
            }
        )
    return rows


def write_test_view(
    args: argparse.Namespace, root: Path, data: dict, test_images: list[Path]
) -> Path | None:
    """Put the held-out test tiles in a folder of their own and return a data yaml that points only at it.

    The folder sits next to the training view, not inside it, and the training yaml never names it, so
    Ultralytics cannot see a test tile while it trains or while it picks best.pt. Only the scoring after
    the last epoch (score_weights) reads this yaml. None when the split has no test list.
    """
    if not test_images:
        return None
    view = (Path(args.runs_dir) / f"{args.name}_test_view").resolve()
    shutil.rmtree(view, ignore_errors=True)  # holds only symlinks and Ultralytics label caches
    for sub in ("images", "labels"):
        (view / sub / "test").mkdir(parents=True)
    for img in test_images:
        label = label_of(root, img)
        if not label.is_file():
            raise FileNotFoundError(label)
        (view / "images" / "test" / img.name).symlink_to(img)
        (view / "labels" / "test" / label.name).symlink_to(label)
    # Ultralytics scores the yaml's "val" entry; "train" must exist, so it names the same folder
    test_data = {
        "path": str(view),
        "train": "images/test",
        "val": "images/test",
        "names": data["names"],
    }
    out = Path(args.runs_dir) / f"{args.name}_test_data.yaml"
    out.write_text(yaml.safe_dump(test_data, sort_keys=False, allow_unicode=True))
    return out


def score_weights(
    weights: Path, data_yaml: Path, imgsz: int, device: int, model: str = "yolov8m"
) -> dict:
    """mAP50, mAP50-95 and per-class AP of one checkpoint on the "val" entry of a data yaml.

    ``model`` is the --model name: it picks the validator (YOLO with NMS, or RT-DETR's score cut).
    """
    val = model_class(model)(str(weights)).val(
        data=str(data_yaml), imgsz=imgsz, batch=32, device=device, plots=False, verbose=False
    )
    names = val.names
    idx = val.box.ap_class_index.tolist()
    return {
        "mAP50": float(val.box.map50),
        "mAP50-95": float(val.box.map),
        "per_class_AP50": {names[i]: float(v) for i, v in zip(idx, val.box.ap50.tolist())},
        "per_class_AP50-95": {names[i]: float(v) for i, v in zip(idx, val.box.ap.tolist())},
    }


def resolve_data_yaml(
    args: argparse.Namespace, film_split: dict | None = None
) -> tuple[Path, dict]:
    """Return a data yaml whose paths point at a local copy of the tiles, and what went into train/val.

    Tile split without oversampling (the paper runs): the dataset folder as it is.
    Otherwise: a symlink view of the dataset with the chosen split; with RFS, train is a list file in
    which each tile appears once per copy (Ultralytics keeps repeated lines).
    """
    if args.data_dir:
        root = Path(args.data_dir).resolve()
    else:
        from clearml import Dataset

        root = Path(Dataset.get(dataset_id=args.dataset_id).get_local_copy())
    src = root / "swrd6.yaml"
    data = yaml.safe_load(src.read_text())
    out = Path(args.runs_dir) / f"{args.name}_data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)

    if (
        args.split == "tile"
        and not args.rfs_threshold
        and not getattr(args, "equal_sampling", False)
    ):
        data["path"] = str(root)
        out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        n_train = sum(1 for _ in (root / "images" / "train").glob("*.png"))
        return out, {"split": "tile", "train_tiles": n_train, "train_entries": n_train}

    val_exposures = set(film_split["val_exposures"]) if args.split == "film" else None
    test_exposures = set(film_split.get("test_exposures") or []) if args.split == "film" else None
    parts = assign_parts(root, val_exposures, test_exposures)
    if film_split and len(parts["val"]) != film_split["expected_val_tiles"]:
        raise RuntimeError(
            f"film split: {len(parts['val'])} val tiles here, "
            f"{film_split['expected_val_tiles']} in {film_split['source']}"
        )
    if film_split and len(parts["test"]) != film_split.get("expected_test_tiles", 0):
        raise RuntimeError(
            f"film split: {len(parts['test'])} test tiles here, "
            f"{film_split.get('expected_test_tiles', 0)} in {film_split['source']}"
        )

    view = (Path(args.runs_dir) / f"{args.name}_view").resolve()
    shutil.rmtree(view, ignore_errors=True)  # holds only symlinks and Ultralytics label caches
    for part in ("train", "val"):  # test tiles are held out: they never enter the view
        images = parts[part]
        for sub in ("images", "labels"):
            (view / sub / part).mkdir(parents=True)
        for img in images:
            label = label_of(root, img)
            # a missing label would silently turn a defect tile into a negative
            if not label.is_file():
                raise FileNotFoundError(label)
            (view / "images" / part / img.name).symlink_to(img)
            (view / "labels" / part / label.name).symlink_to(label)
    test_yaml = write_test_view(args, root, data, parts["test"])

    classes = [tile_classes(label_of(root, img)) for img in parts["train"]]
    copies = [1] * len(classes)
    r_class: dict[int, float] = {}
    equal = getattr(args, "equal_sampling", False)
    target = None
    if args.rfs_threshold or equal:
        if equal:
            target = equal_target(getattr(args, "equal_target", "second"), classes)
            r_tile, r_class = equal_factors(classes, target)
        else:
            r_tile, r_class = repeat_factors(classes, args.rfs_threshold)
        copies = round_repeats(r_tile, args.seed)
        if equal and (spread := equal_spread(classes, copies)) > 0.02:
            print(
                f"[data] WARNING equal sampling: classes differ by {spread:.1%} in tile views per epoch; "
                "a class shares too many tiles with rarer ones to be brought down to the target "
                f"({target:,}); see the per-class table"
            )
        train_list = view / ("train_equal.txt" if equal else "train_rfs.txt")
        train_list.write_text(
            "".join(
                f"{view / 'images' / 'train' / img.name}\n" * k
                for img, k in zip(parts["train"], copies)
            )
        )
        data["train"] = train_list.name
    data["path"] = str(view)
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    info = {
        "split": args.split,
        "split_source": film_split["source"] if film_split else "dataset folders",
        "rfs_threshold": args.rfs_threshold,
        "rfs_seed": args.seed if args.rfs_threshold else None,
        "equal_sampling": equal,
        "equal_sampling_seed": args.seed if equal else None,
        "equal_target": target,
        "train_tiles_left_out": sum(1 for k in copies if k == 0),
        "train_tiles": len(parts["train"]),
        "val_tiles": len(parts["val"]),
        "test_tiles_held_out": len(parts["test"]),
        "test_data_yaml": str(test_yaml) if test_yaml else None,
        "train_entries": sum(copies),
        "per_class": class_table(classes, copies, r_class, data["names"]),
    }
    return out, info


def print_data_info(info: dict) -> None:
    print(
        f"[data] split {info['split']}: train {info['train_tiles']:,} tiles, "
        f"val {info.get('val_tiles', '(dataset folder)')} tiles, "
        f"test held out {info.get('test_tiles_held_out', 0)} tiles; "
        f"train entries per epoch {info['train_entries']:,}"
    )
    if "per_class" in info:
        print("| class | train tiles | share | r_class | tiles per epoch | x |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in info["per_class"]:
            print(
                f"| {r['class']} | {r['train_tiles']:,} | {r['share_of_tiles']:.4f} | {r['r_class']:.2f} "
                f"| {r['tiles_per_epoch']:,} | {r['x_vs_no_oversampling']} |"
            )


def main() -> None:
    from clearml import Task

    # Since 2026-10-05 ~/thesis on the box is a git clone whose origin is the bare repo ~/thesis.git, a path
    # the agent cannot clone. This file imports nothing from the repo, so ship it alone, as the paper runs were.
    Task.force_store_standalone_script()
    # Task.init BEFORE parse_args: ClearML patches argparse so that, on the agent, the arguments stored
    # in the task's "Args" section are injected (the agent passes no command line of its own).
    task = Task.init(
        project_name=project_from_argv(),
        task_name="swrd-paper-baseline (naming…)",
        task_type=Task.TaskTypes.training,
        output_uri=OUTPUT_URI,
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch": False, "matplotlib": True, "tensorboard": False},
    )
    args = parse_args()
    task.set_name(args.name)
    if args.tags:
        task.add_tags(args.tags)
    task.set_base_docker(DOCKER_IMAGE, docker_arguments=DOCKER_ARGS)
    # The image (probed 2026-10-03) has torch 2.9.1+cu128, opencv 5.0, numpy 2.5 but no ultralytics; pin only
    # what is missing so pip does not pull a second torch.
    task.set_packages(["ultralytics==8.4.171", "clearml==2.1.12", "pyyaml==6.0.3"])
    task.set_user_properties(
        paper_batch=PAPER_BATCH[detect_base(args.model)],
        paper_epochs=100,
        paper="Zhao 2025 Table 4",
        task_kind=task_kind(args.model),
    )
    film_split = connect_film_split(task, args.split_file) if args.split == "film" else None
    if args.queue:
        task.execute_remotely(queue_name=args.queue)  # everything below runs on the agent

    from ultralytics import settings

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
    data_yaml, data_info = resolve_data_yaml(args, film_split)
    print_data_info(data_info)
    task.upload_artifact("data_view", artifact_object=data_info)
    devices = [int(d) for d in str(args.devices).split(",")]

    nbs = args.nbs or ULTRALYTICS_NBS
    weight_decay = args.weight_decay if args.weight_decay is not None else ULTRALYTICS_WD
    batch = args.batch
    if args.emulate_paper_batch:
        paper = PAPER_BATCH[detect_base(args.model)]
        nbs = args.nbs or paper
        weight_decay = (
            args.weight_decay
            if args.weight_decay is not None
            else ULTRALYTICS_WD * paper / ULTRALYTICS_NBS
        )
        if batch == -1:
            batch = EMULATE_BATCH[detect_base(args.model)]
    accumulate = max(round(nbs / batch), 1) if batch > 0 else None
    n_train = data_info["train_entries"]  # with RFS, repeated tiles count once per copy
    task.set_user_properties(
        split=args.split,
        rfs_threshold=args.rfs_threshold,
        equal_sampling=args.equal_sampling,
        equal_target=data_info.get("equal_target"),
        train_tiles_left_out=data_info.get("train_tiles_left_out"),
        train_tiles=data_info["train_tiles"],
        train_entries=n_train,
        nbs=nbs,
        batch=batch,
        accumulate=accumulate,
        effective_batch=(batch * accumulate) if accumulate else None,
        weight_decay=weight_decay,
        optimizer_steps_per_epoch=round(n_train * args.fraction / (batch * accumulate))
        if accumulate
        else None,
        paper_steps_per_epoch=round(
            (72585 + 72646) / PAPER_BATCH[detect_base(args.model)]
        ),  # Table 3 train set
    )
    print(
        f"[train] batch {batch} x accumulate {accumulate} = effective {batch * accumulate if accumulate else '?'}; "
        f"nbs {nbs}; weight_decay {weight_decay:.5f}; paper: batch {PAPER_BATCH[detect_base(args.model)]}"
    )

    deterministic = deterministic_for(args.model, args.deterministic)
    print(f"[train] model class {model_class(args.model).__name__}, deterministic {deterministic}")
    # COCO-pretrained, downloaded by Ultralytics on first use (yolov8m.pt, rtdetr-l.pt, ...)
    model = model_class(args.model)(f"{args.model}.pt")
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
        deterministic=deterministic,
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
        "data": data_info,
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
        m50, m5095 = "metrics/mAP50(M)", "metrics/mAP50-95(M)"  # segmentation models only
        if m50 in rows[-1]:
            bm50 = max(rows, key=lambda r: r.get(m50, 0))
            bm5095 = max(rows, key=lambda r: r.get(m5095, 0))
            summary["best_mask_mAP50"] = {"value": bm50[m50], "epoch": int(bm50["epoch"])}
            summary["best_mask_mAP50-95"] = {"value": bm5095[m5095], "epoch": int(bm5095["epoch"])}
            summary["last"].update({m50: rows[-1][m50], m5095: rows[-1][m5095]})
    try:
        val = model_class(args.model)(str(run_dir / "weights" / "best.pt")).val(
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
        seg = getattr(val, "seg", None)  # mask metrics, segmentation models only
        if seg is not None and getattr(seg, "ap_class_index", None) is not None:
            summary["best_pt_val"]["mask_mAP50"] = float(seg.map50)
            summary["best_pt_val"]["mask_mAP50-95"] = float(seg.map)
            summary["best_pt_val"]["per_class_mask_AP50"] = {
                names[i]: float(v) for i, v in zip(seg.ap_class_index.tolist(), seg.ap50.tolist())
            }
            for cname, v in summary["best_pt_val"]["per_class_mask_AP50"].items():
                task.get_logger().report_single_value(f"mask AP50/{cname}", v)
            task.get_logger().report_single_value("best.pt mask mAP50", float(seg.map50))
            task.get_logger().report_single_value("best.pt mask mAP50-95", float(seg.map))
    except Exception as e:  # noqa: BLE001 — the summary must never kill a finished run
        summary["best_pt_val_error"] = repr(e)
    # ---- held-out test set, scored once, after training: it played no part in training or in picking best.pt
    if data_info.get("test_data_yaml"):
        summary["test"] = {"tiles": data_info["test_tiles_held_out"]}
        for ckpt in ("best", "last"):
            try:
                r = score_weights(
                    run_dir / "weights" / f"{ckpt}.pt",
                    Path(data_info["test_data_yaml"]),
                    args.imgsz,
                    devices[0],
                    args.model,
                )
            except Exception as e:  # noqa: BLE001 — as above
                summary["test"][f"{ckpt}_error"] = repr(e)
                continue
            summary["test"][ckpt] = r
            log = task.get_logger()
            log.report_single_value(f"test {ckpt}.pt mAP50", r["mAP50"])
            log.report_single_value(f"test {ckpt}.pt mAP50-95", r["mAP50-95"])
            for cname, v in r["per_class_AP50"].items():
                log.report_single_value(f"test {ckpt}.pt AP50/{cname}", v)
            print(
                f"[test] {ckpt}.pt on {data_info['test_tiles_held_out']:,} test tiles: "
                f"mAP50 {r['mAP50']:.4f}, mAP50-95 {r['mAP50-95']:.4f}"
            )
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
