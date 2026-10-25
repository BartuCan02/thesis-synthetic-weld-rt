# Reference-model comparison (branch `wp1-reference-models`, built and launched 2026-10-25)

Cecilia (2026-10-19): pick one peer-reviewed, state-of-the-art detector as the fixed reference and run every data
strategy on it. Survey and shortlist: `../../docs/literature/model_survey_2026-10-22.md`. Three arms, all on the
bm1 recipe (real tiles, `split_films_test.json`, 352 px, 50 epochs, seed 0, 4 x T4), compared with
`bm1-real-yolov8m-352-50ep` `b8d70cdb…` (val best.pt 0.462 / 0.245, test best.pt 0.469 / 0.235).

| arm | model | paper | code path | task name |
|---|---|---|---|---|
| 1 | RT-DETR-L (HGNetv2-L) | Zhao et al., CVPR 2024 | Ultralytics, `05_train.py --model rtdetr-l` | `bm1-real-rtdetr-l-352-50ep` |
| 2 | DEIM-D-FINE-L (HGNetv2-B4) | Huang et al., CVPR 2025 on Peng et al., ICLR 2025 | github `Intellindust-AI-Lab/DEIM` @ `09d35d53`, `14_train_deim.py` | `bm1-real-deim-dfine-l-352-50ep` |
| 3 | RF-DETR Large (DINOv2-S windowed) | Robinson et al., arXiv 2511.09554 (ICLR 2026 per Roboflow, unverified) | pip `rfdetr[train]==1.11.2`, `15_train_rfdetr.py` | `bm1-real-rfdetr-l-352-50ep` |

## Status 2026-10-25 (launched by Claude on Bartu's "go for it")

| step | result |
|---|---|
| COCO ground truth | `11_export_coco.py` on the box: train 143,918 / val 11,756 / test 4,240 images (matches the bm1 agent log); boxes train 134,996 / val 11,215 / test 4,840. ClearML dataset **`bee4e92541cf4ad5b420865573113204`** (`swrd-paper-tiles-coco 1.0.0`, child of `de772ad9…`). Local copies: `~/swrd_paper_baseline/data/coco_v1.0/{coco,images}` (images = symlink to the v1.0 tiles), smoke subset `~/swrd_paper_baseline/data/coco_smoke` (600 / 200 / 100). |
| RT-DETR smoke | `smoke-rtdetr-l-352` on the box T4, 1 epoch on 1 %: training, val, best.pt and last.pt test scoring and summary all OK (per-class AP present). Ultralytics put the run folder under `~/deeplify-wt-weldsuite/runs/detect/runs/` (the box's Ultralytics `runs_dir` setting); the summary step found it. |
| Arm 1 queued | `bm1-real-rtdetr-l-352-50ep` task **`c0c4f2fcda6f4e97b5b2a5369e07fd7d`**, queue multi-gpu, batch 96 x accumulate 5. |
| DEIM smoke | First attempt failed: torchvision 0.29 dropped the `_transform` hook DEIM implements. Fixed by `patch_deim_transforms` (writes `engine/data/transforms/_tv_compat.py` into the fetched repo). Second attempt `smoke-deim-dfine-l-352`, 10 epochs on the subset: training, best_stg1/best_stg2/last, predictions and pycocotools scores on val and test, summary all OK (val best mAP50 0.057 after 10 x 75 iterations, lr still in the 2,000-iteration warm-up: a smoke artefact). The tuning load reports the 80-class heads and the size-dependent anchors as unmatched, as expected. |
| Arm 2 queued | `bm1-real-deim-dfine-l-352-50ep` task **`4e41d179d6964e359f5b821616634e20`**, checkpoint `~/deim_dfine_l_coco.pth` (model zoo, 126 MB) uploaded as the task's `init_checkpoint` artifact. |
| RF-DETR smoke | `smoke-rfdetr-l-352`, 3 epochs on the subset: see below. |

## How the comparison stays fair

- Same tiles, same film split, same 352 px input, same 50 epochs, same seed. The three arms differ in model and in
  each model's own training recipe (optimizer, augmentation, losses), which is how detector papers compare.
  Arm 1 even shares the Ultralytics augmentation and nbs-480 optimizer schedule with the YOLOv8m run.
- One scorer for everything. Each codebase reports mAP with its own evaluator, so all four models are also scored
  with pycocotools on the same COCO ground truth (`13_score_coco.py`): the DEIM and RF-DETR wrappers write COCO
  predictions on val and test at the end of training; `12_predict_coco.py` writes them for the Ultralytics weights
  (bm1 YOLOv8m and RT-DETR). Ultralytics' own mAP is still reported for arm 1 so it can be read next to the bm1 table.
- The 96-film test set is never visible to a training run: arm 1 holds it out as before; arms 2 and 3 are given
  only `instances_train.json` and `instances_val.json`, and read `instances_test.json` only after the last epoch.

## Launch order (box, `cd ~/thesis/swrd_paper_baseline`, `AWS_PROFILE=data-rw`)

1. COCO ground truth, once (also the dataset arms 2 and 3 read):
   ```
   uv run python scripts/11_export_coco.py --yolo-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid \
       --split-file ~/swrd_paper_baseline/data/work/split_films_test.json --out-dir ~/swrd_paper_baseline/data/coco_v1.0 \
       --upload --parent-id de772ad9363c4067bed5835e13a9be81 --name swrd-paper-tiles-coco --version 1.0.0
   ```
   Expect train 143,918 / val 11,756 / test 4,240 images (the counts of `b8d70cdb…`'s agent log). Note the dataset id.
2. Arm 1, same flags as `bm1-real` plus the model:
   ```
   uv run python scripts/05_train.py --dataset-id de772ad9363c4067bed5835e13a9be81 --split film \
       --split-file ~/swrd_paper_baseline/data/work/split_films_test.json --model rtdetr-l --imgsz 352 --epochs 50 \
       --emulate-paper-batch --batch 96 --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu \
       --tags benchmark-v1 reference-models --name bm1-real-rtdetr-l-352-50ep
   ```
   If the first epoch runs out of memory: `--batch 48` (accumulate 10, same effective batch 480).
3. Arm 2. Download the DEIM-D-FINE-L COCO checkpoint from the model zoo (Google Drive, link in the DEIM README) to
   the box once, then:
   ```
   uv run python scripts/14_train_deim.py --dataset-id <coco dataset id> --init-checkpoint ~/deim_dfine_l_coco.pth \
       --epochs 50 --imgsz 352 --total-batch 32 --devices 0,1,2,3 --seed 0 --queue multi-gpu \
       --tags benchmark-v1 reference-models --name bm1-real-deim-dfine-l-352-50ep
   ```
4. Arm 3:
   ```
   uv run python scripts/15_train_rfdetr.py --dataset-id <coco dataset id> --epochs 50 --imgsz 352 --batch 4 \
       --grad-accum 1 --devices 0,1,2,3 --seed 0 --queue multi-gpu --tags benchmark-v1 reference-models \
       --name bm1-real-rfdetr-l-352-50ep
   ```
5. Unified scores for the Ultralytics arms (bm1 YOLOv8m and RT-DETR), val and test, best and last:
   ```
   uv run python scripts/12_predict_coco.py --data-dir ~/swrd_paper_baseline/data/yolo_v1.0_papergrid \
       --split-file ~/swrd_paper_baseline/data/work/split_films_test.json --part test --task-id b8d70cdb8fa54a72bd91a2ffd931ed1d \
       --checkpoint best --imgsz 352 --gt ~/swrd_paper_baseline/data/coco_v1.0/coco/instances_test.json --name coco-bm1-real-yolov8m-best-test
   ```
   Arms 2 and 3 upload `preds_<part>_<ckpt>.json` and their scores as task artifacts; `13_score_coco.py` rescans any of them.

## What to read

Per arm, val and test, best and last: mAP50, mAP50-95, per-class AP50, all from pycocotools. Decision metrics
(survey, section 1): porosity AP50 near 0.70 (one stride-8 cell per pore suffices for the model), undercut and
lack-of-fusion AP50 (where attention and box-conditioned sampling should help), precision at matched recall
(learned duplicate suppression vs NMS). Compare curves at equal epochs as well as final numbers: arms 2 and 3 start
from 640-px COCO weights and re-fit their positional encodings at 352.

## Review notes for Bartu (what was not run)

- Nothing here has touched a GPU. Unit tests cover the config and data side (`tests/test_export_coco.py`,
  `test_score_coco.py`, `test_train_deim.py`, `test_train_rfdetr.py`, the RT-DETR cases in `test_train_sampling.py`).
  The first run of each wrapper on the agent is the integration test; expect to iterate on package pins.
- Arm 1: `RTDETR("rtdetr-l.pt").train(...)` accepts the same overrides as YOLO. Ultralytics' RT-DETR dataset keeps
  mosaic and the other YOLO augmentations on, and `optimizer=auto` picks SGD at our iteration count, as for YOLOv8m.
  That is the "same recipe" choice; the official RT-DETR recipe (AdamW 1e-4, no mosaic) is a possible second run.
- Arm 2: DEIM's config merge replaces lists whole, so `14_train_deim.py` writes the full transform lists with the two
  size-dependent values (Mosaic canvas 176, Resize 352). Schedule rescaled to 42 augmented + 8 plain epochs
  (`deim_schedule`). `remap_mscoco_category: False` makes DEIM use our 0-based category ids as labels. Batch 32 over
  4 GPUs at 352 px should fit a T4; AMP on. The checkpoint must be fetched by hand (Google Drive).
- Arm 3: rfdetr trains under PyTorch Lightning, so the wrapper runs the training as a torchrun child process of the
  ClearML task and replays rfdetr's TensorBoard scalars afterwards. `resolution=352` is allowed (block size 32).
  Default per-GPU batch 4, effective 16, lr 1e-4, as the rfdetr docs recommend. `rfdetr[train]` pulls Lightning,
  torchmetrics and several COCO evaluators; first install on the agent may take a while.
- Pins: `faster-coco-eval==1.8.0`, `pycocotools==2.0.11`, `calflops==0.3.2`, `transformers==5.19.0`, `scipy==1.18.1`,
  `tensorboard==2.21.0`, `rfdetr==1.11.2`; all at least two weeks old on 2026-10-25. torch stays the image's 2.9.1.
- `pycocotools==2.0.11` was added to `swrd_paper_baseline/pyproject.toml` for `12_predict_coco.py` / `13_score_coco.py`.
