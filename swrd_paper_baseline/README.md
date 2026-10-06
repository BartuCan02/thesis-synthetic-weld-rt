# swrd_paper_baseline — reproduce the SWRD paper's YOLOv8 baseline

Self-contained folder (own `uv` project) for WP1's first deliverable: re-run the detection baseline published
with the dataset — Zhao et al., J. Nondestructive Evaluation 44:50 (2025), `docs/literature/zhao2025_swrd.md` —
starting from the raw release files, with nothing reused from Deeplify's derived datasets or scripts.

Status 2026-10-05: yolov8n done (`067a600b…`, best.pt mAP50 **0.574** vs paper 0.482); yolov8m on 4×T4 running (`4bca601e…`, started 2026-10-05 07:01, ~34 min/epoch). Side run for Felix prepared, see *SWRD + customer films* below.

Status 2026-10-02: **steps 1–2 done on the full 4,930-image set** (190 unpublished T/2 crops reconstructed from the originals, see `results/step1_step2_findings_2026-10-02.md`). Step 3 done: v1.0 rendered (159,914 tiles) and uploaded as ClearML dataset `de772ad9363c4067bed5835e13a9be81`; integrity check clean (`results/check_report_v1.0_2026-10-03.md`). YOLOv8n smoke runs passed on the box and on the multi-gpu agent. Step 4: YOLOv8n 100 epochs, launch command handed to Bartu 2026-10-03. Batch size to be tuned at training time. Every step ends with a gate that Bartu ticks
before the next one starts. Code is written one script at a time and reviewed before it runs.

## Target numbers (paper, Table 5, on their random 10 % tile split)

| model | best mAP50 (epoch) | best mAP50-95 (epoch) |
|---|---|---|
| YOLOv8n | 0.48196 (71) | 0.28667 (71) |
| YOLOv8s | 0.59763 (76) | 0.37998 (75) |
| **YOLOv8m** | **0.66265 (72)** | **0.44827 (74)** |
| YOLOv8l | 0.59839 (62) | 0.38664 (62) |
| YOLOv8x | 0.57274 (64) | 0.39021 (100) |

"Reproduced" means: same pipeline, same model, mAP50 within about ±0.03 of the table. Random split, random
negatives and an unversioned framework make a closer match unlikely; the paper gives no seeds.

## The paper's pipeline, step by step (what we rebuild)

1. **Raw input** — 4,930 weld-cropped 16-bit TIFFs + LabelMe polygon JSONs (`crop_weld_data` in the release;
   mirror `s3://swdr/cropped/crop_weld_images/{L,T}/` and `crop_weld_jsons/`). The 3,675 uncropped originals
   (`Raw_data`, mirror `s3://swdr/raw_tif_16_bit/`) are the true raw data, but the paper's crop boxes are not
   published, so the cropped set is the earliest point from which the paper's pipeline can be followed exactly.
   The originals stay as a reference (to confirm crops and read IQIs) and are not re-cropped.
2. **Tiling** — square window, side = half the image's shorter side, 50 % overlap, left→right then top→bottom.
   Paper: "over 380,000" tiles, 80,648 with a defect; equal number of random defect-free tiles kept.
3. **Preprocessing per tile** — contrast stretch → 8-bit → CLAHE → 3-channel 8-bit image. Parameters unpublished.
4. **Labels** — polygons → axis-aligned boxes in tile coordinates, 6 classes.
5. **Split** — 9:1 random over tiles (Table 3: 72,585+72,646 train, 8,099+8,038 val). No test set.
6. **Train** — Ultralytics YOLOv8 detect, COCO weights, 100 epochs, batch 480/480/480/320/240, everything else
   default (image size 640). Metric: mAP50 and mAP50-95 on val, best epoch.

## Decisions to take in step 0 (defaults proposed; recorded in `docs/decisions.md` once agreed)

| id | question | proposed default | why |
|---|---|---|---|
| D1 | raw input | the release's cropped TIFF+JSON pairs; originals for verification only | paper's crop boxes not published |
| D2 | which models | YOLOv8n first (pipeline check, cheap), then YOLOv8m (headline); s/l/x only if time | Table 5 shows m is the result that matters |
| D3 | framework version | `ultralytics==8.4.171` (current) | paper's 2024 version is unknown; record that defaults may have drifted |
| D4 | compute | box `~/swrd_paper_baseline/` for data; `default` queue (1×T4) for n, `multi-gpu` (4×T4) for m | memory: Ultralytics DDP on the agent hangs at teardown; workaround known |
| D5 | contrast stretch | per-tile linear stretch between the 0.5 and 99.5 percentiles | "contrast stretching" unspecified; percentile stretch is the textbook version; min-max is the alternative to ablate |
| D6 | CLAHE | OpenCV `clipLimit=2.0, tileGridSize=(8,8)` on the 8-bit tile | unspecified; OpenCV's own default (40.0) is far too strong for radiographs |
| D7 | polygon → box in a tile | rasterised polygon piece ≥ 4 px, kept if ≥ 10 % of the polygon is visible or the piece covers ≥ 0.5 % of the tile | unspecified; sweep of 18 rules (2026-10-02): no rule matches all six classes, D7 matches four within 3 % and the positive total within 1 %, and never labels a window the defect's pixels do not enter |
| D10 | window arithmetic | side = round(short/2), stride = round(side/2), a window that ends exactly on the border counts, no extra flush window | sweep of 72 grid variants (2026-10-02): reproduces 412,726 tiles and 79,957 defect tiles vs the paper's "over 380,000" and 80,648 |
| D8 | seeds | one fixed seed for negatives and split (0); report that the paper's are unknown | reproducibility |
| D9 | batch size | as large as fits (T4 16 GB), record it; Ultralytics scales weight decay by batch/64 so batch 480 is not reproducible on our GPUs | paper's 480 on one P40 is not physically possible at 640 px either |

## Steps and gates

### Step 0 — agree D1–D9 and this layout
**Gate:** Bartu approves; `docs/decisions.md` gets the entry.

### Step 1 — inventory from the raw files (`scripts/00_inventory.py`)
List the S3 mirror, pair every TIFF with its JSON, hash-dedupe (36 byte-identical files are known), explain
the 4,740 TIFFs counted today vs 4,930 expected, read image dimensions, histogram every label string.
**Gate:** Fig. 4 counts reproduced from the JSONs (25,401 / 2,017 / 1,754 / 213 / 802 / 745) or each
difference explained → fixes the Chinese-label → 6-class mapping; predicted tile count under the paper's rule ≈ 380k.

### Step 2 — tile index (`scripts/01_tile.py`)
On the 16-bit TIFFs: paper's window rule, one record per tile (film, x, y, side) with its boxes under rule
D7 and two flags, *has_box* and *touches_defect*. **No pixels are written**: the full grid is >380k tiles
(~95 GB as 16-bit PNG) and only 161k of them enter the dataset.
**Gate:** ≈ 80,648 tiles with a box; per-class tile instance counts within a few % of Table 2.

### Step 3 — select, split, render, upload (`scripts/02_select_split.py`, `03_render.py`, `04_upload_clearml.py`)
- `02_select_split.py`: all positives + an equal number of random *clean* tiles (no defect polygon touches
  them, not even a dropped sliver), 9:1 random split over tiles (`split_tiles.json`). Also writes
  `split_films.json`, the same tiles split by exposure (A_/B_ halves together) for step 5.
- `03_render.py`: crops each selected window from the TIFF, D5 stretch → 8-bit → D6 CLAHE, PNG + YOLO label
  file in the Ultralytics layout, `swrd6.yaml`, `render_params.json`. Single-channel PNGs by default (identical
  tensors after loading; `--three-channel` for the literal 24-bit files).
- `04_upload_clearml.py`: ClearML Dataset in project `thesis_wp1_benchmark`, stored on S3, with every
  pipeline parameter and count as metadata. Visual check of 20 tiles against Fig. 12 before uploading.
**Gate:** Table 3 counts within ±1 %; tiles look like Fig. 12 right-hand column; dataset id recorded.

### Step 4 — train (`scripts/05_train.py`)
YOLOv8n, 100 epochs, defaults, ClearML task in project `thesis_wp1_benchmark`. Bartu launches. Then YOLOv8m.
Rough cost, to be measured on the first epochs: n ≈ 1 day on one T4; m ≈ 3–4 days on one T4 or ≈ 1 day on 4×T4.
**Gate:** n within ±0.03 of 0.482 mAP50; m within ±0.03 of 0.663.

### Step 5 — evaluate and write up (`scripts/06_eval.py`)
Best-epoch mAP50 / mAP50-95, plus per-class AP (the paper has none). Then the first thesis-protocol number:
re-score the same model on a **film-level** split to measure how much the tile-level split inflates the result.
Results → `results/`, `experiments/wp1_benchmark/README.md`, `experiments/README.md`.

## Side run: SWRD + customer films, scored on the SWRD val tiles (Felix, 2026-10-05)

Felix's question: keep the paper's configuration, add Deeplify's customer films to the training set, evaluate on
the same SWRD val tiles — does our data help or hurt? (He expects worse: SWRD looks lab-clean next to customer scans.)

Customer films are made to look like SWRD release pairs, then the pipeline above runs on them unchanged:

| step | script | runs where | what it fixes |
|---|---|---|---|
| 0 | `scripts/export_customer_films.py` | deeplify mlops env (Mongo + S3) — **Bartu runs it** | raw_extraction_16bit variant only; polarity canonicalised to SWRD's (metal bright); six-class mask from the dataset_builder readers; other weld-defect classes → `other_defect` polygons; crop = seam bbox + 0.1 × seam short side, grown to contain every label; films without a seam mask or without a six-class pixel are skipped and counted |
| 1–4 | `scripts/run_customer_pipeline.sh` | this venv, the box | inventory → tile (edge drop, D7, `--exclude-from-negatives other_defect`) → select with `--val-ratio 0` (all tiles to train) → render (D5/D6) |
| 5 | `scripts/07_merge_upload.py` | this venv, `AWS_PROFILE=data-rw` | ClearML child dataset of v1.0 `de772ad9…`: parent files untouched (val identical), customer tiles added to `images/train` only; refuses val tiles and id clashes |
| 6 | `scripts/05_train.py` | box → `multi-gpu` queue | identical args to `4bca601e…` with the child dataset id |

```
cd ~/deeplify-wt-weldsuite/ml/scripts/mlops && AWS_PROFILE=data-rw uv run python ~/thesis/swrd_paper_baseline/scripts/export_customer_films.py \
    --mlops-dir . --env-file ~/deeplify/ml/data_management/.env --sources oge aramco maroca \
    --seam-mask-dir ~/seam_masks_defect_v2 --fallback-seam-mask-dir ~/seam_masks_defect --out ~/swrd_paper_baseline/data/raw_customer
cd ~/thesis/swrd_paper_baseline && bash scripts/run_customer_pipeline.sh ~/swrd_paper_baseline/data/raw_customer customer_v1
AWS_PROFILE=data-rw uv run python scripts/07_merge_upload.py --parent-id de772ad9363c4067bed5835e13a9be81 \
    --customer-yolo-dir ~/swrd_paper_baseline/data/yolo_customer_v1 --customer-work-dir ~/swrd_paper_baseline/data/work_customer_v1 \
    --raw-customer-dir ~/swrd_paper_baseline/data/raw_customer
uv run python scripts/05_train.py --dataset-id <child id> --model yolov8m --epochs 100 --emulate-paper-batch --batch 96 \
    --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu --name v1.0+customer-yolov8m-100ep-paperbatch-4gpu
```

Steps 1–5 were smoke-tested on 2026-10-05 with three SWRD films relabelled in English (144 tiles; merge dry run against
the real parent passed). Step 0 could not be run by Claude: reads of the Mongo catalogue are blocked for it, so the
export, its report (`<out>/export_report.json`) and the queueing are Bartu's. Unit tests: `tests/test_customer.py`.

Readout: best.pt mAP50 / mAP50-95 and per-class AP50 of the +customer run against `4bca601e…` on the identical val set.
Caveats to state with the number: the val split is tile-level; customer labels are still under audit at Deeplify;
customer crops come from predicted seams, SWRD crops from the authors.

## Layout
```
swrd_paper_baseline/
  README.md              this file: protocol, decisions, status
  pyproject.toml         uv project, exact pins (ultralytics, torch, tifffile, opencv-headless, ...)
  configs/swrd6.yaml     Ultralytics data file, 6 classes in paper order
  scripts/_common.py     constants from the paper, label map, window rule, D7 box rule, D5/D6 preprocessing
  scripts/00_inventory.py  step 1   raw files vs the paper's counts
  scripts/01_tile.py       step 2   tile index + boxes, no pixels
  scripts/02_select_split.py step 3a  negatives + 9:1 split (tile-level, plus a film-level variant)
  scripts/03_render.py     step 3b  crop, preprocess, write YOLO layout
  scripts/04_upload_clearml.py step 3c  ClearML Dataset with metadata
  scripts/fetch_from_official_zip.py   HTTP-range reader for the official SWXD_Data.zip (list / fetch selected members)
  scripts/reconstruct_missing_crops.py rebuild unpublished crops from Raw_data originals (validated pixel-exact)
  scripts/05_train.py      step 4   Ultralytics training, ClearML task, remote queue
  scripts/06_eval.py       step 5   score saved weights on the tile- or film-level val split
  scripts/_customer.py, export_customer_films.py, run_customer_pipeline.sh, 07_merge_upload.py   side run (customer films)
  tests/test_common.py   unit tests for the window rule, the D7 box rule, the stretch
  results/               numbers, curves, tables (small files only; no images, no weights)
```
Data and runs live on the box and are never committed:
```
~/swrd_paper_baseline/data/raw/crop_weld_images/{L,T}/*.tif   aws s3 sync s3://swdr/cropped/ data/raw/  (AWS_PROFILE=data-rw)
~/swrd_paper_baseline/data/raw/crop_weld_jsons/{L,T}/*.json
~/swrd_paper_baseline/data/work/        inventory.json, inventory_report.md, tiles.jsonl, tiles_summary.json, split_*.json
~/swrd_paper_baseline/data/yolo_tilesplit/   images/{train,val}, labels/{train,val}, swrd6.yaml, render_params.json
```
Code runs from a clone of this repo on the box (`~/thesis/swrd_paper_baseline`, `uv sync`).

## Glossary
- **Ultralytics** — the company and Python package (`pip install ultralytics`) behind the YOLO family (YOLOv5, v8, v11).
  One call trains a detector: `YOLO("yolov8m.pt").train(data="swrd6.yaml", epochs=100)`. It ships COCO-pretrained
  weights, its own augmentation (mosaic, HSV jitter, flips, scaling), its own optimizer defaults and its own mAP
  evaluation. "All unspecified parameters default" in the paper means *these* defaults, for whichever version they had.
- **YOLOv8 detect** — an object *detector*: it predicts axis-aligned boxes with a class and a confidence, not pixel masks.
  The paper turned the polygons into boxes for this.
- **mAP50 / mAP50-95** — mean over classes of the average precision when a predicted box counts as correct at IoU ≥ 0.5
  with a ground-truth box / averaged over IoU thresholds 0.5, 0.55, …, 0.95. Not comparable to segmentation IoU.
- **Tile-level split** — train and val tiles drawn from the same films (and overlapping each other by 50 %); numbers
  are optimistic compared with a film-level split.
