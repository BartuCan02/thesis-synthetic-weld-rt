# wp1_benchmark

Goal: reproduce the SWRD paper's YOLOv8 detection baseline (Zhao et al. 2025, Table 5) from the raw release, then
build the thesis benchmark on the same raw files. Code: `../../swrd_paper_baseline/`.

Data version: ClearML dataset **`de772ad9363c4067bed5835e13a9be81`** = `thesis_wp1_benchmark / swrd-paper-tiles 1.0.0`
(uploaded 2026-10-03; paper grid D10, box rule D7, stretch p0.5–p99.5, CLAHE clip 2.0 grid 8, seed 0; 143,923 train +
15,991 val tiles; 4,894 unique images incl. 190 reconstructed T/2 crops). Build log: `../../swrd_paper_baseline/results/`.

deeplify commit: not used (self-contained Ultralytics pipeline).

## Result of the paper reproduction (2026-10-06)

| model | ours mAP50 / mAP50-95 (best epoch) | paper Table 5 | diff |
|---|---:|---:|---:|
| YOLOv8n | 0.576 (68) / 0.335 (73) | 0.482 (71) / 0.287 (71) | +0.094 / +0.048 |
| YOLOv8m | 0.730 (77) / 0.462 (78) | 0.663 (72) / 0.448 (74) | +0.067 / +0.014 |

Same pipeline as the paper (grid, negatives, split, stretch→8-bit→CLAHE, Ultralytics defaults, effective batch 480 via nbs);
unpublished details filled with D5–D7/D10 (see `../../swrd_paper_baseline/README.md`). Ranking n < m and peak epochs match the paper;
absolute numbers are higher, most plausibly from label-border handling, CLAHE/stretch parameters and the 2026 Ultralytics defaults
(`../../swrd_paper_baseline/results/step1_step2_findings_2026-10-02.md`, section on the gap). Eval JSONs in `../../swrd_paper_baseline/results/`.

## Planned
- Oversampling of the rare classes (repeat factor sampling, t = 0.1) vs none, YOLOv8m on the film split: two runs,
  the one without oversampling is also step 5. Plan, numbers and commands: `oversampling_rfs.md` (not launched yet).

## Runs
| date | ClearML task | config | result | conclusion |
|---|---|---|---|---|
| 2026-10-03 | `4d557a3a97df470b9cdbf3692cddfec7` smoke-v1.0-yolov8n | yolov8n, 1 epoch, 2 % of train, batch 16, box T4 | mAP50 0.043 (porosity 0.25) | pipeline check only: training, val, per-class AP, summary, weights all land in ClearML |
| 2026-10-03 | `69e49dd79da14b22af7bee991b768791` smoke-remote-v1.0-yolov8n | same, on the multi-gpu agent from the ClearML dataset | mAP50 0.014 | remote path works: standalone script, Ultralytics on the image's torch 2.9.1, dataset pulled to the agent cache |
| 2026-10-04/05 | `067a600b97e84c43aaffbc772ffe29f9` v1.0-yolov8n-100ep-paperbatch | yolov8n, 100 ep, imgsz 640, batch 60×8 = 480 (nbs 480, wd 0.00375), 1×T4 on multi-gpu agent, 23 h | **mAP50 0.576 @ep 68, mAP50-95 0.335 @ep 73** (best.pt re-val 0.574 / 0.335); AP50 per class: porosity 0.78, inclusion 0.34, crack 0.60, undercut 0.41, LoF 0.50, LoP 0.81 | paper YOLOv8n: 0.482 / 0.287. Ours +0.09 mAP50 with the same effective batch; peak epoch (68) close to the paper's (71). Output model `879e3c804d7949cab6e99a5c7fb815a1` |
| 2026-10-05 | `3270cb1a67ac479a861e92cd893c9822` v1.0-yolov8m-100ep-paperbatch | yolov8m, batch 24×20, 1×T4 | stopped at epoch 1 by Bartu (≈60 min/epoch) | relaunched on 4 GPUs |
| 2026-10-05/06 | `4bca601e1a2f42ccb90d99f0eaad42ed` v1.0-yolov8m-100ep-paperbatch-4gpu | yolov8m, 100 ep, batch 96 (24/GPU) × 5 = 480, nbs 480, wd 0.00375, 4×T4 DDP, 25 h | **mAP50 0.730 @ep 77, mAP50-95 0.462 @ep 78**; best.pt re-val (`b1208ab34c47449ea0603a964ae0091e`): 0.730 / 0.462; AP50 per class: porosity 0.87, inclusion 0.62, crack 0.70, undercut 0.64, LoF 0.69, LoP 0.86; AP50-95: 0.55 / 0.38 / 0.43 / 0.38 / 0.45 / 0.59 | **paper YOLOv8m: 0.663 / 0.448.** mAP50-95 within 0.014, mAP50 +0.067. Curve peaks at ep 77 (paper 72) then falls with mosaic off. Task marked failed only because the summary step used a wrong path under DDP (fixed); weights = output model `83f25a1579b14da793e453ef6f99736b` |

## Side run for Felix: SWRD + Deeplify customer films, scored on the SWRD val tiles (prepared 2026-10-08)
*Rescued 2026-10-09 from an uncommitted worktree. Written by an earlier Claude session; not yet reviewed by Bartu.*
Question: same paper configuration, same SWRD val tiles, does adding our customer data help or hurt? Customer films are
exactly those of the latest Deeplify multiclass run `weld_defect_all_v2.12` (1,183 OGE, 133 Aramco, 559 Maroca; 1,076
label-free negatives), exported in the SWRD release layout (`swrd_paper_baseline/scripts/export_customer_films.py`:
raw 16-bit, SWRD polarity, seam crop from human label or v1.3 prediction grown to hold every label, other weld-defect
classes as `other_defect` = never a box, never a negative) and tiled with the v1.0 pipeline (`run_customer_pipeline.sh`).

| dataset | content | status |
|---|---|---|
| `9b43a1ecb46b46658bc62523a1ae507b` | v1.0 + 22,132 train tiles from 557 films of the v3.0 set | superseded (wrong film set) |
| `2067134b370c4757b1aaa9334f1a1397` | v1.0 + 35,744 train tiles from 1,838 of the 1,875 v2.12 films | superseded (v2.13 is the latest Deeplify run) |
| `91dbe45c7cda41d8a21835e8e6376e7c` (v1.0.1; `1587e7d3…` v1.0.0 is the same content with 22,101 dangling parent references, unusable) | v1.0 + 44,202 train tiles (22,101 with a box + 22,101 clean) from 2,491 of the 2,613 v2.13 films (`weld_defect_all_v2.13_pullv5`, dataset `15dd436e…`); val folder identical to v1.0 | **done 2026-10-15**: `bf1303065b67450da75dc71dbac3188f` `swrd-paper-film+customer-v213-yolov8m-100ep-paperbatch-4gpu`, in project **`weld_defect_segmentation`** (Bartu's choice), film split + no RFS, identical args to the film baseline `29f71fe4…`. **best.pt mAP50 0.466 / mAP50-95 0.250 vs baseline 0.453 / 0.240** (+0.013 / +0.009; best epoch 37 vs 30). AP50 per class (customer run vs baseline): porosity 0.749 vs 0.736, inclusion 0.261 vs 0.260, crack 0.558 vs 0.573, undercut 0.139 vs 0.101, LoF 0.299 vs 0.263, LoP 0.788 vs 0.783. Verdict: adding the customer films helps a little on SWRD (rare classes undercut and LoF gain most, crack loses 0.015); the gap is about three times the 0.004 run-to-run spread seen between identical-data runs, single seed. Attempts `f4aa2155…` and `cbe9e1a2…` (archived) died at dataset download with "Extracted file missing": ClearML stores a child file whose content equals a parent file as a parent reference, and every empty label of a clean customer tile equals the parent's empty labels. Fixed in `07_merge_upload.py` (empty label → newline, non-empty duplicate → trailing zeros); the cache wipe (`55dd7c87…`) in between was a wrong diagnosis |

**Readout (2026-10-15, run `bf130306…` completed 2026-10-14 21:00 UTC, 32 h):** best.pt val mAP50 **0.466** / mAP50-95 **0.249**
against the film baseline A `29f71fe4…` 0.452 / 0.240 (B 0.450 / 0.239, C 0.437 / 0.236). Per-class AP50, side run vs A:
porosity 0.749 vs 0.736, inclusion 0.261 vs 0.260, crack 0.558 vs 0.573, undercut 0.139 vs 0.101, lack of fusion 0.299 vs 0.263,
lack of penetration 0.788 vs 0.783. Precision 0.573 vs 0.539, recall 0.446 vs 0.456. Adding the customer films helped a little
(+0.013 mAP50), mostly on undercut and lack of fusion, the two classes with the most customer boxes; crack (227 customer boxes) lost.
Felix expected worse. One seed per arm, so the overall gap is at the edge of what one seed can show; the undercut and LoF gaps are larger
than any A/B/C difference seen so far. Not a scope change: thesis methods stay SWRD-only.


Comparison is against the **film-split** baseline (thesis protocol, 2026-10-08): the training script puts every tile of the 369
val exposures into val and everything else, customer tiles included, into train, so the val set is exactly the baseline's.
v2.13 customer tiles: side median 248 px (SWRD 343), min 74, boxes porosity 9,119 / inclusion 5,716 / crack 227 /
undercut 14,949 / LoF 1,496 / LoP 1,017; 2,586 films exported (26 skipped: newest label has no six-class pixel; 1 no seam),
seams: human label 981, v1.3 prediction 1,605. Caveats for the write-up: tile-level val split; Maroca labels are the HQ
project's unreviewed prelabels as of 22 Sept (same as v2.12 trained on); 258 seams predicted at export time.
Export reports: box `~/swrd_paper_baseline/data/raw_customer_v21{2,3}/export_report.json`.

## Preprocessing comparison on Deeplify's model: SegFormer on paper tiles vs v2.13 (launched 2026-10-15)
Bartu's question after the Felix run: is the paper's preprocessing better than Deeplify's? Answered with the same
model (SegFormer mit-b4, framework recipe), the same films and split (v2.13's: SWRD 3,701/305, customer 2,059/527
films present in the export), the same 9 classes and the same metric; only the preprocessing differs.

| arm | preprocessing | dataset | run |
|---|---|---|---|
| Deeplify gen-2 | seam crop, per-crop window, 512 px patches, residual channel | `15dd436e…` (55,107 / 7,243 patches) | `c3df0377…` `weld_defect_all_v2.13_pullv5`, 22 ep, best val IoU 0.425 |
| paper | half-short-side tiles, per-tile P0.5–99.5 stretch, CLAHE; all 9-class-positive tiles + 1:1 clean | `8a6aa755088142b8b54e892b34c10940` (208,300 / 30,220 tiles, `08_render_segmentation.py`, norm 0.4875/0.2453) | `da522e1206144671ba032d79350719b4` `weld_defect_all_v2.14_papertiles_9cls_v213split`, **30 ep** (Bartu's choice), in_channels 1, invert_p 0, otherwise v2.13's overrides, launched from `deeplify-wt-weldsuite` @ `78987d92c` + its 272 KB uncommitted diff |

Readout: framework val IoU per class and overall (same metric as the weld-defect project; it ignores background
false positives) plus post-hoc true IoU for both runs. Caveats: 30 vs 22 epochs; the paper arm has ~3.8x more
training units (every 9-class-positive tile, 50 % overlap) so an epoch is not the same amount of data; customer
labels are the catalogue's newest as of the export (same pull as v2.13).

## Segmentation vs detection, same model family (built 2026-10-16, run to be launched by Bartu)
Bartu's question: does training with masks instead of boxes find the defects better? Answered with one run: YOLOv8m-seg
on exactly baseline A's tiles, film split, recipe and val tiles, compared by **box mAP50 on the identical 15,996 val
tiles** (the seg model reports box "(B)" and mask "(M)" metrics side by side). Mask mAP50 is the first segmentation
number of the benchmark. Not a SegFormer comparison and not pixel IoU; those stay with the preprocessing-comparison run.

Dataset `c385c61a95bd40f191ecd8ca60894f7e` = `thesis_wp1_benchmark / swrd-paper-tiles-seg 1.0.0`: the PNGs of v1.0 (`de772ad9…`,
byte-identical, hardlinked at build time) with polygon labels of the same 151,051 D7 instances (`03_render.py
--labels polygon --reuse-images`). Each label is the traced outer boundary of the rasterised visible piece, i.e. the very
mask the D7 rule measured; the recomputed D7 boxes matched `tiles.jsonl` byte for byte on all 159,914 tiles. Build stats:
607 multi-piece instances (0.40 %, pieces joined by a zero-width bridge as Ultralytics' COCO converter does), 65 of them
(0.04 %, thin edge slivers) where the bridge line lowers the filled-polygon overlap below IoU 0.8, 0 fallbacks, 27.9
vertices per instance, labels 82 MB. The box Ultralytics derives from a polygon spans the boundary pixels inclusively, so
it is 1 px narrower and shorter than the D7 box (exclusive upper edges); the centre is identical.

Smoke on the box (one T4): `fc792848…` yolov8n-seg 1 epoch on 1,722 tiles, summary with box and mask metrics OK;
yolov8m-seg at batch 24 = 12.4 GB of 15.4 GB (tasks `smoke-seg-v1.0-yolov8m-seg-b24*`). `f9eaed55…` is an aborted
Task.init: training on the box needs `AWS_PROFILE=data-rw` for the artifact bucket (archive it). Run command, identical
to A except the dataset and model:
```
cd ~/thesis/swrd_paper_baseline && uv run python scripts/05_train.py --dataset-id c385c61a95bd40f191ecd8ca60894f7e \
    --split film --split-file ~/swrd_paper_baseline/data/work/split_films.json --model yolov8m-seg --epochs 100 \
    --emulate-paper-batch --batch 96 --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu \
    --name v1.0seg-film-yolov8m-seg-100ep-paperbatch-4gpu
```
If the 4-GPU run hits CUDA OOM, `--batch 48` keeps the effective batch (nbs 480, accumulate 10). Readout: best.pt box
mAP50 / mAP50-95 and per-class AP50 against A `29f71fe4…` 0.453 / 0.240 (identical-data runs differ by ~0.004, one seed
per arm); mask mAP50 / mAP50-95 and per-class mask AP50 as the segmentation column. Ultralytics picks best.pt by a fitness
that includes the mask metrics, so also read `best_mAP50` (box, from results.csv) in the summary artifact.
