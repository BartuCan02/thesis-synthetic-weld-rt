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
| `2067134b370c4757b1aaa9334f1a1397` | v1.0 + 35,744 train tiles (17,872 with a box + 17,872 clean) from 1,838 of the 1,875 v2.12 films; val identical to v1.0 (15,991 tiles) | **to be trained**: `v1.0+customer-v212-yolov8m-100ep-paperbatch-4gpu`, identical args to `4bca601e…` |

Customer tiles: side median 248 px (SWRD 343), min 54 (narrow seams), boxes porosity 7,842 / inclusion 5,275 / crack 206 /
undercut 11,784 / LoF 1,474 / LoP 715. Caveats for the write-up: tile-level val split; Maroca labels are the HQ
project's unreviewed prelabels as of 22 Sept (same as v2.12 trained on); 258 seams predicted at export time.
Export report: box `~/swrd_paper_baseline/data/raw_customer_v212/export_report.json`.
