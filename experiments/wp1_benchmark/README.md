# wp1_benchmark

Goal: reproduce the SWRD paper's YOLOv8 detection baseline (Zhao et al. 2025, Table 5) from the raw release, then
build the thesis benchmark on the same raw files. Code: `../../swrd_paper_baseline/`.

Data version: ClearML dataset **`de772ad9363c4067bed5835e13a9be81`** = `thesis_wp1_benchmark / swrd-paper-tiles 1.0.0`
(uploaded 2026-10-03; paper grid D10, box rule D7, stretch p0.5–p99.5, CLAHE clip 2.0 grid 8, seed 0; 143,923 train +
15,991 val tiles; 4,894 unique images incl. 190 reconstructed T/2 crops). Build log: `../../swrd_paper_baseline/results/`.

deeplify commit: not used (self-contained Ultralytics pipeline).

## Runs
| date | ClearML task | config | result | conclusion |
|---|---|---|---|---|
| 2026-10-03 | `4d557a3a97df470b9cdbf3692cddfec7` smoke-v1.0-yolov8n | yolov8n, 1 epoch, 2 % of train, batch 16, box T4 | mAP50 0.043 (porosity 0.25) | pipeline check only: training, val, per-class AP, summary, weights all land in ClearML |
| 2026-10-03 | `69e49dd79da14b22af7bee991b768791` smoke-remote-v1.0-yolov8n | same, on the multi-gpu agent from the ClearML dataset | mAP50 0.014 | remote path works: standalone script, Ultralytics on the image's torch 2.9.1, dataset pulled to the agent cache |
| 2026-10-04/05 | `067a600b97e84c43aaffbc772ffe29f9` v1.0-yolov8n-100ep-paperbatch | yolov8n, 100 ep, imgsz 640, batch 60×8 = 480 (nbs 480, wd 0.00375), 1×T4 on multi-gpu agent, 23 h | **mAP50 0.576 @ep 68, mAP50-95 0.335 @ep 73** (best.pt re-val 0.574 / 0.335); AP50 per class: porosity 0.78, inclusion 0.34, crack 0.60, undercut 0.41, LoF 0.50, LoP 0.81 | paper YOLOv8n: 0.482 / 0.287. Ours +0.09 mAP50 with the same effective batch; peak epoch (68) close to the paper's (71). Output model `879e3c804d7949cab6e99a5c7fb815a1` |
| 2026-10-05 → | `3270cb1a67ac479a861e92cd893c9822` v1.0-yolov8m-100ep-paperbatch | yolov8m, batch 24×20 = 480, 1×T4 | running; ~60 min/epoch → ~4.3 days | paper YOLOv8m: 0.663 / 0.448 |
