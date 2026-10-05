# Deeplify March-2026 baseline — verified facts (reference only)

> 2026-10-02: the plan to re-train this run was **dropped**; Bartu decided WP1 reproduces the SWRD *paper's* YOLOv8 baseline instead (see `../../swrd_paper_baseline/README.md`). The table below stays because every thesis result is still compared against this run.

Verified 2026-10-02 from the ClearML record and the code at the commit it ran.

## What the March baseline actually was (verified 2026-10-02, not from memory)

| item | value | source |
|---|---|---|
| ClearML task | `ef3b737535184373a3f659d609d3713d` `512x512_bs64`, project `weld_defect_segmentation` | API |
| created / stopped | 2026-03-09 19:28 UTC → 2026-03-10 20:59 UTC (25.5 h), **stopped by hand** (`tasks.stop`), not converged | API |
| epochs run | 17 validation points (epochs 0–16); best `val_iou_epoch` **0.4838 at epoch 15**; epoch 16 = 0.4806 | scalars |
| steps | 322 optimizer steps / epoch = 82,625 patches ÷ (8 batch × 4 GPUs × 8 accumulation) | arithmetic |
| code | repo `deeplify-ai/deeplify`, branch `weld-defect-segmentation`, commit **`4d9f5bc224edf6c808cd78003a8275b47002a646`** ("update batch size", Felix, 2026-03-09), working dir `ml/training/weld_defect_segmentation`, entry `scripts/training/main.py` | task.script |
| uncommitted diff | 1,095 bytes touching only `ml/training/weld_defect_detection/...` (the *older* WandB codebase) — **not executed**, irrelevant | task.script.diff |
| model | `smp.Segformer(encoder_name="mit_b4", encoder_weights="imagenet", in_channels=1, classes=7)` | model_factory.py |
| loss | Dice only (`smp.losses.DiceLoss(mode="multiclass", from_logits=True)`) | lightningmodule.py |
| optimizer | AdamW lr 2e-4, weight decay 0.01, no scheduler | lightningmodule.py |
| batch | 8 per GPU × 4 GPUs (DDP) × 8 accumulation = 256 effective; run name "bs64" = 8×8 | hyperparams |
| precision | FP32 (no precision option existed) | base.yaml @ commit |
| seed | `seed_everything(42)` | main.py |
| augmentation | p=0.8: RandomResizedCrop(0.7–1.0, ratio 0.9–1.1) + H/V flip 0.5 + rotation ±5° + ColorJitter(0.3, 0.3) + sharpness 0.7/1.3 + GaussianBlur p 0.2 + Gaussian noise σ 0.01 p 0.2; CutMix p 0.3 per batch; **no polarity inversion** | datamodule.py |
| normalization | mean 0.5, std 0.25 (**not** the measured `train_stats.json` value) | base.yaml @ commit |
| data | ClearML dataset `be5ca720d8af4eb7a911e863c9989ad7` "SWRD Weld Defect Segmentation v7", built 2026-03-04 on the `default` agent | API |
| data layout | `train_patches/{images,masks}` 82,625 + `val_patches/{images,masks}` 20,491 = 103,116 patches; `split.json`, `patch_index.json`, `train_stats.json` | dataset listing |
| data build | `data_processing/create_dataset.py` @ commit: per-film **min-max stretch to 8-bit**, LabelMe polygons → `cv2.fillPoly`, 6 Chinese labels mapped (气孔→1, 夹渣/夹钨→2, 裂纹→3, 咬边→4, 未熔合→5, 未焊透→6), everything else → background, 512×512 tiles with last tile anchored at the edge, zero-padding, **all tiles kept** (no defect-free filtering), film-level `train_test_split(test_size=val_ratio, random_state=seed)` | create_dataset.py |
| metric | `JaccardIndex(task="multiclass", num_classes=7, ignore_index=0, average="macro")`, torchmetrics object passed to `self.log(on_epoch=True)` → **one `compute()` on the accumulated confusion over the whole val set**, synced across ranks. Background false positives are invisible (ignore_index=0) | lightningmodule.py |
| per-class best-epoch values | porosity 0.6206, inclusion 0.2389, crack 0.7415, undercut 0.2964, LoF 0.4149, LoP 0.7295 | scalars |
| pip pins on the task | torch 2.10.0, torchvision 0.25.0, lightning 2.6.1, segmentation-models-pytorch 0.5.0, timm 1.0.25, torchmetrics 1.8.2, clearml 2.1.3, hydra-core 1.3.2, numpy 2.4.2 | task.script.requirements |
| docker | `862264091922.dkr.ecr.eu-central-1.amazonaws.com/model-training:latest` (moving tag) | task |
| worker | `ip-172-31-33-116` (`multi-gpu` queue), still registered and idle on 2026-10-02 | workers API |
| eval task | `6a0a0ac812774b568718b4d7c46363b7` = `validate_all` stage on model `ce0108c617864f839760d981c8d1eb29`, plots only | API |

v7 build task `eea1dd59a8844592b4c498ded887a1d5` (builder commit `12213218`): `--val-ratio 0.2 --seed 42`, 3,792 train / 948 val films,
measured train pixel mean 0.4243 / std 0.2174 (the run used 0.5 / 0.25 instead).
