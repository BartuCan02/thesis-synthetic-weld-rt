# Does oversampling help the under-represented classes? (WP1, planned 2026-10-08)

Status: **code on branch `wp1-oversampling-rfs`; Bartu chose YOLOv8m only (2026-10-08). Not launched yet.**

## The question

Some defect classes are rare in SWRD. Undercut appears in 0.5 % of the training tiles; porosity appears in 31 %.
The rare classes are also the weak ones: YOLOv8m on the paper split scored AP50 0.62 on inclusion, 0.64 on
undercut and 0.69 on lack of fusion, against 0.87 on porosity (run `4bca601e…`).

The question: if the rare-class tiles are shown to the model more often, does their AP go up?
And does that cost the frequent classes anything?

## Why the thesis needs this number

Felix framed inpainting as a way to *enlarge and rebalance* the training set (decision log, 2026-09-21).
Showing real rare tiles more often is the free way to rebalance. It needs no generator at all.
So it is the baseline that synthetic rebalancing has to beat. If synthetic defects only raise rare-class AP as
much as oversampling does, the generator has added nothing. It belongs to M1 (23 Oct): "real-only +
classical-augmentation baselines".

## The method: repeat factor sampling (RFS)

Source: Gupta, Dollár, Girshick, *LVIS: A Dataset for Large Vocabulary Instance Segmentation*, CVPR 2019.

Terms:
- **share of a class** = the fraction of training tiles that contain at least one box of that class.
- **threshold t** = the share below which a class counts as rare. Here t = 0.1: a class found in fewer than
  1 in 10 training tiles gets oversampled.
- **repeat factor of a class** r = max(1, √(t / share)). A class above the threshold keeps r = 1.
  The square root makes the boost gentler than full balancing: a class 4× below the threshold gets r = 2, not 4.
- **repeat factor of a tile** = the largest r among the classes in that tile. A defect-free tile keeps 1.
- **copies**: a tile with r = 4.33 is listed 4 times, plus a 5th time with probability 0.33. The draw uses a fixed
  seed (0) and is made once, so the list is the same every epoch. LVIS redraws it every epoch; with a file list
  that is not possible.

How it is run: Ultralytics trains from a text file that lists image paths and keeps repeated lines. A tile listed
four times is loaded four times per epoch, each time with fresh augmentation (mosaic, flips, HSV, scale).
Nothing else in the recipe changes.

Measured on the film-split training set (143,918 tiles), t = 0.1. Ultralytics' own loader reproduced every count
on the box, 2026-10-08:

| class | train tiles | share | r | tiles seen per epoch | × |
|---|---:|---:|---:|---:|---:|
| porosity | 44,259 | 0.3075 | 1.00 | 46,247 | 1.05 |
| inclusion | 6,422 | 0.0446 | 1.50 | 9,748 | 1.52 |
| crack | 10,259 | 0.0713 | 1.18 | 12,287 | 1.20 |
| undercut | 766 | 0.0053 | 4.33 | 3,337 | 4.36 |
| lack_of_fusion | 5,172 | 0.0359 | 1.67 | 8,701 | 1.68 |
| lack_of_penetration | 15,042 | 0.1045 | 1.00 | 15,160 | 1.01 |
| **all tiles** | **143,918** | | | **155,065** | **1.077** |

Porosity and lack of penetration rise a little without being oversampled themselves. They share tiles with the
rare classes.

Other thresholds, same training set: t = 0.05 boosts undercut (×3.1), lack of fusion (×1.2) and inclusion
barely (×1.06), +2 % tiles.
t = 0.2 boosts every class except porosity (undercut ×6.1), +21 % tiles. t = 0.1 is the middle: it touches the
four rare classes and costs 7.7 % more images per epoch.

## The design: two YOLOv8m runs, identical except the sampling

YOLOv8m is the headline model of the paper reproduction (0.730 mAP50 on the tile split, `4bca601e…`).
YOLOv8n was considered and dropped (Bartu, 2026-10-08).

| | run A | run B |
|---|---|---|
| model | YOLOv8m, COCO weights | same |
| data | `swrd-paper-tiles 1.0.0` (`de772ad9…`), film split | same |
| recipe | 100 epochs, imgsz 640, effective batch 480 (24 per GPU × 4 GPUs × accumulate 5, as in `4bca601e…`), nbs 480, wd 0.00375, seed 0 | same |
| sampling | every training tile once per epoch | RFS t = 0.1 |
| hardware | multi-gpu agent, 4×T4, DDP | same |

**Why the film split and not the paper's tile split.** In the paper split, a val tile and a train tile are often
cut from the same film, overlapping each other by half. Oversampling shows those train tiles more often. A model
that memorises them then looks better on the overlapping val tiles, even if it has learnt nothing general.
The film split keeps every film entirely in train or entirely in val, so a gain has to be a real one.
Film split: `split_films.json` from `02_select_split.py` (seed 0). It holds 369 val exposures (15,996 tiles) and
shares no film with train. Its val set contains 177 undercut tiles (185 boxes), so undercut AP is not empty.

**Run A is also step 5.** The film-split retrain was already the next WP1 step (decision log 2026-10-06).
Comparing A with the tile-split run `4bca601e…` (0.730 mAP50) measures how much the paper split inflates the number.

## Commands (Bartu launches)

On the laptop, publish the branch to the box:
```bash
git push origin wp1-oversampling-rfs && git push box wp1-oversampling-rfs
```
On the box, check it out (the checkout is clean on `main`):
```bash
git -C ~/thesis fetch origin && git -C ~/thesis switch wp1-oversampling-rfs
```
Smoke test first: run B's exact settings, but 1 epoch on 2 % of the train list (about 15–20 min with validation):
```bash
cd ~/thesis/swrd_paper_baseline && AWS_PROFILE=data-rw uv run python scripts/05_train.py --dataset-id de772ad9363c4067bed5835e13a9be81 --split film --split-file ~/swrd_paper_baseline/data/work/split_films.json --rfs-threshold 0.1 --model yolov8m --epochs 1 --fraction 0.02 --emulate-paper-batch --batch 96 --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu --name smoke-film-rfs0.1-yolov8m
```
What to check in the smoke task:
1. *Execution → Source code*: no repository, the whole script stored. If a repository `/home/ec2-user/thesis.git`
   shows up, the agent cannot clone it and the run will fail.
2. *Configuration → film_split*: 369 `val_exposures`, `expected_val_tiles` 15996.
3. Console: `[data] split film: train 143,918 tiles, val 15996 tiles; train entries per epoch 155,065` and the class
   table above; Ultralytics' val scan shows 15,996 images.
4. *Artifacts*: `data_view` holds the same numbers.

Then run A and run B. The multi-gpu agent runs one task at a time, in the order they were queued, so B starts
when A finishes.

Run A (no oversampling), then run B (oversampling):
```bash
cd ~/thesis/swrd_paper_baseline && AWS_PROFILE=data-rw uv run python scripts/05_train.py --dataset-id de772ad9363c4067bed5835e13a9be81 --split film --split-file ~/swrd_paper_baseline/data/work/split_films.json --model yolov8m --epochs 100 --emulate-paper-batch --batch 96 --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu --name v1.0-film-yolov8m-100ep-paperbatch-4gpu
```
```bash
cd ~/thesis/swrd_paper_baseline && AWS_PROFILE=data-rw uv run python scripts/05_train.py --dataset-id de772ad9363c4067bed5835e13a9be81 --split film --split-file ~/swrd_paper_baseline/data/work/split_films.json --rfs-threshold 0.1 --model yolov8m --epochs 100 --emulate-paper-batch --batch 96 --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu --name v1.0-film-rfs0.1-yolov8m-100ep-paperbatch-4gpu
```
Duration: YOLOv8m on 4×T4 took 25 h for 100 epochs (`4bca601e…`). Expect about 25 h for A and 27 h for B
(B sees 8 % more images per epoch), so about 52 h, a little over 2 days, for both.

## How to read the result

From each run's `summary.json` (`best_pt_val`), compare B − A:
- **Main readout:** AP50 and AP50-95 of undercut, lack of fusion, inclusion and crack, the four oversampled classes.
- **Guard:** porosity and lack of penetration must not drop, and neither must overall mAP50 / mAP50-95.
- **Side readout:** A against the tile-split run `4bca601e…` gives the tile-split inflation (step 5).

Caveats to state with the number:
1. **One seed per arm.** Seed-to-seed noise has not been measured on this setup yet. A difference of a few
   hundredths on one class is not evidence on its own. If B looks better, run the same pair with `--seed 1`.
2. **B sees 7.7 % more images** over the 100 epochs (155,065 vs 143,918 per epoch); epochs and schedule are the
   same. The tile-split YOLOv8m curve peaked at epoch 77 of 100 and then fell, so extra length alone is unlikely
   to explain a gain.
3. **The extra copies are drawn once**, not per epoch (see the method).
4. **best.pt is chosen on the same val set it is scored on.** This applies equally to both arms; there is no test set
   in this protocol yet.
5. **Tiles overlap by half**, so one physical undercut sits in several tiles already. At ×4.4 the model sees each
   undercut many times per epoch. Over-fitting on undercut is a real risk; that is part of what the run measures.

Not in this run: Ultralytics 8.4.171 also offers a class-weighted loss (`cls_pw`). That re-weights the loss instead
of the sampling, a different technique and a possible later arm.

## Code

- `swrd_paper_baseline/scripts/05_train.py`: new flags `--split {tile,film}`, `--split-file`, `--rfs-threshold`.
  With none of them set, training is exactly as in the paper runs. Any non-default choice builds a symlink copy of
  the dataset layout next to the run (`runs/<name>_view`). The downloaded dataset is never written to.
  Also new: `Task.force_store_standalone_script()`, because the box checkout is now a git clone whose origin is a
  path the agent cannot reach.
- `swrd_paper_baseline/tests/test_train_sampling.py`: the RFS formula, the rounding, the film split and the view.
