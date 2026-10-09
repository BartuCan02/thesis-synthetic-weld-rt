# Synthetic run C: real rare-class defects inserted physically into other SWRD films

Branch `wp2-synthetic-physical-v1`, started 2026-10-09. Thesis WP2 (physics-based insertion), first training
test. Method from the WP2 trial (`experiments/wp2_physics/trial_log_compositing/`, review page shared with Felix).

## The question

Run A trains YOLOv8m on the film split with no rebalancing. Run B shows the rare-class tiles more often
(repeat factor sampling, `experiments/wp1_benchmark/oversampling_rfs.md`). Run C adds the same amount of
extra rare-class tiles as B, but each extra tile is a **new** image: a real SWRD defect moved into another
SWRD training film by physical insertion. Everything else is identical to run A.

| | run A | run B | run C (this folder) |
|---|---|---|---|
| training tiles | 143,918 real | 143,918 real, rare ones repeated | 143,918 real + about 11,300 synthetic |
| extra rare-class tiles per epoch | none | about 11,300 repeats | about 11,300 new images |
| everything else | YOLOv8m, film split, 100 epochs, paper batch, seed 0 | same | same |
| validation | 15,996 real tiles of 369 val exposures | same | same, byte-identical |

Readout per class on the real val set: C against A says whether moved real defects help at all; C against B
says whether a new image beats seeing the same real tile again.

## How many, and which classes

The four classes that RFS t = 0.1 boosts: inclusion, crack, undercut, lack of fusion. Porosity and lack of
penetration are frequent enough and get nothing. Per class, C adds as many training tiles as B adds per
epoch (`results/budget.json`, from `01_budget.py`, training films only):

| class | real train instances | real train tiles | RFS factor | extra tiles (= B) | tiles per instance | synthetic defects | usable real sources | uses per source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inclusion | 2,030 | 6,422 | 1.50 | 3,258 | 4.21 | 774 | 1,110 | 0.7 |
| crack | 1,821 | 10,259 | 1.18 | 1,967 | 6.59 | 299 | 749 | 0.4 |
| undercut | 199 | 766 | 4.33 | 2,554 | 4.22 | 606 | 158 | 3.8 |
| lack of fusion | 757 | 5,172 | 1.67 | 3,514 | 7.40 | 475 | 495 | 1.0 |
| **total** | | | | **11,293 (+7.8 %)** | | **2,154** | | |

"Tiles per instance" is measured on the real training tiles (boxes of the class / instances of the class), so
the synthetic defect count lands near the tile target. Each synthetic film carries up to 4 defects, so about
540 synthetic films. Undercut is the hard case: 158 usable real undercuts are each reused about 4 times, on
different films and positions, mirrored with probability 0.5. B shows each real undercut tile 4.3 times too.

## What a synthetic film is

A real SWRD training film (the host) with up to 4 real defects from other training films added to it:

1. **Remove** the source defect by inpainting and refill real grain beside it (Felix's removal, 2026-10-07).
2. **Defect alone** = original minus removed, smoothed 1 px, darkening only.
3. **Place** it where the host's brightness profile across the weld matches the source's, away from anything
   already labelled on the host by one tile side, and one tile side from the other inserted defects.
4. **Add** it, scaled by host grain / source grain. A host whose grain differs from the source's by more than
   2x is rejected for that defect, which tries another host.
5. The host's own labels are kept; the inserted polygons are added with `flags.synthetic = true`.

Only tiles that contain an inserted defect (under the v1.0 box rule) enter the dataset. Other tiles of a
synthetic film would only repeat the host. This matches B, which repeats positive tiles only.

### Exclusions (leakage and data quality)

- Sources and hosts come from **training exposures only** (not in `split_films.json` val), and never from the
  3 training exposures that are byte-identical copies of a val film.
- **158 of the 4,650 "16-bit" films are 8-bit images scaled by 257** (`00_grey_steps.py`). A smooth defect added
  to them would show finer grey steps than its surroundings, so they are excluded as sources and hosts.
- Tungsten inclusions (denser than steel), defects on a T-joint's crossing weld, defects cut by the film border,
  and elongated defects that do not run along the weld are not used as sources.
- Hosts are landscape films at least 2,000 px wide; portrait source films are turned 90 degrees for extraction.

## Steps and where things live

Code in this folder; data on the box under `~/swrd_synthetic_physical/data/` (not in git). The tiler, the
renderer and the training script are the baseline's own (`swrd_paper_baseline/scripts/`), called with the
parameters recorded for the v1.0 dataset, so tiles are built exactly like the real ones.

| step | script | output |
|---|---|---|
| 0 | `00_grey_steps.py` | `results/grey_steps.json` |
| 1 | `01_budget.py` | `results/budget.json` |
| 2 | `02_make_films.py` | `raw_physical_v1/` films, labels, `inserted.jsonl`, `make_films_report.json`, `qc_sheet.png` |
| 3 | baseline `00_inventory.py`, `01_tile.py` | `work_physical_v1/tiles.jsonl` |
| 4 | `03_select_tiles.py` | `work_physical_v1/split_tiles.json`, `selection_report.json` |
| 5 | baseline `03_render.py` | `yolo_physical_v1/` 8-bit tiles + labels |
| 6 | `04_merge_upload.py` | ClearML child dataset of v1.0 (`de772ad9…`), val unchanged |

All of steps 0 to 5 plus a dry run of step 6: `scripts/run_pipeline.sh physical v1`, on the box, from the
worktree `~/thesis-synth` (branch `wp2-synthetic-physical-v1`; `~/thesis` stays on `wp1-oversampling-rfs`).

## Training (Bartu launches)

Identical to run A (`29f71fe4…`) except `--dataset-id` and `--name`. Launch from `~/thesis/swrd_paper_baseline`,
the same checkout and `05_train.py` as run A. The multi-gpu agent runs one task at a time, so queue it after B.

```bash
cd ~/thesis/swrd_paper_baseline && AWS_PROFILE=data-rw uv run python scripts/05_train.py --dataset-id <SYNTH_DATASET_ID> --split film --split-file ~/swrd_paper_baseline/data/work/split_films.json --model yolov8m --epochs 100 --emulate-paper-batch --batch 96 --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu --name v1.0-film-synth-physical-v1-yolov8m-100ep-paperbatch-4gpu
```

Check in the task: Configuration → film_split shows 369 val exposures and `expected_val_tiles` 15996; the console
shows train = 143,918 + the synthetic tile count, val 15,996.

## The naive arm (later, for the WP4 A/B)

Same sources, flips, hosts and positions, alpha-blended instead of added:
`scripts/run_pipeline.sh naive v1`, then `04_merge_upload.py --arm naive`.

## Caveats to state with the result

1. One seed per arm, as for A and B.
2. These are real defects moved, not new defect shapes. Felix dropped cut-and-paste as an augmentation method
   (2026-09-15); this run is the physics step and the defect source for the A/B, to be confirmed with him.
3. Undercut sources are reused about 4 times each; overfitting on undercut is possible, as in B.
4. The extracted defect still carries some source grain (noise model is RQ3 work).
5. Sources were not filtered by visibility (unlike the trial's review page), so faint defects are included.

## Results

| date | ClearML | what | result |
|---|---|---|---|
| | | dataset | |
| | | run C | |
