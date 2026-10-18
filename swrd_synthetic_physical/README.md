# Synthetic insertion of real rare-class defects into other SWRD films (runs C and D)

Thesis WP2 (physics-based insertion). Version 2 of the pipeline built run C's data (2026-10-09, below). **Version 3**
(branch `wp2-synthetic-v3`, 2026-10-16, after the Felix call recorded in `docs/meetings/2026-10-16_felix.md`) builds the
data for the second physics run, run D. Everything below the version-3 section describes run C and still applies to
the parts version 3 keeps.

## Version 3 (run D): what changed and why

| | run C (version 2) | run D (version 3) |
|---|---|---|
| where across the weld | row with the best match of the source's brightness profile; no explicit rule, nothing checked | the defect's across-weld coordinate u is measured on the source (bead geometry: 0 = centre line, 1 = toe) and the defect goes to the host row with the same u, inside the host's main seam polygon, profile match >= 0.5 as a check; measured again on the synthetic films and plotted against the real distribution per class |
| the defect itself | the real residual, mirrored left-right with p = 0.5 | varied per use: size 0.7-1.4x, stretch 0.8-1.25x, rotation <= 5 deg (elongated classes; any angle for inclusions), depth 0.7-1.3x, left-right and top-bottom mirror (a top-bottom mirror also mirrors u) |
| depth lost to the 1 px smoothing | not corrected (synthetic cracks/inclusions about 20 % fainter than real in the 8-bit tiles) | measured per defect as median depth inside the polygon before/after the blur and put back (factor clipped to <= 1.6) |
| negatives with the pipeline's traces | none | null implants: on about half of the films the removal-and-refill step runs on one clean spot with nothing added; their tiles are negatives in training, 300 of them held out for the post-training check (`work_*/null_holdout.json`) |
| budget | matched to repeat factor sampling t = 0.1 (11,293 tiles, +7.8 %) | 3x that: 33,879 target tiles (+23.5 %), 6,462 synthetic defects, budget no longer tied to RFS (`results/budget_v3.json`) |
| defects per synthetic film | up to 4 | up to 5 |
| training | 100 epochs | 50 epochs (Bartu, 2026-10-16); needs the 50-epoch baseline A' for a fair readout |

Unchanged: physical insertion (removal, darkening-only residual, grain-ratio contrast scale 0.5-2x, raw16 additive),
sources and hosts from training films only, exclusions (8-bit films, val copies, tungsten, T-joint crossing welds,
borders), the version-2 rule (no inserted pixel on a tile the baseline trains on), tiling and rendering with the
baseline's v1.0 parameters, val byte-identical.

**Across-weld coordinate, measured on the real training films** (`05_measure_positions.py`, 26,488 polygons on 3,707
films, 98 % with a defined bead geometry; `results/positions_real.json`):

| class | n | median \|u\| | share inside the bead (\|u\| <= 1) | accepted source band [p2.5, p97.5] of \|u\| |
|---|---:|---:|---:|---|
| porosity | 21,588 | 0.44 | 83 % | 0.02-1.58 |
| inclusion | 1,732 | 0.58 | 73 % | 0.03-1.82 |
| crack | 1,661 | 0.36 | 74 % | 0.01-1.82 |
| undercut | 182 | **1.34** | **15 %** | 0.21-1.84 |
| lack of fusion | 668 | 0.53 | 83 % | 0.05-1.53 |
| lack of penetration | 657 | 0.28 | 97 % | 0.01-1.12 |

Undercut sits at or outside the toe, lack of penetration near the centre line: the measurement reproduces the
textbook positions, which is the answer to Felix's question. A source whose u is undefined (bead cut on both sides,
mostly T-joint halves) or outside its class band is placed by the version-2 profile match instead and flagged
`placement: profile` in `inserted.jsonl`.

**Budget per class (version 3, `01_budget.py --scale 3`):**

| class | synthetic defects | target tiles | usable real sources | uses per source |
|---|---:|---:|---:|---:|
| inclusion | 2,322 | 9,774 | 1,763 | 1.3 |
| crack | 897 | 5,901 | 1,665 | 0.5 |
| undercut | 1,818 | 7,662 | 185 | 9.8 |
| lack of fusion | 1,425 | 10,542 | 669 | 2.1 |
| **total** | **6,462** | **33,879** | | |

Undercut is reused about 10 times per source; each use is a different variation, host, position and side of the
bead. The real undercut distribution in SWRD (182 training instances) is the limit, not the pipeline.

**Build (on the box, worktree `~/thesis-synth-v3`, detached in tmux so it survives a closed laptop):**

```bash
tmux new-session -d -s v3 "cd ~/thesis-synth-v3/swrd_synthetic_physical/scripts && UPLOAD=1 bash run_pipeline.sh physical v3 > ~/swrd_synthetic_physical/data/pipeline_physical_v3.log 2>&1"
```

Steps: budget x3 -> real positions (cached) -> films -> tiles -> selection (+ null tiles) -> render -> automatic top-up
if a class is below 97 % of its tile target -> positions of the inserted defects, `position_check.png`,
`compare_sheet.png` (real | run C | v3) -> ClearML upload as `swrd-paper-tiles-plus-synthetic-physical` **3.0.0**
(child of v1.0 `de772ad9...`). The dataset id is the last `[merge] dataset id` line of the log. Expected: about 1,500
synthetic films, 32,000-35,000 positive tiles, 2,500-3,500 null tiles, 6-10 h of CPU.

**Training run D (Bartu launches; identical to run A except dataset, epochs and name):**

```bash
cd ~/thesis/swrd_paper_baseline && AWS_PROFILE=data-rw uv run python scripts/05_train.py --dataset-id <DATASET_ID_FROM_LOG> --split film --split-file ~/swrd_paper_baseline/data/work/split_films.json --model yolov8m --epochs 50 --emulate-paper-batch --batch 96 --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu --name v1.0-film-synth-physical-v3.0.0-yolov8m-50ep-paperbatch-4gpu
```

**50-epoch baseline A' (needed once, same command on the parent dataset):**

```bash
cd ~/thesis/swrd_paper_baseline && AWS_PROFILE=data-rw uv run python scripts/05_train.py --dataset-id de772ad9363c4067bed5835e13a9be81 --split film --split-file ~/swrd_paper_baseline/data/work/split_films.json --model yolov8m --epochs 50 --emulate-paper-batch --batch 96 --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu --name v1.0-film-real-yolov8m-50ep-paperbatch-4gpu
```

Run A (100 epochs) is not a fair reference for a 50-epoch run: Ultralytics' learning-rate schedule and the close-mosaic
phase scale with the epoch count, so a 50-epoch run is a different recipe, not a truncated one.

**Smoke build before the full launch (100 defects, 2026-10-16, `raw_physical_v3smoke`):** 89 placed after 10 rounds
(85 by geometry, 4 by profile fallback), 4.6 tiles per defect, 49 films, 11 null implants. Two findings:
(1) **run C had placed about a quarter of its undercuts inside the bead** (|u| < 1), where real undercuts almost never
are (real median |u| 1.34, run C 1.16, KS distance 0.30); v3 reproduces the real position (median 1.34, KS 0.14 on
19 samples). (2) The 1 px blur is not the main cause of run C's faint defects: the measured depth compensation is
1.0-1.1 (cracks ~1.09), so the faintness must come from elsewhere (per-tile stretch, or the amplitude of real-vs-
inserted contrast); the 0.7-1.3 amplitude variation now covers that range either way.

**Full build result (2026-10-17 00:57 UTC, 4 h 42 min on the box):** dataset `4e72b695729840018da24ca311013edd`
(`swrd-paper-tiles-plus-synthetic-physical` 3.0.0): 177,348 train files = 143,923 parent + 31,153 positive synthetic
tiles + 2,272 null tiles; val 15,991, byte-identical. 6,760 defects on 3,750 synthetic films (2,646 distinct hosts: some
hosts carry two synthetic films from the retry rounds, at different positions), 859 null implants, 300 null tiles held out
(`work_physical_v3/null_holdout.json`). Placement 5,245 by geometry, 504 by profile fallback. Per class, tiles of target:
inclusion 9,564 / 9,775, crack 5,216 / 5,901, undercut 7,110 / 7,663, lack of fusion 9,263 / 10,543 (88-98 %; one automatic
top-up pass). Contrast scale p5/p50/p95 0.61 / 1.03 / 1.71, depth compensation 1.01 / 1.04 / 1.12, profile match median 0.96.
Checks: `raw_physical_v3/position_check.png` (real vs run C vs v3 per class) and `compare_sheet.png`.

**Run D launched 2026-10-17 (Bartu's go-ahead, launched from the box):** ClearML task `1fd29082b4cc4022971a2cdc24742604`,
project `thesis_wp1_benchmark`, queue `multi-gpu`, 50 epochs, otherwise run A's arguments. The 50-epoch baseline A' is
not launched yet.

**Realism check with run A's detector (2026-10-18, before run D finished; Bartu's request).** Run A's best.pt
(model `d7ab854e…`) scored on the 33,425 synthetic v3 tiles alone (31,153 positives + 2,272 null tiles; eval task
`333273c13584458386579148e42c64f5`) and, with identical settings, on the real film-split val set (`0711262f…`):

| class | real val AP50 | synthetic AP50 | real val recall | synthetic recall | real val precision | synthetic precision |
|---|---:|---:|---:|---:|---:|---:|
| inclusion | 0.260 | 0.244 | 0.28 | **0.41** | 0.37 | 0.28 |
| crack | 0.573 | 0.325 | 0.55 | **0.58** | 0.60 | 0.24 |
| undercut | 0.101 | 0.286 | 0.08 | **0.14** | 0.44 | 0.63 |
| lack of fusion | 0.263 | 0.351 | 0.35 | **0.37** | 0.38 | 0.47 |

Recall is the comparable column (AP and precision depend on the class mix of the set: the synthetic set has no porosity
or lack-of-penetration tiles and no random negatives, so cross-class confusions weigh differently). A detector that never
saw a synthetic tile finds the synthetic defects of every class at least as often as the real val ones. So the synthetic
defects are plausible to the model; they are not a different-looking population. Caveats: the sources are training
defects A has seen in their original films (helps synthetic recall), and the host backgrounds are training films whose
neighbouring tiles A has seen. Reading: what limits run A on the rare classes is not the number of rare examples but
their appearance range: even the undercuts A trained on are found only 14 % of the time when moved, and the 16 val
films' undercuts 8 %. Copies of training defects, however well placed and varied, cannot add appearance that the
training set does not contain; that diversity has to come from a learned generator (WP3) or from more real films.

**Run D result (completed 2026-10-18 10:59 UTC, 15.2 h, task `1fd29082…`):** best.pt val mAP50 **0.424** / mAP50-95
**0.229** (curve best 0.429 at epoch 26, flat from epoch 25). Per-class AP50 against run A (100 epochs, 0.452 / 0.240):
porosity 0.743 vs 0.736, inclusion 0.240 vs 0.260, crack 0.544 vs 0.573, **undercut 0.041 vs 0.101**, lack of fusion 0.224
vs 0.263, lack of penetration 0.752 vs 0.783. Every boosted class is below A, undercut by most; worse than run C (0.437).
Caveat: 50 vs 100 epochs, the fair reference A' is not run. **Null-tile check** (best.pt at conf 0.25): 17 of 300 held-out
null tiles get a detection vs 13 of 300 trained null tiles and 10 of 300 real clean val tiles; the boxes are mostly
porosity (13 / 12 / 8), only 6 of the four inserted classes on the held-out set. The model did not learn the insertion
trace as a defect cue. Reading, with the realism check above: the synthetic defects are plausible and placement is
right, yet they lower the rare-class scores; the most likely mechanism is that 31,153 extra tiles of 2,500 known shapes
shift the model's class priors and box statistics toward the synthetic population without adding appearance it can use
on the val films.

**Run E (naive arm) launched automatically 2026-10-18 12:53 UTC:** dataset `4dadc0eda75147b3b881ec70241b83ad`
(`swrd-paper-tiles-plus-synthetic-naive` 3.0.0), task `16cd8721af2146ca84f0ee2f038dd2ea`, run D's arguments. Expected
end about 2026-10-19 04:00 UTC. D vs E on the identical val tiles = the physical-vs-naive A/B.

**Readout for run D:** per-class AP50 and mAP50 / mAP50-95 on the identical val tiles against A'; false calls per unit weld
length as the secondary readout (run C traded precision for recall and mAP hid it); detections on the 300 held-out
null tiles (any = insertion artefacts learned).

---

# Run C (version 2): real rare-class defects inserted physically into other SWRD films

Built 2026-10-09 (branches `wp2-synthetic-physical-v1`, then `wp2-synthetic-v2`, both merged into `main`). Thesis WP2 (physics-based insertion), first training
test. Method from the WP2 trial (`experiments/wp2_physics/trial_log_compositing/`, review page shared with Felix).

## The question

Run A trains YOLOv8m on the film split with no rebalancing. Run B shows the rare-class tiles more often
(repeat factor sampling, `experiments/wp1_benchmark/oversampling_rfs.md`). Run C adds the same amount of
extra rare-class tiles as B, but each extra tile is a **new** image: a real SWRD defect moved into another
SWRD training film by physical insertion. Everything else is identical to run A.

| | run A | run B | run C (this folder) |
|---|---|---|---|
| training tiles | 143,918 real | 143,918 real, rare ones repeated | 143,918 real + 11,256 synthetic |
| extra rare-class tiles per epoch | none | about 11,300 repeats | 11,256 new images |
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

"Tiles per instance" is measured on the real training tiles (boxes of the class / instances of the class). The
synthetic defects landed in fewer tiles than that (and 4 % found no valid spot), so a second pass topped every class
up to its tile target on hosts not used before (`02_make_films.py --topup-from`, least-used sources first).

**What was built (dataset `646e699eaa3246468b7f345c271c94c8`, version 2.0.1, 2026-10-12; replaces 2.0.0 `dc3907f3…`, see Results):**

| class | synthetic defects | distinct real sources | synthetic tiles | target tiles | reached |
|---|---:|---:|---:|---:|---:|
| inclusion | 859 | 859 | 3,270 | 3,258 | 100 % |
| crack | 317 | 317 | 1,912 | 1,967 | 97 % |
| undercut | 738 | 157 | 2,603 | 2,554 | 102 % |
| lack of fusion | 611 | 476 | 3,471 | 3,514 | 99 % |
| **total** | **2,525** | | **11,256** | **11,293** | **100 %** |

2,525 defects on 1,193 synthetic films, every film on a different host. Each kept tile holds exactly one inserted
class. Undercut is the hard case: 157 real undercuts reused about 4.7 times each, on different films and positions,
mirrored with probability 0.5 (B shows each real undercut tile 4.3 times). Contrast scale p5/p50/p95 = 0.60 / 1.03 /
1.67; placement profile match median 0.97. Reports and QC sheets: `results/`.

## What a synthetic film is

A real SWRD training film (the host) with up to 4 real defects from other training films added to it:

1. **Remove** the source defect by inpainting and refill real grain beside it (Felix's removal, 2026-10-07).
2. **Defect alone** = original minus removed, smoothed 1 px, darkening only.
3. **Place** it where the host's brightness profile across the weld matches the source's, away from anything
   already labelled on the host by one tile side, and one tile side from the other inserted defects.
4. **Add** it, scaled by host grain / source grain. A host whose grain differs from the source's by more than
   2x is rejected for that defect, which tries another host.
5. The host's own labels are kept; the inserted polygons are added with `flags.synthetic = true`.

**Version 2 rule (Bartu, 2026-10-09):** no inserted pixel may lie inside any tile the baseline already trains on
(its positives and its randomly sampled clean tiles). So the model never sees an inserted defect's spot clean, and
every synthetic tile shows background that is new to it. Checked after the build: 0 inserted defects touch a baseline
tile, and 0 kept tiles contain a real host defect. Version 1 (any clean spot) was stopped before upload.

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
| 2 | `02_make_films.py` | `raw_physical_v2/` films, labels, `inserted.jsonl`, `make_films_report.json`, `qc_sheet.png` |
| 3 | baseline `00_inventory.py`, `01_tile.py` | `work_physical_v2/tiles.jsonl` |
| 4 | `03_select_tiles.py` | `work_physical_v2/split_tiles.json`, `selection_report.json` |
| 5 | baseline `03_render.py` | `yolo_physical_v2/` 8-bit tiles + labels |
| 6 | `04_merge_upload.py` | ClearML child dataset of v1.0 (`de772ad9…`), val unchanged |

Steps 0 to 5 plus a dry run of step 6: `scripts/run_pipeline.sh physical v2`, then the top-up
`TOPUP=1 scripts/run_pipeline.sh physical v2`, on the box. Upload:
`04_merge_upload.py ... --arm physical --version 2.0.1`. About 3.5 h of CPU on the box in total.

## Training (Bartu launches)

Identical to run A (`29f71fe4…`) except `--dataset-id` and `--name`. Launch from `~/thesis/swrd_paper_baseline`;
`05_train.py` is the same on `main` and on `wp1-oversampling-rfs` (commit 40f53fb), so either checkout gives run A's code. The multi-gpu agent runs one task at a time, so queue it after B.

```bash
cd ~/thesis/swrd_paper_baseline && AWS_PROFILE=data-rw uv run python scripts/05_train.py --dataset-id 646e699eaa3246468b7f345c271c94c8 --split film --split-file ~/swrd_paper_baseline/data/work/split_films.json --model yolov8m --epochs 100 --emulate-paper-batch --batch 96 --devices 0,1,2,3 --workers 10 --seed 0 --queue multi-gpu --name v1.0-film-synth-physical-v2.0.1-yolov8m-100ep-paperbatch-4gpu
```

Check in the task: Configuration → film_split shows 369 val exposures and `expected_val_tiles` 15996; the console
shows `split film: train 155,174 tiles, val 15996 tiles` (143,918 real + 11,256 synthetic).

## The naive arm (run E, the WP4 A/B control)

**Started 2026-10-18 08:06 UTC on the box** (tmux `v3naive`, script `~/swrd_synthetic_physical/data/naive_chain.sh`, logs
`pipeline_naive_v3.log` and `naive_chain.log`): `UPLOAD=1 run_pipeline.sh naive v3`, i.e. the version-3 pipeline with the
same seed, sources, variations, hosts and positions as the physical build, alpha-blended instead of added; registered
as `swrd-paper-tiles-plus-synthetic-naive` 3.0.0. When the dataset id appears in the log the chain launches the training
run `v1.0-film-synth-naive-v3.0.0-yolov8m-50ep-paperbatch-4gpu` (run D's arguments) on `multi-gpu`, behind run D.
Readout: physical (run D) vs naive (run E) per class on the identical val tiles is the physical-vs-naive A/B of RQ3/WP4;
neither arm has to beat real-only for that comparison to hold.


Same sources, flips, hosts and positions, alpha-blended instead of added:
`scripts/run_pipeline.sh naive v2` (+ the same top-up), then `04_merge_upload.py --arm naive --version 1.0.0`.

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
| 2026-10-09 | `dc3907f320474cd1ab034ea7689b89b1` | dataset 2.0.0, **broken, do not use** (see below) | 155,179 train + 15,991 val files; 11,256 synthetic train tiles |
| 2026-10-11 | `67ab619187874451b5b11369e48779d0` | run C on 2.0.0 | **failed before training**: `Dataset merging failed`, 2 files missing |
| 2026-10-12 | `646e699eaa3246468b7f345c271c94c8` | dataset `swrd-paper-tiles-plus-synthetic-physical` 2.0.1 | same tiles; 2 labels rewritten; full local copy assembled on the box (155,179 train + 15,991 val) |
| 2026-10-13 | `7e2c90894a094766828910740f8a4670` | run C on 2.0.1, YOLOv8m, 100 epochs, seed 0 | best.pt val mAP50 **0.437** / mAP50-95 0.236 (A 0.452 / 0.240, B 0.450 / 0.239). Per class AP50: porosity 0.733, inclusion 0.266, crack 0.535, undercut 0.070, lack of fusion 0.243, lack of penetration 0.775. Three of the four boosted classes below A; no gain. Plateau precision 0.52 / recall 0.46 vs A 0.55 / 0.45. |

**Why 2.0.0 broke.** ClearML 2.1.12 de-duplicates by content: a new file whose SHA-256 equals any file of the parent
is stored as a reference to the parent's copy, but under the new file name. Two synthetic crack labels were
byte-identical to real crack labels by chance (both boxes span the full tile width, so only the rows matter, and the
rows matched). The agent then looked for the synthetic names inside the parent's folder and failed. Images were
never affected. `04_merge_upload.py` now hashes every synthetic file first: a matching image stops the upload, a
matching label gets a trailing zero on each number (same values, different bytes).
