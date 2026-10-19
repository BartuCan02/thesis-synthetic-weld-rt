# Decision log

Source of truth for scope. Newest entry at the bottom. Every entry: date, who decided, what, why.
When a decision here contradicts the proposal PDF or a memory note, this file wins.

## 2026-08-20 — Bartu — proposal v0.1
First full proposal draft (`docs/proposal/proposal_v0.1_2026-08-20.pdf`). Included cut-and-paste
augmentation, physics simulation, GAN and diffusion as candidate methods; customer data still in scope.

## 2026-09-15 (week of) — Felix — scope fixed
- SWRD only. Customer data excluded because Deeplify's customer label quality is still under audit.
- Cut-and-paste augmentation dropped as a method.
- Add inpainting augmentation of defects as the primary learned method.
- Mine medical imaging publications for transferable technique.
- Everything else from v0.1 stays.

## 2026-09-20 — Bartu — SWRD metadata verdict (question from Felix)
Felix asked whether SWRD carries the physics parameters. Answer: no. Verified in the dataset paper and in
the files on `s3://swdr/`. No kV, thickness, source-to-film distance; TIFF dpi is a Windows default.
Customer DICONDE headers do not rescue this (fields present, left empty by operators; only PixelSpacing is
real). Consequence: the physics formulation is made metadata-free by construction (log-space additivity,
sign constraint, smooth thickness profile, noise statistics) and the attenuation scale is calibrated from
the film itself (weld cap height, IQI wires and lead markers on uncropped originals).

## 2026-09-20 — Bartu — novelty claim rewritten
The v0.1 claim that physically grounded compositing was "unexplored" was false (Rogers 2016, Mery &
Katsaggelos 2017, Mery 2005, Pezeshk 2016, Robins 2017/2019). Replaced by the two-part gap: learned
generator + attenuation constraint has never been coupled; the physical-vs-naive compositing A/B on real
images has never been run (Bhowmik 2019 has only the naive arm). The A/B becomes RQ3's headline experiment.
Proposal v0.2 written (`docs/proposal/proposal_v0.2_2026-09-20.pdf`).

## 2026-09-21 — Bartu (relaying Felix) — inpainting framing corrected
Felix means inpainting as an augmentation technique to enlarge and rebalance the training set, not "a
generative model that synthesises only the masked region". RQ2 in proposal v0.2 still uses the old
mask-conditioned-generation wording and needs rewording in v0.3.

## 2026-09-21 — Cecilia — official dates and milestones
Start Mon 28 Sep 2026, submit Fri 26 Mar 2027, 26 weeks. Milestones: M1 23 Oct (benchmark frozen),
M2 4 Dec (physics gate), M3 15 Jan (inpainting gate), M4 29 Jan (hybrid gate, A/B launched),
M5 26 Feb (results frozen), M6 12 Mar (full draft), M7 26 Mar (submit). Fortnightly meetings.
WP6 (hybrid generator) is the designated cut if the schedule slips; dropping it keeps the physics-vs-naive
A/B, which is the core claim. Timeline at `docs/proposal/timeline_2026-09-21.pdf`.

## 2026-09-30 — Bartu — thesis gets its own repo
Thesis work moves out of the deeplify monorepo into this repo. Reason: SWRD-only and reproducible outside
Deeplify, and the proposal build scripts were lost with a session scratchpad. Only the PDFs survive; v0.3
must be rebuilt as source under `docs/proposal/` (not done yet). Deeplify's segmentation framework stays a
pinned dependency, not a copy.

## 2026-10-01 — Bartu — what the SWRD release actually contains; split-by-exposure rule for M1
*Rescued 2026-10-09 from an uncommitted worktree. Written by an earlier Claude session; not yet reviewed by Bartu.*
Checked the official download (`http://www.tz-ndt.com/#/download`): one archive `SWXD_Data.zip`,
115.86 GB, on Google Drive (folder `1LNUt101wufTBJpRAgrZAU1h-Tfx629wO`) and a Baidu mirror; the
"instruction" link is dead. Contents, reconstructed from Deeplify's processing scripts
(`deeplify/ml/training/weld_defect_detection/data_processing_swrd/`) and the S3 mirror:
- `Raw_data/images/*.tif` + `Raw_data/json/*.json`: the **uncropped originals**, 16-bit, with their own
  LabelMe polygons (e.g. `DJ-RT-20220623-22`). Mirrored at `s3://swdr/raw_tif_16_bit/`. These carry the
  IQIs and lead markers, so the physical-scale route is open. Earlier notes implying the originals were not
  released were wrong.
- `crop_weld_data/crop_weld_images/*.tif` + `crop_weld_jsons/*.json`: the 4,930 weld-cropped strips
  (median 6,943 x 717 px), 16-bit. Mirrored at `s3://swdr/cropped/`. This is what every dataset build uses.
- Not shipped: the 380k/161k sliding-window tiles, the 8-bit CLAHE images, the YOLO boxes. Tiling,
  negative ratio and preprocessing are ours to define and must be written into the M1 protocol.
- Unclear: a `download/` folder of `.DCM` files that one script inverts and rotates. Check whether it is
  SWRD or unrelated before touching polarity assumptions.

Consequence for M1: `weld_defect_segmentation/data_processing/create_dataset.py` splits with
`train_test_split` over the cropped image list (val_ratio 0.2, seed 42). T-joint crops `A_x`/`B_x` come
from the same exposure, so the March baseline's split leaks between train and val. The frozen benchmark
must group by original exposure (and keep all tiles of one exposure in one split). Re-score the baseline
on the leak-free split before using it as the reference number.

## 2026-10-02 — Bartu — WP1 starts by reproducing the SWRD paper's baseline, not Deeplify's
The first WP1 deliverable is a reproduction of the YOLOv8 detection baseline published with SWRD (Zhao et al. 2025,
Table 5: YOLOv8m mAP50 0.66265 / mAP50-95 0.44827), rebuilt from the raw release files (cropped 16-bit TIFF + LabelMe
JSON) in a self-contained folder `swrd_paper_baseline/` with its own `uv` project (Ultralytics). Nothing is reused from
Deeplify's derived datasets (SWRD v7) or its January SWRD scripts. The March-2026 Deeplify SegFormer run is kept as a
reference number only (facts verified today in `experiments/wp1_benchmark/deeplify_march_baseline_facts.md`); the
plan to re-train it was dropped. Paper protocol and its gaps: `docs/literature/zhao2025_swrd.md`. Step plan and the
open parameter decisions D1–D9: `swrd_paper_baseline/README.md` (awaiting approval).

## 2026-10-02 — Bartu — the 190 unpublished SWRD crops are reconstructed from the originals
The SWRD release (`SWXD_Data.zip`, 124,398,609,179 bytes; Deeplify's `s3://swdr/cropped/` is a faithful copy)
ships 4,740 weld-crop images but 4,930 label files: all 190 crops of `crop_weld_jsons/T/2/` (films
`DJ-RT-20240105-*`) have no image. The paper's 4,930 was the authors' local set. Decision: rebuild them from the
uncropped originals (`Raw_data/images`, present in the mirror) by recovering each crop's offset from the polygon
translation between the cropped and the original label file, then cutting the original TIFF. Validated on 80
released T-joint crops (40 A_, 40 B_): all 80 pixel-identical, no rotation. All 190 rebuilt (`rot90=0`, every
crop matched ≥ 1 polygon). Script `swrd_paper_baseline/scripts/reconstruct_missing_crops.py`; logs in
`swrd_paper_baseline/results/`. The 3,679 original-film label files were fetched from the archive by HTTP range
(`fetch_from_official_zip.py`), 10 MB. The authors are being asked to publish the images and the unstated
preprocessing parameters (drafts in `swrd_paper_baseline/results/`).

## 2026-10-05 — Felix (fortnightly meeting, relayed by Bartu) — paper preprocessing is the WP1 configuration; customer-data side run
*Rescued 2026-10-09 from an uncommitted worktree. Written by an earlier Claude session; not yet reviewed by Bartu.*
- The SWRD paper's own preprocessing (half-short-side tiles, 50 % overlap, per-tile contrast stretch → 8-bit → CLAHE,
  YOLOv8 detect, mAP50 on the paper's tile split) is the benchmark configuration. Thesis methods are compared against
  the paper's baseline in this configuration first; Deeplify's own configuration is an optional second arm.
- Side run, a diagnostic and not a scope change: same configuration, train on SWRD v1.0 tiles plus Deeplify's customer
  films tiled identically, evaluate on the unchanged SWRD val tiles. Question: does adding our data help or hurt?
  Felix expects worse (SWRD looks lab-clean). "SWRD only" for the thesis methods stands.
- Order of work after the baseline: physics route first, starting with porosity (segment the weld, insert simulated
  pores, metadata-free), compared against randomly inpainted defects; GAN and diffusion after. Class balance: no
  equalising, a minimum count per class; synthetic examples for every class over time. CAD-based RT simulation stays
  parked (needs the acquisition metadata SWRD lacks).

## 2026-10-06 — Bartu — SWRD paper baseline reproduced; numbers land above the paper
YOLOv8n and YOLOv8m trained on `swrd-paper-tiles 1.0.0` with the paper's effective batch (480 via Ultralytics `nbs`,
weight decay 0.00375): n 0.576 / 0.335, m 0.730 / 0.462 (mAP50 / mAP50-95, best epoch) vs the paper's 0.482 / 0.287
and 0.663 / 0.448. Ranking and peak epochs match; absolute numbers are higher. Accepted as the WP1 reference for
"what the paper's recipe gives on this data". Unexplained part of the gap (label-border rule, CLAHE/stretch
parameters, Ultralytics version) stays documented, not chased, unless the authors reply. Compute lesson: use all
4 GPUs of the multi-gpu agent (DDP finished cleanly; the old teardown hang did not occur). Next: film-level split
retrain of n (thesis protocol, step 5) and the full-coverage grid v1.1.

## 2026-10-07 — Felix (suggestion, not yet decided) — remove defects from SWRD films to make paired data
Felix removed a crack from an SWRD film with an image-editing tool and proposed: SWRD labels are accurate
enough to remove the defect first, then use the original defective film as the target for the generation
process, i.e. (defect-free, defective) pairs. Assessment so far: a masked-inpainting model does not need the
removal step for training (the mask hides the defect anyway), but the pairs serve three other purposes:
(1) clean host films at realistic positions for the physical-vs-naive A/B (only 353 films are natively
defect-free), (2) a supervised target for the defect-perturbation residual in log space, which is the
"learned generator + attenuation constraint" coupling of the novelty claim, (3) a paired held-out set to
score generators against a real defective reference. Risk: removal artefacts become a shortcut the model
learns. Check: run the March baseline / WP1 detector on removed films; residual detections = failed
removals. Removal must run on raw16, not on 8-bit exports. Reading plan: `docs/literature/wp2_physics_reading_plan.md`.

## 2026-10-08 — Bartu — WP1 tests oversampling of the rare classes, on YOLOv8m
Question: does showing the rare-class tiles more often raise their AP? It is the free rebalancing baseline that
synthetic rebalancing must beat (M1, classical baselines). Design: repeat factor sampling (Gupta et al.,
LVIS, CVPR 2019) with threshold t = 0.1, i.e. undercut ×4.4, lack of fusion ×1.7, inclusion ×1.5, crack ×1.2,
+7.7 % tiles per epoch. Film-level split (`split_films.json`); two YOLOv8m runs, identical except the sampling
(Bartu chose m only; an n pair was dropped). The run without oversampling is also the planned step-5 film-split
retrain and becomes the film-split YOLOv8m baseline that later detection results are compared against. Open for M1:
whether the thesis benchmark is detection (mAP) or segmentation (per-class IoU, as RQ4 in proposal v0.2 says); any
later baseline reuses the same `split_films.json`. `05_train.py --split` defaults to `film` (Bartu); the paper
protocol needs `--split tile`. Film split, not tile split, because oversampling repeats train tiles whose
half-overlapping neighbours sit in the tile-split val set. Plan and commands:
`experiments/wp1_benchmark/oversampling_rfs.md`; code on branch `wp1-oversampling-rfs`.
Found on the way: since the box checkout became a git clone, a launch from `~/thesis` records the repo
`/home/ec2-user/thesis.git`, which the agent cannot clone. `05_train.py` now calls `Task.force_store_standalone_script()`.

## 2026-10-08 — Bartu — the side run uses exactly the customer films of the latest Deeplify multiclass run
*Rescued 2026-10-09 from an uncommitted worktree. Written by an earlier Claude session; not yet reviewed by Bartu.*
The customer arm is pinned to the film set of `weld_defect_all_v2.12` (dataset `f40220b1…`, 2026-09-22): 1,183 OGE,
133 Aramco, 559 Maroca films, 1,076 of them label-free negatives, so that "adding our data" varies only the data and
not its labels or selection. A first build on the v3.0 film set (602 films, dataset `9b43a1ec…`) is superseded.
Per film: raw 16-bit variant, polarity canonicalised to SWRD's (verified on rendered panels for all three sources),
crop = seam bbox + 10 % of the seam width grown to hold every label; seam = human label when present, else the v1.3
seam prediction (cache or predicted at export time). Other weld-defect classes become `other_defect` polygons: never
a box, never a negative tile, unlabelled background inside positive tiles, exactly like SWRD's own extra labels.
Caveat recorded for the write-up: the Maroca defect labels in the catalogue on 22 Sept are the HQ project's model
prelabels before review; the v2.12 run trained on the same ones. Code: `swrd_paper_baseline/scripts/export_customer_films.py`,
`run_customer_pipeline.sh`, `07_merge_upload.py`.

## 2026-10-09 — Bartu — synthetic run C: physical insertion of real rare-class defects, matched to RFS
Third arm next to run A (film split, `29f71fe4…`) and run B (RFS t = 0.1): add, per rare class, as many extra
training tiles as RFS adds per epoch, but each one a new image, a real SWRD defect moved into another SWRD
training film with the WP2 trial's physical insertion (removal + subtraction + grain-scaled addition, placement by
the profile across the weld). Budget from the film-split training set: inclusion 774, crack 299, undercut 606,
lack of fusion 475 defects, about 11,300 tiles (+7.8 %), about 540 synthetic films. Only tiles that hold an inserted
defect are added; val stays byte-identical (child dataset of v1.0). Training identical to run A except the dataset.
Sources and hosts from training exposures only; the 3 training exposures that are byte copies of a val film and the
158 films that are 8-bit data stored as 16-bit are excluded. Code and plan: `swrd_synthetic_physical/` on branch
`wp2-synthetic-physical-v1`. Open with Felix: moving real defects is close to the cut-and-paste dropped on
2026-09-15; framed here as the physics step and the defect source for the physical-vs-naive A/B.
Update 2026-10-09 (Bartu): version 2 rule, no inserted defect on a spot the baseline already trains on. Built and
uploaded: ClearML dataset `dc3907f320474cd1ab034ea7689b89b1` (`swrd-paper-tiles-plus-synthetic-physical` 2.0.0,
child of v1.0): 2,525 defects on 1,193 films, 11,256 synthetic train tiles (inclusion 3,270, crack 1,912, undercut
2,603, lack of fusion 3,471; 97–102 % of the RFS-matched targets). Run C command in `swrd_synthetic_physical/README.md`.
Update 2026-10-12 (Bartu): run C `67ab6191…` failed before training because ClearML 2.1.12 content de-duplication
mis-filed 2 synthetic labels that were byte-identical to parent labels. Dataset re-uploaded as 2.0.1
`646e699eaa3246468b7f345c271c94c8` (same tiles, 2 labels made unique, upload script now checks); `dc3907f3…` must not be used.

## 2026-10-13 — Bartu — side run re-pinned to v2.13 and compared against the film-split baseline
Deeplify's latest multiclass run moved to `weld_defect_all_v2.13_pullv5` (2026-10-09, 2,613 customer films: v2.12's
1,875 plus 738 from a new Labelbox pull), so the customer arm uses that film set (dataset `1587e7d3…`, 44,202 train
tiles). The thesis baseline is now the film-level split (`29f71fe4…`, mAP50 0.453 / mAP50-95 0.240), so the customer run
is launched with `--split film --rfs-threshold 0.0` and the same arguments: task `bf130306…` on dataset `91dbe45c…` in ClearML project
`weld_defect_segmentation` (Bartu: the side run is Deeplify's question, so it lives next to the Deeplify runs; `05_train.py`
got a `--project` option for that). Two earlier attempts failed at dataset download because ClearML de-duplicates child
files against the parent by content (the empty labels of clean tiles); `07_merge_upload.py` now rewrites such labels, see
`experiments/wp1_benchmark/README.md`.
Readout: best.pt mAP50 / mAP50-95 and per-class AP50 on the identical SWRD film-split val tiles. Customer labels are the
catalogue's newest as of the export (same as the v2.13 pull unless re-pulled since); 26 films whose newest label has no
six-class defect were skipped.
Update 2026-10-13 (Bartu): run C `7e2c90894a094766828910740f8a4670` completed. best.pt val mAP50 0.437 / mAP50-95
0.236 against A 0.452 / 0.240 and B 0.450 / 0.239. Undercut 0.070 (A 0.101), crack 0.535 (A 0.573), lack of fusion
0.243 (A 0.263), inclusion 0.266 (A 0.260). Neither repeating real rare tiles (B) nor moving real rare defects onto new
films (C) helped on this benchmark; C trades precision for recall. One seed per arm. Per-class montages of val
predictions registered on A and B as debug samples "Validation by class" (`~/debug_pairs/class_montage.py` on the box).
Update 2026-10-15 (Bartu): side run `bf130306…` completed. best.pt mAP50 0.466 / mAP50-95 0.249 vs A 0.452 / 0.240; undercut
0.139 (A 0.101), lack of fusion 0.299 (A 0.263), crack 0.558 (A 0.573), others within 0.01. Customer films helped slightly,
against Felix's expectation. Diagnostic only; SWRD-only scope unchanged. Details in `experiments/wp1_benchmark/README.md`.

## 2026-10-15 — Bartu — result of the customer-data side run (Felix's question)
SWRD + v2.13 customer films, paper configuration, film split: best mAP50 0.466 / mAP50-95 0.250 against the SWRD-only
film-split baseline 0.453 / 0.240 (task `bf130306…` vs `29f71fe4…`, identical arguments, identical val tiles). Adding
our data helped a little rather than hurting: undercut 0.10 → 0.14, lack of fusion 0.26 → 0.30, porosity 0.74 → 0.75,
inclusion and lack of penetration unchanged, crack 0.57 → 0.56. Single seed; identical-data runs differ by about 0.004,
so the +0.013 is small but real. This does not say whether the paper's preprocessing helps: that needs the same model
and films with the photometry varied (proposed: YOLOv8n, 40 epochs, SWRD only, per-tile stretch+CLAHE vs per-crop
window; about 6 GPU hours), not a comparison with the gen-2 SegFormer runs, which differ in model, task, metric and val set.

## 2026-10-15 — Bartu — preprocessing comparison on Deeplify's own model
After the Felix run, Bartu wants to know whether the paper's preprocessing is better than Deeplify's gen-2
preprocessing, with the model held fixed. Decision: train Deeplify's SegFormer (framework recipe, 9 classes incl.
other_defect and excess_material, v2.13's film split and validation films) on the paper-preprocessed tiles, and compare
with `weld_defect_all_v2.13_pullv5` by the framework's IoU on the same validation films. Segmentation, not detection:
SegFormer has no box head. 30 epochs (Bartu). Dataset `8a6aa755…`, task `da522e12…` in `weld_defect_segmentation`.
The export now emits other_defect and excess_material separately so the 9-class masks match v2.13's classes.

## 2026-10-16 — Bartu — segmentation vs detection on the same model family: one YOLOv8m-seg run
Bartu asked whether segmentation beats detection, without the March SegFormer and without a new eval script. There is no
run-free answer: a metric needs predictions, and the detector has no masks to score. Cheapest fair test: YOLOv8m-seg on
baseline A's tiles, film split, recipe and val tiles, compared by box mAP50 on the identical val tiles (the seg model
reports box and mask mAP together); mask mAP50 becomes the benchmark's first segmentation number. Built: polygon labels
for the same 151,051 D7 instances on v1.0's byte-identical PNGs (`03_render.py --labels polygon --reuse-images`), D7 boxes
re-derived and checked byte for byte; dataset `c385c61a95bd40f191ecd8ca60894f7e` (`swrd-paper-tiles-seg` 1.0.0); `05_train.py` takes
`--model yolov8m-seg`. Label geometry: traced boundary of the rasterised visible piece, multi-piece instances bridged
(607, 0.4 %), implied box 1 px smaller than D7 (inclusive vs exclusive edge). Memory probe on the box: 12.4 GB per T4 at
batch 24, so A's `--batch 96` stands. Command in `experiments/wp1_benchmark/README.md`. Bartu launches.

## 2026-10-16 — Felix (call) — second physics run: vary the inserted defect, make placement explicit and measured
Felix on run C: (1) a pasted defect is the same defect again; rotate, scale and stretch it when inserting. (2) He asked
how the pipeline guarantees each class sits where it occurs (undercut at the bead edge) and whether placement is per
tile or per film. Answer after checking the code: placement is per film (tiles cut afterwards by the baseline tiler);
the position across the weld is copied from the source by matching the brightness profile across the weld (median
score 0.975, 5th percentile 0.72), with no explicit per-class rule, no check against the host's seam polygon, and no
verification of the outcome. (3) Two papers: Tyystjärvi 2022 (virtual flaws in weld RT: affine + noise + flips on the
extracted flaw, placed inside a weld mask; combined real+virtual best at every data fraction, gain only below ~25 % of
the data and for the near-empty crack class, nothing at full data) and Virkkunen 2021 (origin of the method, UT:
amplitude scaling to emulate other sizes, null implants so the model cannot learn the insertion itself). Notes in
`docs/literature/`, meeting note `docs/meetings/2026-10-16_felix.md`.
Plan for the second physics run (to confirm with Felix): per-class position measured on the real labels as a
normalised across-weld coordinate and sampled from it, inside the host seam, with a real-vs-synthetic plot per class;
affine + amplitude variation of the residual (scale 0.7–1.4, stretch 0.8–1.25, ±5° for elongated classes, amplitude
0.7–1.3); null implants; faintness compensated; scored on full data with a synthetic share well above run C's 7.8 %
(Bartu: no data-fraction sweep; about 3× run C as the starting budget, no longer tied to RFS), with false calls per
weld length as a secondary readout. No seed repeat (Bartu): the
~0.004 identical-data estimate stands unless a result lands inside the ~0.02 band.
Run C's full-data no-gain result is consistent with Tyystjärvi's 100 % column and is no longer surprising.

## 2026-10-16 — Bartu — run D: version-3 synthetic build launched (full data, 3x budget, 50 epochs)
After the Felix call (entry above), the second physics run is built with pipeline version 3 on branch
`wp2-synthetic-v3` (`swrd_synthetic_physical/README.md`, top section). Decisions: full SWRD training data plus a
synthetic share 3x run C's (33,879 target tiles, +23.5 %, 6,462 defects; undercut reused ~10x per source); defects
varied per use (size 0.7–1.4x, stretch 0.8–1.25x, rotation <= 5 deg for elongated classes, depth 0.7–1.3x, both
mirrors); placement by the measured across-weld coordinate u of the source, inside the host seam, profile match >= 0.5
as a check; depth lost to the 1 px blur put back (<= 1.6x); null implants on about half of the films (their tiles
train as negatives, 300 held out for the artefact check); 50 epochs (Bartu, time), which requires a 50-epoch
baseline A' on the parent dataset for a fair readout (run A's 100-epoch schedule is a different recipe).
Measured on the real training films (26,488 polygons): undercut median |u| 1.34 with 15 % inside the bead, lack of
penetration median 0.28 with 97 % inside, i.e. the coordinate reproduces the textbook positions; this is the answer to
Felix's placement question, and `position_check.png` compares real vs run C vs v3 per class.
Build runs detached on the box (tmux session `v3`, log `~/swrd_synthetic_physical/data/pipeline_physical_v3.log`),
uploads as dataset version 3.0.0 at the end. Training commands in the README; Bartu launches.

## 2026-10-17 — Bartu — run D launched on the version-3 dataset
Build finished 00:57 UTC: dataset `4e72b695729840018da24ca311013edd` (3.0.0), 31,153 positive synthetic tiles (+21.6 %)
and 2,272 null tiles on top of the unchanged v1.0 parent; 6,760 defects on 3,750 films, 859 null implants, 300 null
tiles held out. Position check on the full build is in `raw_physical_v3/position_check.png`. Run D = YOLOv8m, film split,
50 epochs, run A's other arguments: task `1fd29082b4cc4022971a2cdc24742604` on `multi-gpu`, launched by Claude on Bartu's
instruction. Open: the 50-epoch baseline A' on the parent dataset (needed for the readout; command in the README).

## 2026-10-18 — Bartu — realism check: run A's detector finds synthetic v3 defects as easily as real ones
Run A's weights on the 33,425 synthetic v3 tiles vs the real film-split val set (eval tasks `333273c1…`, `0711262f…`):
recall inclusion 0.41 vs 0.28, crack 0.58 vs 0.55, undercut 0.14 vs 0.08, lack of fusion 0.37 vs 0.35. The synthetic
defects are plausible to a model that never saw them; they are not a different-looking population. Together with run C
(no gain), RFS (no gain) and run D's curve (plateau at mAP50 0.43 at epoch 40 of 50, below A's 0.452), the reading is that
more examples of the same rare-class shapes do not help this detector; the missing appearance range must come from a
generator that produces new shapes (WP3) or from more real films (the customer side run gained +0.013). Run D still
running; its per-class table and null-tile check are read when it ends. Table and caveats in
`swrd_synthetic_physical/README.md`.

## 2026-10-18 — Bartu — physics track next steps; naive arm (run E) building and chained to training
After the realism check, Bartu asked what makes the physics part contribute. Decision: bank the A/B first, then make
the physics step add appearance range rather than examples. (1) Naive arm started on the box with the version-3
pipeline (same seed, sources, variations, hosts, positions; alpha blend), uploads as `synthetic-naive` 3.0.0 and launches
its 50-epoch run on `multi-gpu` behind run D automatically (`naive_chain.sh`). Run D vs run E on identical val tiles is the
physical-vs-naive A/B. (2) Proposed next physics build: contrast-to-noise sweep with half the insertions near the
detection threshold (CNR 1-2), per-use unsharpness 0-2 px, host-grain-matched noise on the defect (RQ3 noise model).
(3) Proposed WP2 content: procedural thickness maps per class (pore = sphere chord, crack = thin curve, undercut =
groove at the toe, lack of fusion = planar gap) converted with a grey-per-pixel-of-steel scale fitted on real pore dips,
placed by the across-weld coordinate: new shapes under an attenuation constraint, the clean form of the novelty claim.
(4) Report recall at a fixed false-call rate per weld length next to AP for the rare classes. Not started: (2)-(4).

## 2026-10-18 — Bartu — run D result: no gain, loss on the boosted classes; run E (naive) running
Run D (physical v3, 50 epochs, task `1fd29082…`): best.pt mAP50 0.424 / mAP50-95 0.229 vs A 0.452 / 0.240 (100 epochs,
fair A' not run) and C 0.437. AP50 undercut 0.041 vs 0.101, crack 0.544 vs 0.573, lack of fusion 0.224 vs 0.263, inclusion
0.240 vs 0.260; porosity and lack of penetration within 0.03. Null-tile check: 17/300 held-out null tiles get a detection
vs 10/300 real clean val tiles, mostly porosity boxes; the insertion trace was not learned as a defect. With the realism
check (synthetic defects found as often as real by A), the conclusion stands: copies of known shapes, however well
inserted, do not add usable appearance and at a 22 % share they pull the rare-class scores down. Run E (naive arm,
dataset `4dadc0ed…`, task `16cd8721…`) started 12:53 UTC on the same recipe; D vs E is the physical-vs-naive A/B.
Next physics work is the appearance-range list in the 2026-10-18 entry above (CNR sweep, unsharpness, grain, procedural
thickness maps); the copy-based v3 build is the last of its kind.

## 2026-10-18 — Bartu — 50-epoch baseline A' queued
Run A's recipe on the parent dataset `de772ad9…` with 50 epochs, task `b1357e6011544643b07389cdc95297ea`
(`v1.0-film-real-yolov8m-50ep-paperbatch-4gpu`), queued on `multi-gpu` behind run E. It is the fair reference for runs
D and E, which both train 50 epochs. Expected start about 04:00 UTC on 19 October, end about 19:00 UTC.

## 2026-10-18 — Bartu — debug samples of runs C and D repaired; E and A' repaired automatically when done
The multi-gpu agent still registers debug images at the file server's private address `172.31.21.134:8081`, which a
browser cannot open (agent clearml.conf, Felix's to change). Runs C `7e2c9089…` and D `1fd29082…` re-pointed to the live
host with `~/testset_prep/rehost_all.py --only=<id> --apply`: 9/9 links each, all load. Run E `16cd8721…` and A' `b1357e60…`
are repaired by a waiter on the box (tmux `rehost`, `~/testset_prep/rehost_when_done.py`, log `rehost_when_done.log`)
that polls their status every 10 minutes and runs the same repair when each finishes. Every later agent run needs the
same step until the agent config is fixed.

## 2026-10-19 — Bartu — run E result: the naive arm beats the physical arm
Run E (naive alpha blend, same seed, sources, variations, hosts and positions as run D, 50 epochs): best.pt mAP50 0.459 /
mAP50-95 0.246 vs run D (physical) 0.424 / 0.229 and run A (real, 100 epochs) 0.452 / 0.240. AP50 E / D: undercut 0.132 /
0.041, lack of fusion 0.273 / 0.224, inclusion 0.265 / 0.240, crack 0.568 / 0.544; porosity and lack of penetration within
0.03. The A/B at this setup favours naive blending by 0.035 mAP50, on every boosted class, one seed each. This contradicts
the thesis hypothesis and has to be explained before it is written up; the fair 50-epoch real-only reference A'
(`b1357e60…`) ends about 17:30 UTC today and decides whether naive also beats real-only. Candidate explanations to test,
none checked yet: the physical residual loses appearance (darkening-only clip drops bright rims, the 1 px blur and the
grain-ratio scale weaken edges and contrast; run C's defects were already ~20 % fainter in the 8-bit tiles), while the
naive paste keeps the source defect's full local appearance; and a second seed of D and E to bound the gap.

## 2026-10-19 — Bartu — why E beat D: the naive pastes are gross, part of the physical insertions are near-invisible
Following single source defects through the builds as 8-bit training tiles (`swrd_synthetic_physical/scripts/08_follow_source.py`,
three per class, seed 5): run E's naive pastes carry the source film's grey level, SWRD films differ in level up to ~10x,
so after the per-tile stretch most pastes are solid black shapes or white patches with crisp outlines. The same
insertions in run D are often faint and some are near-invisible (depth factor down to 0.7 times grain scaling). So E
trained on conspicuous class silhouettes, D partly on boxes around nothing visible, which acts like label noise. The D vs E
gap is therefore not yet the physical-vs-naive question of the thesis: run E is a far cruder paste than the literature's
naive arm, and run D's contrast is too low on part of its defects. Before the A/B is rerun: (1) physical arm: reject or
rescale insertions whose contrast-to-noise ratio in the tile is below that of real defects of the class; (2) naive arm:
match the pasted crop's mean level to the host before blending (the usual naive baseline). Page with the sheets shared in
the session; not yet measured over the whole build.

## 2026-10-19 — Bartu — correction: runs D and E are matched in distribution, not defect by defect
Earlier entries say run E has "the same sources, variations, hosts and positions" as run D. Measured on the two
`inserted.jsonl` files: 6,760 (D) vs 6,767 (E) defects; 80 % of the defects (5,427) come from the same source, but only 47 %
(3,162) sit on the same film at the same spot. The first insertion on each film is identical; later spots are chosen on
the film as it looks after earlier insertions, which the two methods change differently. Tiles per class agree within
2 % (D 9,564 / 5,216 / 7,110 / 9,263, E 9,657 / 5,193 / 7,010 / 9,227 for inclusion / crack / undercut / lack of fusion).
Visual comparison on matched tiles (Cecilia deck, 19 Oct): many physical defects are faint after the per-tile stretch,
some nearly invisible; the naive pastes are strong and often sit in a patch of different brightness. Working hypothesis
for D < E: faint physical insertions act as label noise. Next check: contrast and edge sharpness of real, physical and
naive defects in the 8-bit tiles.

## 2026-10-19 — Bartu — null-implant check for run E
Run E's best.pt at conf 0.25: 32 of its 300 held-out null tiles get a detection, 22 of 300 trained null tiles, 17 of 300
real clean val tiles (a different random draw from the same pool than the run D check). Run D's model on run E's held-out
null tiles: 18 of 300. Boxes on E's held-out nulls are mostly porosity (22), 9 of the inserted classes. Reading: both models
fire on processed clean spots about 1.7-1.9 times as often as on real clean tiles (D 17 vs 10, E 32 vs 17); for E the
excess is borderline significant on 300 tiles, for D it is not. Run E also fires more on real clean tiles than run D,
consistent with a higher detection rate overall. Neither model learned the processing trace as a strong defect cue.
Held-out null tiles of the naive build rendered at `~/swrd_synthetic_physical/data/yolo_naive_v3_nullholdout`.

## 2026-10-19 — Bartu — realism check on run E's tiles: run A's model misses the naive pastes
Run A's best.pt on the 33,317 synthetic tiles of the naive v3 build (eval task `49b729257b6845859352637959553857`), same
settings as the run D check. Recall real val / physical v3 (run D tiles) / naive v3 (run E tiles): inclusion 0.28 / 0.41 /
0.12, crack 0.55 / 0.58 / 0.26, undercut 0.08 / 0.14 / 0.03, lack of fusion 0.35 / 0.37 / 0.24. A model trained on real
films finds the naive pastes about half as often as real defects: to it they look different, as the sheets showed
(solid black or white silhouettes). Yet training on them gave the higher validation score (E 0.459 vs D 0.424). So E's
synthetic defects are less realistic and still more useful than D's. Most likely reading: D's near-invisible
insertions hurt (boxes around nothing visible), E's conspicuous ones do not; whether E actually helps over real-only
waits for A'.

## 2026-10-19 — Bartu — compared arms get exactly equal training data counts
Runs B and C differed by 109 extra tiles (11,147 vs 11,256) and D and E by 108 (33,425 vs 33,317; positives 31,153 vs
31,087, null tiles 2,272 vs 2,230), because neither repeat sampling nor defect insertion can hit a tile target exactly.
The effect is far below seed noise (0.06–0.07 % of an epoch vs ~0.004 mAP50 between identical-data runs), but Bartu
decided that from now on every pair of arms that is compared gets exactly the same counts, as a matter of scientific
rigour: same number of synthetic tiles per class, same number of null tiles, same total. How: after both builds,
randomly drop the surplus tiles of the larger arm per class (seeded, recorded in the dataset metadata) before upload.
For a repeat-sampling arm, fix its copy count per class to the synthetic arm's tile count instead of drawing it.
Runs B–E stay as they are and are reported with their actual counts.
