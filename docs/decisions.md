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
retrain. Film split, not tile split, because oversampling repeats train tiles whose
half-overlapping neighbours sit in the tile-split val set. Plan and commands:
`experiments/wp1_benchmark/oversampling_rfs.md`; code on branch `wp1-oversampling-rfs`.
Found on the way: since the box checkout became a git clone, a launch from `~/thesis` records the repo
`/home/ec2-user/thesis.git`, which the agent cannot clone. `05_train.py` now calls `Task.force_store_standalone_script()`.
