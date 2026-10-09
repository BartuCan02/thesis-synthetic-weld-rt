# CLAUDE.md — Master's thesis: synthetic defective weld radiographs

This directory is Bartu Can's master's thesis. It is a standalone research repo, separate from the
Deeplify product monorepo at `~/Desktop/Deeplify/deeplify`. Open thesis sessions here, not there.

**Title:** Realistic Synthetic Data Generation for Defective Radiographic Testing Scans of Welds
**Where:** Deeplify GmbH (industrial) + TUM (academic)
**Dates:** start Mon 28 Sep 2026, submission Fri 30 Mar 2027 (26 weeks). Fortnightly supervisor meetings.

## Read first, every session

1. `docs/decisions.md` — the dated decision log. It is the source of truth for scope. It overrides the
   proposal PDF and overrides anything in memory.
2. `docs/proposal/proposal_v0.2_2026-09-20.pdf` — the current proposal. **It is v0.2, not final.** Scope and
   methodology can still change; check `docs/decisions.md` for anything newer.
3. `experiments/README.md` — which runs exist, their ClearML ids, and what each one showed.

At the end of a session that changed scope, results, or plans, append to `docs/decisions.md` or the
relevant `experiments/<wp>/README.md`. Do not rely on memory alone to carry a fact into the next session.

## People

- **Felix** = industrial supervisor at Deeplify. Scope decisions so far came from him.
- **Cecilia** (she/her) = university supervisor at TUM. Dates and milestones were agreed with her.
- "Supervisor" alone is ambiguous. Always say which one.

## Fixed scope (from `docs/decisions.md`, week of 2026-09-15)

- **SWRD only.** No customer data (customer label quality is still under audit at Deeplify).
- **No cut-and-paste augmentation.** Dropped as a method; the literature supports dropping it.
- **Inpainting is an augmentation technique**, a way to enlarge and rebalance the training set. Do not
  describe it as an architecture choice or contrast it with "generating whole radiographs".
- **Mine medical imaging papers** for transferable technique (lesion insertion, lesion-focused diffusion).
- **The headline experiment is the A/B:** identical model trained on physically composited synthetic data
  versus naively blended synthetic data, scored on real SWRD film. Nobody has published this comparison.
- **WP6 (hybrid generator) is the designated cut** if the schedule slips.

## Research questions (proposal v0.2)

- RQ1 Which paradigm (physics simulation, GAN, mask-conditioned diffusion inpainting) gives the most useful
  synthetic weld defects under one common evaluation?
- RQ2 Can class- and mask-conditioned inpainting inject convincing defects into real defect-free SWRD film,
  using the seam polygons as placement prior, with an exact mask? (Wording to be revised: see decisions.)
- RQ3 Do physics-informed constraints (log-space additivity, sign constraint, thickness-perturbation space,
  photon and grain noise) improve realism and utility when the attenuation scale is calibrated from the image?
- RQ4 Does it help downstream? Per-class IoU and instance detection rate on a fixed real SWRD validation
  set against the real-only baseline, swept over synthetic fractions. Secondary: FID/KID, blinded inspector study.

## Milestones (all Fridays)

| | date | gate |
|---|---|---|
| M1 | 23 Oct 2026 | benchmark frozen (real-only + classical-augmentation baselines, eval protocol) |
| M2 | 4 Dec 2026 | physics gate |
| M3 | 15 Jan 2027 | inpainting gate |
| M4 | 29 Jan 2027 | hybrid gate, A/B launched |
| M5 | 26 Feb 2027 | results frozen |
| M6 | 12 Mar 2027 | full draft |
| M7 | 26 Mar 2027 | submit |

## The baseline every result is compared against

March-2026 7-class SegFormer mit_b4, trained on SWRD v7 only. ClearML model `ce0108c617864f839760d981c8d1eb29`,
training run `512x512_bs64` (`ef3b7375…`), eval task `6a0a0ac812774b568718b4d7c46363b7`, project
`weld_defect_segmentation`, dataset `be5ca720d8af4eb7a911e863c9989ad7`.

| class | val IoU |
|---|---|
| macro over 6 defect classes | **0.4838** |
| porosity | 0.6206 |
| inclusion | 0.2389 |
| crack | 0.7415 |
| undercut | 0.2964 |
| lack of fusion | 0.4149 |
| lack of penetration | 0.7295 |

Caveat that matters for the thesis: the framework's `val_iou` uses `ignore_index=0`, so false positives on
background are invisible to it. On multiclass runs the metric is diluted, not broken, and comparisons to the
baseline stay valid because the baseline used the same metric. **True IoU must be computed post-hoc** and the
thesis should report the post-hoc number.

## SWRD facts (measured on all 4,930 label files, do not re-derive)

- 3,675 original films → 4,930 cropped weld images (L/ 2,422 standard, T/ 2,508 T-joints split A_/B_). 16-bit TIFF.
- Every image has a weld-seam polygon. 353 images are annotated defect-free. 4,577 have ≥1 defect.
- 9 defect classes + seam + pseudo-defect (13,311 instances on 2,664 films) in the files; the paper advertises 6.
- Undercut: 228 instances / 132 films. Porosity: 26,674 / 3,359. Median polygon = 0.024% of a film.
- **No acquisition metadata.** No kV, no thickness, no source-to-film distance, no real dpi (TIFF says 96 dpi,
  Windows default). Switching to customer DICONDE does not fix this: those header fields exist but are empty.
- Uncropped originals carry wire IQIs and lead markers (route to physical scale). The cropped release removes them.
- Polarity: more metal → higher pixel value, pores are dark. Single raw16 variant, never inverted.
- 158 of the 4,650 uint16 crops are 8-bit data scaled by 257 (grey values on a 257 grid; measured 2026-10-09,
  `swrd_synthetic_physical/results/grey_steps.json`). Raw-value analyses must exclude or flag them.
- Data: `s3://swdr/` (bucket name has the typo). Paper: Zhao, Wu et al., J. Nondestructive Evaluation 2025,
  doi:10.1007/s10921-025-01186-w.

## Novelty claim (verified by literature sweep 2026-09-20; do not overclaim)

Physically grounded compositing into real transmission images is established in three communities:
security screening (Rogers 2016, Mery & Katsaggelos 2017), NDT (Mery 2005, casting and weld radiographs),
medicine (Pezeshk 2016, Robins 2017/2019). The defensible gap is two-part: (1) nobody has coupled a learned
defect-appearance generator with an attenuation constraint; (2) the physically-composited vs naively-blended
A/B on real images has never been run (Bhowmik 2019 supplies only the naive arm). Closest competitor:
Cheng et al., J. Manufacturing Processes 2026, doi:10.1016/j.jmapro.2026.03.047.

Also load-bearing: a raw grey value is not a transmission. Mery & Katsaggelos calibrate offset and gain
before taking logs; replicate that.

## Compute and tooling

- Training uses Deeplify's segmentation framework at `deeplify/ml/training/segmentation` (SegFormer/UNet,
  Lightning, ClearML). Pin the deeplify commit used in `experiments/README.md`; do not copy the framework here.
- SWRD preprocessing scripts already exist at `deeplify/ml/training/weld_defect_detection/data_processing_swrd/`.
- Runs go to the ClearML server via the data-management EC2 box (see memory: data-mgmt-ec2-access). The repo is
  checked out on the box at `~/thesis` (`main`), tracking the bare repo `~/thesis.git` there (remote `box` on the
  laptop). GitHub: `https://github.com/BartuCan02/thesis-synthetic-weld-rt` (remote `origin`, private; the box has no
  GitHub credentials, so it pulls from `~/thesis.git`). Sync: on the laptop `git push origin main && git push box main`,
  on the box `git -C ~/thesis pull --ff-only`.
  Code for ClearML agents is sent as standalone scripts (`Task.init` + `execute_remotely`), so the agents never
  need to clone this repo.
- **Bartu launches long runs himself.** Hand over the exact command; do not auto-launch training.
- Python via `uv`, exact version pins (`==`), Python 3.12. Ruff for lint/format.
- Manuscript is LaTeX in `thesis/`.

## Working style

- Explain new topics in plain words, define every term, one idea per sentence.
- After any code change, list the changed files and line ranges.
- Close background watchers when done; never write a pgrep loop that can match itself.
- Commits attribute to GitHub `BartuCan02` via the repo-local noreply email (already set).
