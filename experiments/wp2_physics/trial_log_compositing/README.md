# Trial: moving real SWRD defects into defect-free SWRD films (2026-10-08)

Branch `wp2-log-compositing-trial`. A first look, not an experiment: one defect per class, six host films,
judged by eye. Built after reading Mery & Katsaggelos 2017 (CVPRW), with Felix's defect-removal idea
(2026-10-07) as the way to get a defect "on its own".

## What it does

1. **Find the additive space** (`pore_dip_test.py`, runs on the data box). For 13,271 pores on 600 films,
   regress log(pore dip) on log(local background) within each film. Elasticity 0 means raw grey values
   already add. Elasticity 1 means Mery's log space with B = 0.
2. **Pick films** (`select_films.py`, data box). One isolated, clearly visible source defect per class
   that runs along the main weld, plus defect-free landscape 16-bit hosts. Manifest: `results/films_manifest.json`.
3. **Extract, place, insert** (`run_trial.py` + `src/rtsynth/compositing.py`, laptop):
   - remove the defect by Telea inpainting in the additive space, refill the removed area with real grain
     taken beside it;
   - extracted defect = original - removed, lightly smoothed (sigma 1 px), darkening only (sign constraint);
   - place it on the host where the host's brightness profile across the weld matches the source's;
     the score is the worst of three slices along the patch, so spots straddling a brightness step lose;
   - **physical arm:** host + scale x extracted defect, where scale = host grain / source grain keeps the
     defect's contrast-to-noise ratio;
   - **naive arm (A/B control):** alpha blend of the raw source crop, same soft mask, same place.

## Results

**Additive space.** Elasticity 0.23, 95% interval 0.09 to 0.38 (`results/pore_dip_test.png`). The binned means
lie on the flat line. SWRD's raw 16-bit values are already close to additive, consistent with a scanner that
is linear in film density. Mery's log step is not needed for SWRD; the trial uses Z = I. The log space stays
available as `--space log --offset B`.

**Insertions** (`figures/overview.png`, one detail figure per class in `figures/`):

| class | source film | host film | placement match | contrast scale | verdict by eye |
|---|---|---|---|---|---|
| porosity | A_DJ-RT-20230721-149 | A_DJ-RT-20230106-86 | 0.997 | 0.90 | plausible, slightly soft |
| inclusion | A_DJ-RT-20230718-13 | A_DJ-RT-T-20230803-35 | 0.996 | 1.21 | plausible |
| crack | A_DJ-RT-20230711-1 | A_DJ-RT-T-20230803-46 | 0.997 | 1.15 | plausible |
| undercut | A_DJ-RT-20221215-33 | A_DJ-RT-T-20231007-21 | 0.969 | 1.51 | plausible |
| lack of fusion | A_DJ-RT-20230321-44 | A_DJ-RT-20221228-12 | 0.998 | 1.01 | plausible |
| lack of penetration | A_DJ-RT-20230328-40 | A_DJ-RT-20230110-58 | 0.960 | 1.24 | plausible, carries some source texture |

The naive arm fails visibly on every class: it pastes the source film's background level, so the defect
appears as a black or white blob.

## What the trial taught us (in order of discovery)

- **Additivity within a film is not enough.** Without contrast scaling, a pore moved from a smooth film into a
  grainy one disappeared. Grain differs up to about 3x between SWRD films.
- **Placement needs the weld geometry.** The seam polygons cover almost the whole crop, so they cannot place
  a defect relative to the bead. Profile matching does. One random spot sat next to a brightness step
  (score 0.76) and looked wrong.
- **T-joint crossing welds.** An undercut on a T-joint's crossing weld runs vertically and has no counterpart
  on a host. Such sources are now excluded.
- **Removal quality limits Felix's idea.** Classical inpainting leaves a visible ghost on the inclusion and a
  smooth band on the large lack of penetration (see the "defect removed" panels). Removal is good enough to
  extract small defects, not yet good enough to produce clean training hosts. This is the shortcut risk noted
  in `docs/decisions.md` (2026-10-07).
- **Clean hosts are scarce in landscape.** Of the 357 defect-free crops, 320 are portrait T-joint halves.
  33 landscape 16-bit hosts exist, all with pseudo-defects or a second seam polygon used as keep-out zones.

## Limits of this trial

- Six examples, judged by eye. No detector, no realism study.
- Sources were chosen for visibility (CNR at or above the class median). Do not reuse that filter for training data.
- The extracted defect still carries some source grain. The sign clip of that grain adds a faint dark bias
  around the defect. A real noise model is RQ3 work.
- Grain-ratio scaling assumes both films share film type and gradient.
- The naive arm here is the literature's pure alpha blend. A brightness-matched or Poisson paste is a
  stronger control and should be decided before the WP4 A/B.

## Reproduce

```bash
# on the data box (raw release at ~/swrd_paper_baseline/data/raw)
python pore_dip_test.py --raw ~/swrd_paper_baseline/data/raw --common-dir ~/thesis/swrd_paper_baseline/scripts --out pore_dip
python select_films.py --raw ~/swrd_paper_baseline/data/raw --common-dir ~/thesis/swrd_paper_baseline/scripts --out films
# on the laptop: copy films/ to experiments/wp2_physics/trial_log_compositing/data/films (git-ignored), then
uv run python experiments/wp2_physics/trial_log_compositing/run_trial.py
```

16-bit composites and LabelMe JSON of the moved polygon land in `data/outputs/` (git-ignored).
