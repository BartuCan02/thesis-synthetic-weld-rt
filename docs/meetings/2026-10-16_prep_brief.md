# Meeting prep brief, written 2026-10-15 (week 3 of 26)

For the Cecilia call (16 Oct) and the Felix meeting (about 19 Oct). Facts are from `docs/decisions.md` and
`experiments/wp1_benchmark/`. M1, benchmark frozen, is Fri 23 Oct.

## What has been done

**Scope (fixed with Felix, week of 15 Sep).** SWRD only. Cut-and-paste dropped. Inpainting is an augmentation
technique to enlarge and rebalance the training set. Headline experiment: physically composited vs naively blended
synthetic data, same model, scored on real film. WP6 hybrid generator is the designated cut.

**Data.** SWRD release: 4,930 weld crops (190 of them rebuilt from the uncropped originals, pixel-identical check on
80 films), 16-bit, seam polygon on every image, 6 benchmark classes (9 in the files). No acquisition metadata (no kV,
thickness, distance), so the physics is metadata-free by construction. 158 crops are 8-bit data stored as 16-bit.

**WP1, paper baseline reproduced (2026-10-06).** Paper recipe rebuilt from the raw files: tiles of half the short side,
50 % overlap, per-tile percentile stretch, 8-bit, CLAHE, YOLOv8, tile-level 9:1 split, 100 epochs, effective batch 480.
YOLOv8m mAP50 0.730 / mAP50-95 0.462 vs the paper's 0.663 / 0.448. Ranking and peak epochs match; the gap is
unpublished parameters, documented, not chased.

**WP1, thesis protocol = film-level split (2026-10-08).** Same tiles split by exposure, no film on both sides.
Same recipe drops to 0.452 / 0.240. That drop is the main finding so far: the model generalises poorly to
unseen films; the paper number is inflated by overlapping tiles of the same film.

**WP1, three arms on the film split, seed 0, 100 epochs, 4xT4, about 30 h each.**

| arm | mAP50 | mAP50-95 | porosity | inclusion | crack | undercut | LoF | LoP |
|---|---|---|---|---|---|---|---|---|
| A real only | 0.452 | 0.240 | 0.736 | 0.260 | 0.573 | 0.101 | 0.263 | 0.783 |
| B repeat-factor oversampling t=0.1 | 0.450 | 0.239 | 0.739 | 0.249 | 0.574 | 0.120 | 0.246 | 0.775 |
| C real rare defects moved to other films, physical insertion, RFS-matched budget | 0.437 | 0.236 | 0.733 | 0.266 | 0.535 | 0.070 | 0.243 | 0.775 |
| side run A + 2,491 Deeplify customer films (diagnostic, Felix's question) | 0.466 | 0.249 | 0.749 | 0.261 | 0.558 | 0.139 | 0.299 | 0.788 |

Readings: repeating real rare tiles gives nothing; moving real defects gives nothing or loses; adding new real
films gives +0.013 overall and +0.04 on undercut and lack of fusion, the classes with the most customer boxes.
So the model lacks different-looking defects on different films, not more exposures to the same ones. That is
the target for the generators: appearance diversity, not relocation. The side run is a diagnostic, scope stays SWRD-only.

**Caveats that go with every number.** One seed per arm, seed noise unmeasured (differences under about 0.02 are
not interpretable). No test set, best checkpoint picked on val. Undercut AP rests on 16 val films. Three train
films are byte copies of val films (small leak, to remove at the freeze). Val negatives are a random half of the
clean tiles, so false positives on the discarded tiles are unmeasured. Split is random by film, not stratified.

**WP2 so far.** Compositing trial (2026-10-08): raw16 is already near-additive, so Mery's log step is not needed;
cross-film contrast scaling and placement by the profile across the weld are required; classical inpainting is
too weak to remove defects for clean hosts. Run C's synthetic cracks and inclusions measured about 20 % fainter
than real in the 8-bit tiles. Reading plan written; image-formation note and noise measurement not started.

**Qualitative look at crack predictions (run A).** Clear long cracks are found at confidence 0.6 to 0.77. Misses are
(a) crack pieces clipped at tile borders, an artefact of tiling, (b) faint cracks in heavy grain, (c) branched cracks
labelled per branch where the model returns one box. (a) and (c) are metric artefacts; (b) is the real weakness.

## Questions for Cecilia

1. Benchmark metric: proposal v0.2 RQ4 says per-class IoU and instance detection rate; WP1 is built on detection mAP
   in the paper's configuration (Felix, 5 Oct). Which is the thesis benchmark, or both with one as primary?
2. Is one seed per arm acceptable for M1 if I add a measured noise bar (second seed of the baseline), or does she
   expect three seeds per arm for the headline A/B later?
3. Is val-only evaluation without a test set acceptable for a thesis, if stated, or should a test split be carved
   now, at the cost of invalidating the three runs done?
4. RQ2 wording: still says mask-conditioned generation; needs rewording to inpainting-as-augmentation in v0.3.
   When does she want proposal v0.3?
5. M1 deliverable: she listed "benchmark frozen, baseline reproduced, chapter 2 draft". Is a literature chapter
   draft expected on 23 Oct or at M2?

## Questions for Felix

1. Run C: moving real defects between films is close to the cut-and-paste dropped on 15 Sep. Is it accepted as the
   physics step and defect source for the A/B, or should WP2 synthesise the perturbation shape too?
2. Preprocessing in the thesis benchmark: keep the paper's per-tile stretch (his 5 Oct decision), knowing it rescales
   each tile and distorts the contrast WP2 inserts, or fix one film-level normalisation at M1 (one re-render,
   one short run)?
3. Detection vs segmentation (same as Cecilia Q1); the elongated classes favour segmentation.
4. Shorter recipe for all later runs: native tile size (416 instead of 640, tiles have median side 343 px) and 50
   epochs, about 7 h instead of 30. One check run against A before adopting. Agree?
5. Side run result: customer films helped a little, mostly undercut and LoF. Anything he wants done with it, or
   does it close the question?
6. His defect-removal pairs idea (7 Oct): classical inpainting failed on raw16 in the trial. Worth a learned removal
   step, or park it?
7. Compute: a g5 instance (A10G) would be about 3x the T4s. Possible?

## Decisions I want to leave the meetings with

- Metric for M1 (mAP, IoU, or both).
- Preprocessing frozen as paper per-tile, or film-level.
- Recipe frozen (resolution, epochs, seeds policy).
- Run C's method accepted or reframed; what WP2 builds next (porosity first, as Felix said on 5 Oct).
