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
