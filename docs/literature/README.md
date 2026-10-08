# Literature notes

One markdown file per paper: `<firstauthor><year>_<shortname>.md`. Each note: full citation with DOI, what
they did, the one number that matters, how it relates to this thesis, verified-or-not.

Overview of the papers read so far (link, summary, key points, relevance): `literature_review.xlsx` in the
repo root. Add a row when a paper is finished.

Verified in the 2026-09-20 sweep (Crossref/CVF), all cited in proposal v0.2:
- Rogers 2016 (TIP, IEEE ICCST) — multiplicative projection from Beer–Lambert
- Mery & Katsaggelos 2017 (CVPRW) — explicit log-additive form, offset/gain calibration
- Mery, Hahn, Hitschfeld 2005 (Insight) — CAD flaws onto real casting and weld radiographs
- Pezeshk 2016 (SPIE), Robins 2017 (PMB), Robins 2019 (Med Phys) — lesion insertion + validation
- Bhowmik 2019 (BMVC-W) — the naive alpha-blend arm only, lost 10 mAP
- Gao 2023 SyntheX (Nat Mach Intell) — simulated X-rays beat real on 2 of 3 tasks
- Cheng 2026 (J. Manuf. Processes) — closest competitor, get the PDF
- Cho 2026 (J. Intell. Manuf.) — inpainting diffusion + mask generation
- Zhang 2024 LeFusion, Hu 2023 label-free liver tumor — medical lesion synthesis

## Open novelty checks (added 2026-10-08, full text not yet read)
Both found while reading Mery & Katsaggelos 2017. Read before proposal v0.3; they may force a narrower novelty claim.
- Saavedra, Banerjee, Mery 2021, Neural Computing and Applications 33:7803-7819, doi:10.1007/s00521-020-05521-2.
  PGGAN-generated threat objects + "superimposition" used to train YOLO/RetinaNet, tested on real GDXray.
  Check: (a) is the superimposition the 2017 log model? If yes, "learned generator + attenuation constraint" already
  exists for baggage, and gap (1) must be narrowed (e.g. to NDT defects, or to a generator that outputs log residuals).
  (b) do they compare against naive blending?
- "Improved threat item detection in baggage X-ray imagery through image projection", J. Visual Communication and
  Image Representation 2025 (ScienceDirect pii S1047320325001312). Pixel-wise material-aware blending; abstract
  reports mAP gains over real-only on GDXray, PIDray and an in-house set. Check: is there a naive-blend control arm?
  If yes, gap (2), the physical-vs-naive A/B, has a baggage precedent.
