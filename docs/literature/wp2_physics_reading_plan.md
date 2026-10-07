# WP2 physics track: reading plan (written 2026-10-07, week 2)

Purpose: before writing any compositing code, be able to write down the full image-formation chain for an
SWRD film and say which quantity in it is known, which is calibrated from the image, and which is assumed.
Output of this reading: `experiments/wp2_physics/image_formation_model.md` (one page, equations + unknowns)
plus one literature note per paper in this folder.

## Tier 0: radiographic testing physics (2 days, textbook level)
Goal: understand the chain source -> steel -> film -> scanner -> 16-bit pixel.
- Halmshaw, *Industrial Radiology: Theory and Practice*, 2nd ed., Chapman & Hall 1995. Chapters on film
  characteristic curve (density vs log exposure, gamma, toe/shoulder), scatter and build-up, unsharpness,
  contrast sensitivity and IQIs. (Alternative: ASNT NDT Handbook vol. 4, Radiographic Testing.)
- Mery, *Computer Vision for X-Ray Testing*, Springer (2nd ed. 2021), ch. 2 "Images for X-ray testing"
  and the simulation chapter. Beer-Lambert for polychromatic beams, film/detector response, noise, and his
  own flaw-simulation method. Mery is the single most relevant author for this thesis.
- NIST XCOM: mass attenuation coefficient of iron vs photon energy. Pull the curve, do not quote numbers
  from memory.
- ISO 17636-1 (film RT of welds): what the IQI wires mean, so the uncropped originals can give a scale.
Key fact to settle here: SWRD films were digitised on a Microtek MII5000LC (density range up to 4.5D).
A film scanner's pixel is a function of optical density, and density is ~linear in log exposure in the
film's linear region. So where exactly a thickness change is additive (pixel space, log-pixel space) is a
property of the film-plus-scanner chain, not just Beer-Lambert. This decides the form of the RQ3 constraint.

## Tier 1: the compositing papers already cited (read for equations, 2 days)
Extract from each: the forward model, how background/offset/gain is calibrated, how realism was checked.
- Mery & Katsaggelos 2017, CVPRW: log-additive insertion, offset and gain calibration before taking logs.
  Replicate this calibration; it is load-bearing ("a raw grey value is not a transmission").
- Mery, Hahn, Hitschfeld 2005, Insight: CAD flaws as thickness reductions onto real casting/weld radiographs.
- Rogers 2016 (threat image projection, security screening): multiplicative projection from Beer-Lambert.
- Bhowmik 2019, BMVC-W: the naive alpha-blend arm, lost ~10 mAP. This is the control arm of our A/B.
- Pezeshk 2016 (SPIE), Robins 2017 (PMB), Robins 2019 (Med Phys): lesion insertion in CT/mammography and
  how they *validated* insertion (observer studies, statistical indistinguishability). Template for WP7.

## Tier 2: full forward simulation (RQ1's "physics simulation" paradigm, 1 day, survey only)
Decide whether a full-simulation arm is feasible with no acquisition metadata, or whether simulation is
used only to produce defect perturbation templates delta(mu*t) that are then composited.
- aRTist (BAM, Bellon et al.): analytical RT simulation, industrial standard.
- gVirtualXray (Vidal et al.): open-source GPU Beer-Lambert simulator.
- CIVA RT (CEA): commercial, reference only.
- Search terms: "aRTist weld defect deep learning", "gVirtualXray weld", "simulated radiograph pore
  detection cast" (Fraunhofer work on synthetic pores in cast aluminium CT; verify citations before use).

## Tier 3: noise (half a day)
- Foi et al. 2008, IEEE TIP, "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw
  data": fit a signal-dependent noise curve from a single image. Directly applicable: measure SWRD's
  noise-vs-intensity curve on defect-free films, which gives the grain + photon term of RQ3 without metadata.
- Film granularity (Selwyn) in Halmshaw; how it scales with density.

## Tier 4: defect removal / paired data (Felix's 2026-10-07 suggestion, half a day)
- Xia, Chartsias, Tsaftaris 2020, Medical Image Analysis, pseudo-healthy synthesis: remove pathology to make
  (healthy, pathological) pairs. The medical analogue of removing a crack from an SWRD film.
- Hu 2023 label-free liver tumor (already listed) is the reverse direction.
- Classical inpainting for the removal step on raw16: Telea 2004 (OpenCV `inpaint`, supports 16-bit).
- Learned: LaMa (Suvorov 2022, WACV). Caveat: 8-bit, would need adaptation.

## First measurement to do alongside the reading (half a day, grounds everything above)
On ~20 defect-free SWRD films (raw16): pixel histogram and clipping at 0/65535; a profile across the seam;
noise variance vs local mean (Foi-style). Writes to `experiments/wp2_physics/`. No ClearML needed.

## Timebox
WP1 (M1 23 Oct) stays the priority. Reading: ~2 days per week in weeks 2-3, done by 2026-10-16.
