# Zhao, Wu, Zhang, Wen, Wang, Li, Yu — SWRD: A Dataset of Radiographic Image of Seam Weld for Defect Detection

J. Nondestructive Evaluation 44:50 (2025). doi:10.1007/s10921-025-01186-w. Received 1 Nov 2024, published 7 May 2025.
Beijing Institute of Technology. PDF: `~/Downloads/s10921-025-01186-w.pdf` (not committed). Read in full 2026-10-02.
Download: http://www.tz-ndt.com/#/download (one archive `SWXD_Data.zip`, 115.86 GB; see decisions 2026-10-01 for contents).

## What they did
- **Acquisition (Sect. 2, Table 1):** film radiographs of longitudinal seam welds on large pipes, single-wall vertical
  exposure, flat-panel detector *inside* the pipe (Fig. 1). Films digitised on a Microtek MII5000LC: industrial CCD,
  **1200 dpi**, 16 bit, 355.6 x 5000 mm scan range, dynamic density 3.5D/4.0D/4.5D. **This is the only physical-scale
  statement anywhere: 1200 dpi = 21.2 um/px if the films were scanned at full resolution. Verify against IQI wires
  before trusting it; the TIFF tag says 96 dpi.**
- **Counts:** 3,675 films = 2,420 standard + 1,255 T-joint. Six defect classes from GB/T 6417.1-2005 (= EN ISO
  6520-1:1998): porosity, inclusion, crack, undercut, lack of fusion, lack of penetration. MIG welding, so inclusions
  are mostly non-metallic. Lack-of-fusion subtypes are *not* distinguished.
- **Fig. 4, instances on the original films (number / mean polygon area in px^2):** porosity 25,401 / 1,619.33;
  inclusion 2,017 / 3,363.07; crack 1,754 / 13,195.21; undercut 213 / 8,883.29; lack of fusion 802 / 17,657.22;
  lack of penetration 745 / 54,412.66.
- **Sect. 3.1 weld cropping:** non-weld regions removed; T-joint films split into a primary-weld and a secondary-weld
  image -> **4,930 weld images**, polygon coordinates recalculated. (These are the `crop_weld_images` + `crop_weld_jsons`
  in the release. The crop boxes themselves are not published.)
- **Sect. 3.2 sliding window:** square window with side = **half the image's shorter side**, slid left-to-right and
  top-to-bottom with **50 % overlap** -> "over 380,000" tiles, **80,648** containing a defect annotation; an **equal
  number** of randomly chosen defect-free tiles added as negatives.
- **Sect. 3.3 preprocessing:** contrast stretching -> 16-bit to 8-bit -> CLAHE -> saved as **24-bit three-channel**
  images. No parameters given (stretch percentiles, CLAHE clip limit and tile grid).
- **Table 2, instances on tiles:** porosity 102,927; inclusion 9,141; crack 13,515; undercut 1,015; lack of fusion
  6,862; lack of penetration 20,422 (overlapping tiles count an instance several times).
- **Sect. 4.1, Table 3 split:** tiles divided **9:1 at random** into train/val (no test set; the split is over tiles,
  not films). Train 72,585 defect + 72,646 background; val 8,099 defect + 8,038 background (= 80,684 + 80,684 =
  161,368 tiles; the text's 80,648 differs by 36).
- **Sect. 4.2, Table 4 training:** Ultralytics **YOLOv8 detect** n/s/m/l/x, COCO-pretrained, **100 epochs**, batch
  **n 480, s 480, m 480, l 320, x 240**, "all unspecified parameters default" (so imgsz 640). One NVIDIA Tesla P40
  (24 GB) + Xeon E5-2680 v4 is named; batch 480 at 640 px does not fit one P40, so either several GPUs or a smaller
  image size went unreported.
- **Table 5 results (val set):**

| model | best mAP50 (epoch) | best mAP50-95 (epoch) |
|---|---|---|
| YOLOv8n | 0.48196 (71) | 0.28667 (71) |
| YOLOv8s | 0.59763 (76) | 0.37998 (75) |
| **YOLOv8m** | **0.66265 (72)** | **0.44827 (74)** |
| YOLOv8l | 0.59839 (62) | 0.38664 (62) |
| YOLOv8x | 0.57274 (64) | 0.39021 (100) |

  No per-class AP is reported. The curves (Figs. 13-14) peak around epoch 60-80 and then fall for l and x.
- **Sect. 5 inference:** same sliding window on a new film, detect per tile, stitch boxes back in window order.

## The one number that matters
YOLOv8m, mAP50 0.66 / mAP50-95 0.49 on a random 10 % of tiles. Headline in the abstract; Table 5 says 0.66265 / 0.44827.
(The abstract's 0.49 does not match Table 5's 0.448; cite the table.)

## How it relates to this thesis
- It is the published reference point for SWRD. WP1 reproduces it first (`swrd_paper_baseline/`), then builds the
  thesis benchmark (segmentation, film-level split) on the same raw files.
- Its weaknesses are our M1 protocol items: tile-level random split leaks overlapping tiles and same-film tiles into
  val; no test set; equal-negatives rule is arbitrary; preprocessing parameters unpublished; no per-class numbers.
- The files contain more than the paper says: 9 defect labels + seam + pseudo-defect, 26,674 porosity and 228 undercut
  instances vs 25,401 / 213 in Fig. 4 (CLAUDE.md). Reproducing Fig. 4 from the JSONs is the test of the class mapping.
- 1200 dpi scanner statement -> a candidate physical scale for WP2 (needs IQI check).

## Verified
Numbers above transcribed from the published PDF, pages 3-4 and 10-13, on 2026-10-02.
