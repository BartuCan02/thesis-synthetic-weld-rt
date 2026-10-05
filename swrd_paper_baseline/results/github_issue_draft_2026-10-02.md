# Draft GitHub issue for bit628/RapidX-Annotator (the authors' repo that links the SWRD download)

Title: SWRD dataset: 190 T-joint crop images missing from SWXD_Data.zip, and baseline preprocessing details

Body:

Hello, and thank you for releasing SWRD (J. Nondestructive Evaluation 44:50, 2025). I am reproducing the
YOLOv8 baseline of Sect. 4 and ran into two things I hope you can help with.

**1. Missing images.** In `SWXD_Data.zip` (124,398,609,179 bytes, Google Drive folder
`1LNUt101wufTBJpRAgrZAU1h-Tfx629wO`), `crop_weld_data/crop_weld_jsons/T/` has 2,508 label files but
`crop_weld_data/crop_weld_images/T/` has 2,318 images. The 190 label files in `crop_weld_jsons/T/2/`
(`A_DJ-RT-20240105-*.json`, `B_DJ-RT-20240105-*.json`) have no matching image anywhere in the archive, so the
release has 4,740 weld images where the paper reports 4,930. Could the `T/2` images be added to the download?
The uncropped originals `Raw_data/images/DJ-RT-20240105-*.tif` are present, so I can also reconstruct the crops
from them if you confirm that the crops are plain cut-outs without resampling.

**2. Preprocessing and training parameters not stated in the paper**, needed for an exact reproduction:
- Sect. 3.3 contrast stretching: min-max or percentile-based, and per tile or per image?
- CLAHE clip limit and tile grid size, applied before or after tiling?
- Sect. 3.2: rule for a polygon cut by a tile border (clipped box always kept, or a visibility threshold?).
  Were windows that do not fit the regular 50 % grid at the image border discarded? With that reading I get
  423,651 tiles and 81,630 tiles with defects, close to your "over 380,000" and 80,648.
- Table 3: was the 9:1 split drawn at random over tiles? Is the validation tile list available?
- Table 4: Ultralytics version, image size, and how batch 480 was run (several GPUs?).
- Is a reference YOLOv8m weight file available?

Happy to share my preparation scripts and counts in return. Thanks!
