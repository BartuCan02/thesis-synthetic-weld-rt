# Draft e-mail to the SWRD authors — parameters the paper does not state

To: Xinghua Yu <xyu@bit.edu.cn> (corresponding author); cc Xuefeng Zhao if an address can be found.
Subject: SWRD dataset — reproducing the YOLOv8 baseline: a few preprocessing details

Dear Prof. Yu,

I am a master's student at TU Munich working with Deeplify GmbH on synthetic data for weld radiograph
inspection. We build on SWRD (Zhao et al., J. Nondestructive Evaluation 44:50, 2025) and are reproducing the
YOLOv8 baseline of Sect. 4 before using it as a reference. From the cropped images and polygon files we
reproduce your tile counts closely (Table 2 within about 3 %, 81,630 vs 80,648 tiles with defects), but a few
details are not in the paper. Could you share them, or point us to the code if it is available?

1. Sect. 3.3 contrast stretching: linear min–max per tile, or percentile-based (which percentiles)? Applied per
   tile or per weld image?
2. CLAHE: clip limit and tile grid size (e.g. OpenCV `createCLAHE(clipLimit=?, tileGridSize=?)`), applied to the
   8-bit tile or to the full image before cropping?
3. Sect. 3.2: how is a polygon that is cut by a tile border converted into a bounding box — clipped box kept
   always, or only above some visible-area threshold? Were tiles along the image border that do not fit the
   regular 50 % grid discarded (our count of 423,651 tiles suggests yes)?
4. Table 3: was the 9:1 split drawn at random over tiles (as we read it), and is the list of validation tiles
   available?
5. Table 4: which Ultralytics version, image size (default 640?), and how was batch size 480 run — several GPUs
   or a smaller image size? Was the reported mAP computed on the validation set of Table 3?
6. Is there a reference weight file (YOLOv8m) we could evaluate against our data preparation?
7. The Google Drive archive `SWXD_Data.zip`: does `crop_weld_data/crop_weld_images/T/2/` contain 190 images
   (`DJ-RT-20240105-*`)? Our copy lacks exactly these.

Any of these would help us make the comparison exact. We will of course cite SWRD and are happy to share our
preparation scripts and counts in return.

With kind regards,
Bartu Can
TU Munich / Deeplify GmbH, bartu.can@deeplify.de
