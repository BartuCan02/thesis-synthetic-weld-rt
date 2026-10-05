# Dataset check — /home/ec2-user/swrd_paper_baseline/data/yolo_v1.0_papergrid

## train

- tiles: **143,923** = 71,994 with boxes (paper 72,585) + 71,929 negatives (paper 72,646)
- unpaired: 0 images without label, 0 labels without image
- problems: none
- tile side px: min 82, median 338, max 810; < 160 px: 2,842 (2.0%), > 640 px: 528 (0.4%)
- pixel mean over tiles: 137.5 ± 20.5; tile std median 55.5; flat tiles (std < 2): 37; saturated > 30 %: 131 (0.09%)
- boxes: 136,122 total; per positive tile median 1, max 46
- box size px: width median 43 (p5 15, p95 340), height median 33 (p5 14, p95 94); boxes < 8 px on a side: 3,054 (2.2%)

| class | boxes | touching a tile edge (cut) |
|---|---:|---:|
| porosity | 92,795 | 32,259 (35%) |
| inclusion | 8,408 | 4,466 (53%) |
| crack | 12,094 | 10,698 (88%) |
| undercut | 911 | 749 (82%) |
| lack_of_fusion | 5,693 | 5,195 (91%) |
| lack_of_penetration | 16,221 | 16,014 (99%) |

## val

- tiles: **15,991** = 7,963 with boxes (paper 8,099) + 8,028 negatives (paper 8,038)
- unpaired: 0 images without label, 0 labels without image
- problems: none
- tile side px: min 82, median 338, max 800; < 160 px: 323 (2.0%), > 640 px: 63 (0.4%)
- pixel mean over tiles: 137.4 ± 20.6; tile std median 55.6; flat tiles (std < 2): 6; saturated > 30 %: 24 (0.15%)
- boxes: 14,929 total; per positive tile median 1, max 38
- box size px: width median 43 (p5 15, p95 338), height median 33 (p5 14, p95 90); boxes < 8 px on a side: 328 (2.2%)

| class | boxes | touching a tile edge (cut) |
|---|---:|---:|
| porosity | 10,197 | 3,557 (35%) |
| inclusion | 1,002 | 513 (51%) |
| crack | 1,262 | 1,113 (88%) |
| undercut | 113 | 89 (79%) |
| lack_of_fusion | 657 | 602 (92%) |
| lack_of_penetration | 1,698 | 1,682 (99%) |
