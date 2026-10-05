# SWRD raw-release inventory

TIFFs: **4740** (paper Sect. 3.1: 4930); JSONs: **4930**
- subsets: {'L': 2422, 'T': 2318}
- exposures (A_/B_ halves merged): 3581
- images without JSON: 0; JSONs without image: 190
- byte-identical duplicate groups: 36 (36 redundant files)
- JSON imageHeight/imageWidth != TIFF size: 0
- dtypes: {'uint16': 4460, 'uint8': 280}; portrait (H > W): 1157
- width  median/min/max: 6940 / 165 / 8463; height median/min/max: 781 / 163 / 2217
- films with >= 1 mapped defect: 4399; defect-free: 341

## Label strings in the JSONs

| label | count | mapped to |
|---|---:|---|
| 气孔 | 25830 | porosity |
| 伪缺陷 | 12365 | — (ignored) |
| 焊缝 | 6967 | — (ignored) |
| 夹渣 | 2182 | inclusion |
| 裂纹 | 1984 | crack |
| 未熔合 | 850 | lack_of_fusion |
| 未焊透 | 724 | lack_of_penetration |
| 咬边 | 227 | undercut |
| 夹钨 | 33 | inclusion |
| 焊瘤 | 12 | — (ignored) |
| 内凹 | 10 | — (ignored) |
|  | 1 | — (ignored) |

shape types: {'polygon': 51185}

## Fig. 4 check (instances and mean polygon area on the weld images)

| class | ours | paper | diff | mean area ours | paper |
|---|---:|---:|---:|---:|---:|
| porosity | 25830 | 25401 | +429 | 1663.3 | 1619.3 |
| inclusion | 2215 | 2017 | +198 | 3336.3 | 3363.1 |
| crack | 1984 | 1754 | +230 | 11909.4 | 13195.2 |
| undercut | 227 | 213 | +14 | 8657.6 | 8883.3 |
| lack_of_fusion | 850 | 802 | +48 | 16846.9 | 17657.2 |
| lack_of_penetration | 724 | 745 | -21 | 53876.4 | 54412.7 |

Note: Fig. 4 was counted on the 3,675 original films; a T-joint defect cut by the A_/B_ split or a polygon straddling a crop border would count twice here.

## Tile count under Sect. 3.2 (side = short/2, 50 % overlap)

- edge=flush: **553,775**
- edge=drop:  **427,103**
- paper: over 380,000
- (alternative reading, window = whole short side, flush: 74,274)

## Duplicate groups (first 20)

- A_DJ-RT-20230321-103, A_DJ-RT-20230909-75
- A_DJ-RT-20230321-104, A_DJ-RT-20230909-76
- A_DJ-RT-20230321-105, A_DJ-RT-20230909-77
- A_DJ-RT-20230321-106, A_DJ-RT-20230909-78
- A_DJ-RT-20230321-130, A_DJ-RT-20230909-88
- A_DJ-RT-20230321-146, A_DJ-RT-20230909-90
- A_DJ-RT-20230321-18, A_DJ-RT-20230909-17
- A_DJ-RT-20230321-2, A_DJ-RT-20230909-2
- A_DJ-RT-20230321-20, A_DJ-RT-20230909-19
- A_DJ-RT-20230321-23, A_DJ-RT-20230909-22
- A_DJ-RT-20230321-24, A_DJ-RT-20230909-23
- A_DJ-RT-20230321-25, A_DJ-RT-20230909-24
- A_DJ-RT-20230321-26, A_DJ-RT-20230909-25
- A_DJ-RT-20230321-27, A_DJ-RT-20230909-26
- A_DJ-RT-20230321-28, A_DJ-RT-20230909-27
- A_DJ-RT-20230321-3, A_DJ-RT-20230909-3
- A_DJ-RT-20230321-31, A_DJ-RT-20230909-30
- A_DJ-RT-20230321-35, A_DJ-RT-20230909-34
- A_DJ-RT-20230321-37, A_DJ-RT-20230909-36
- A_DJ-RT-20230321-42, A_DJ-RT-20230909-41

## JSONs without image (first 20 of 190)

- crop_weld_jsons/T/2/A_DJ-RT-20240105-100.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-101.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-105.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-106.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-108.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-109.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-11.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-110.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-112.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-113.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-114.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-115.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-116.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-117.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-12.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-120.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-121.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-122.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-123.json
- crop_weld_jsons/T/2/A_DJ-RT-20240105-124.json

## Unmapped label strings (ignored at training time; decide in step 1 gate)

- 伪缺陷: 12365
- 焊缝: 6967
- 焊瘤: 12
- 内凹: 10
- : 1
