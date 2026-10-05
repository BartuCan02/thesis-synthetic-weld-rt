# SWRD raw-release inventory

TIFFs: **4930** (paper Sect. 3.1: 4930); JSONs: **4930**
- subsets: {'L': 2422, 'T': 2508}
- exposures (A_/B_ halves merged): 3676
- images without JSON: 0; JSONs without image: 0
- byte-identical duplicate groups: 36 (36 redundant files)
- JSON imageHeight/imageWidth != TIFF size: 0
- dtypes: {'uint16': 4650, 'uint8': 280}; portrait (H > W): 1252
- width  median/min/max: 6933 / 165 / 8463; height median/min/max: 781 / 163 / 2217
- films with >= 1 mapped defect: 4573; defect-free: 357

## Label strings in the JSONs

| label | count | mapped to |
|---|---:|---|
| 气孔 | 26674 | porosity |
| 伪缺陷 | 13311 | — (ignored) |
| 焊缝 | 7340 | — (ignored) |
| 夹渣 | 2202 | inclusion |
| 裂纹 | 2071 | crack |
| 未熔合 | 857 | lack_of_fusion |
| 未焊透 | 796 | lack_of_penetration |
| 咬边 | 228 | undercut |
| 夹钨 | 33 | inclusion |
| 焊瘤 | 12 | — (ignored) |
| 内凹 | 10 | — (ignored) |
|  | 1 | — (ignored) |

shape types: {'polygon': 53535}

## Fig. 4 check (instances and mean polygon area on the weld images)

| class | ours | paper | diff | mean area ours | paper |
|---|---:|---:|---:|---:|---:|
| porosity | 26674 | 25401 | +1273 | 1642.4 | 1619.3 |
| inclusion | 2235 | 2017 | +218 | 3346.1 | 3363.1 |
| crack | 2071 | 1754 | +317 | 11701.1 | 13195.2 |
| undercut | 228 | 213 | +15 | 8633.8 | 8883.3 |
| lack_of_fusion | 857 | 802 | +55 | 16787.9 | 17657.2 |
| lack_of_penetration | 796 | 745 | +51 | 49763.2 | 54412.7 |

Note: Fig. 4 was counted on the 3,675 original films; a T-joint defect cut by the A_/B_ split or a polygon straddling a crop border would count twice here.

## Tile count under Sect. 3.2 (side = short/2, 50 % overlap)

- edge=flush: **576,506**
- edge=drop:  **444,756**
- paper: over 380,000
- (alternative reading, window = whole short side, flush: 77,326)

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

## Unmapped label strings (ignored at training time; decide in step 1 gate)

- 伪缺陷: 13311
- 焊缝: 7340
- 焊瘤: 12
- 内凹: 10
- : 1
