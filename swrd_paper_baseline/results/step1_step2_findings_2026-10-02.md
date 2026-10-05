# Steps 1–2 findings — inventory and tile index (2026-10-02)

Box: `~/swrd_paper_baseline/data/raw` = `aws s3 sync s3://swdr/cropped/` (40 GB, 9,670 files). Full inventory
report: `inventory_report_2026-10-02.md`. Tiling summaries: `tiles_summary_edge_*.json`.

## Inventory (step 1 gate)

| fact | value | consequence |
|---|---|---|
| TIFFs / JSONs | 4,740 / 4,930 | **190 images missing from Deeplify's mirror**: all of `crop_weld_jsons/T/2/` (95 A_ + 95 B_, films `DJ-RT-20240105-*`) have labels but no image anywhere in `s3://swdr/`. Paper: 4,930. |
| exposures (A_/B_ merged) | 3,581 (+95 missing = 3,676) | matches the paper's 3,675 original films |
| byte-identical duplicates | 36 pairs, all `A_DJ-RT-20230321-N` = `A_DJ-RT-20230909-M`, identical labels | drop the second copy (`--skip-duplicates`) → 4,704 unique images |
| bit depth | 4,460 uint16, **280 uint8** (all in `L/1`, mostly `A_DJ-RT-20230910-*`, `A_DJ-RT-20230719-*`) | the release is not uniformly 16-bit; the per-tile stretch handles both, record it |
| orientation | 1,157 portrait (1,156 of them T-joint) | window rule uses the shorter side, so unaffected |
| size | width median 6,940 (165–8,463), height median 781 (163–2,217) | window side median ≈ 390 px |
| labels | 12 strings; six mapped; ignored: 伪缺陷 pseudo-defect 12,365, 焊缝 seam 6,967, 焊瘤 excess metal 12, 内凹 root concavity 10, one empty | 焊瘤/内凹 are real imperfections → never negatives; pseudo-defect tiles stay eligible as negatives (the paper had no such class either) |
| `A_` prefix | also used on standard (L) films, not only T-joint halves | `exposure_id` strips it everywhere; harmless |

Fig. 4 check (instances on the weld images, ours with 190 images missing and 36 duplicates present):

| class | ours | paper | diff |
|---|---:|---:|---:|
| porosity | 25,830 | 25,401 | +1.7 % |
| inclusion (夹渣 2,182 + 夹钨 33) | 2,215 | 2,017 | +9.8 % |
| crack | 1,984 | 1,754 | +13 % |
| undercut | 227 | 213 | +6.6 % |
| lack of fusion | 850 | 802 | +6.0 % |
| lack of penetration | 724 | 745 | −2.8 % |

Explanation: Fig. 4 counts the 3,675 *original* films; the cropped set counts a polygon once per crop it
appears in (A_/B_ halves, crop borders) and the 36 duplicates twice. Mean areas agree to within 2–10 %, cracks
lowest (cut polygons are smaller and more numerous — consistent). Whether 夹钨 belongs in "inclusion" cannot be
decided from the counts; kept merged (ISO 6520-1 treats tungsten inclusions as inclusions).

## Tile index (step 2 gate) — which border reading is the paper's

Both runs on the 4,704 unique images, D7 defaults (min side 4 px, ≥10 % of polygon visible or ≥0.5 % of tile).

| | edge = flush | **edge = drop** | paper |
|---|---:|---:|---:|
| tiles | 549,274 | **423,651** | "over 380,000" |
| tiles with a box | 102,352 | **81,630** | 80,648 |
| tiles touching any defect | 108,021 | 85,992 | — |
| clean tiles (negative pool) | 441,253 | 337,659 | — |
| porosity instances | 130,078 | 105,895 | 102,927 |
| inclusion | 12,101 | 9,833 | 9,141 |
| crack | 17,108 | 13,259 | 13,515 |
| undercut | 1,399 | 1,061 | 1,015 |
| lack of fusion | 8,666 | 6,839 | 6,862 |
| lack of penetration | 22,390 | 18,125 | 20,422 |

**edge = drop is the paper's rule**: positives within +1.2 % and five of six classes within ±3 % of Table 2,
with 4 % fewer source images than they had. Lack of penetration is 11 % short. A loose variant of D7 (keep every
visible piece ≥ 2 px, `tiles_summary_edge_drop_loose_2026-10-02.json`) gives 85,313 positives (+5.8 %), overshoots
five classes by 4–16 % and still leaves lack of penetration 8 % short — so the rule is not the cause and D7's
defaults are the closer match. The remaining LoP gap stays unexplained (their clip rule is unknown).

## Open decisions for Bartu
1. Proceed with 4,704 images (96 % of the release) and record the T/2 gap, or obtain `crop_weld_images/T/2/`
   (190 TIFFs, ≈1.7 GB) from the official download before building the dataset.
2. Confirm edge = drop, duplicates removed, 焊瘤/内凹 excluded from negatives, pseudo-defect tiles eligible.
3. D7 defaults confirmed (the loose variant matches worse).

## Update, same day — full 4,930-image set after reconstructing the 190 T/2 crops

Reconstruction validated on 80 released T-joint crops (40 A_, 40 B_): 80/80 pixel-identical, no rotation
(`reconstruct_validate_B_2026-10-02.json`). All 190 missing crops rebuilt from the originals (`reconstruct_rebuild_T2_2026-10-02.json`).

Inventory on 4,930 images (`inventory_report_full4930_2026-10-02.md`): 0 orphans either way; 3,676 exposures
(paper 3,675); 36 duplicate pairs unchanged; 4,650 uint16 + 280 uint8; 1,252 portrait; 4,573 films with a mapped
defect, 357 defect-free. Fig. 4 check: porosity 26,674 / inclusion 2,235 / crack 2,071 / undercut 228 / LoF 857 /
LoP 796 vs paper 25,401 / 2,017 / 1,754 / 213 / 802 / 745 — the same +2…+18 % pattern as before (crop-border and
A_/B_ double counting; the paper counted original films). These are exactly the "SWRD facts" in CLAUDE.md.

Tile index, edge = drop, 4,894 unique images (`tiles_summary_full4930_edge_drop_2026-10-02.json`):

| | ours, full set | ours, 4,704 imgs (earlier) | paper |
|---|---:|---:|---:|
| tiles | 441,304 | 423,651 | "over 380,000" |
| tiles with a box | 84,874 | 81,630 | 80,648 |
| porosity | 109,491 | 105,895 | 102,927 |
| inclusion | 9,941 | 9,833 | 9,141 |
| crack | 14,082 | 13,259 | 13,515 |
| undercut | 1,067 | 1,061 | 1,015 |
| lack of fusion | 6,897 | 6,839 | 6,862 |
| lack of penetration | 18,843 | 18,125 | 20,422 |

Reading: with the complete image set we sit **+5 % above the paper's positive-tile count** and +16 % above
"380,000" tiles. The earlier +1 % was partly luck (4 % of the images missing cancelled a rule difference). The
remaining gap points at the authors' unpublished window/box details (e.g. whether a window that ends exactly on
the border counts, or a visibility threshold) — asked in the issue/e-mail drafts. Lack of penetration stays
8 % short. edge = drop and the D7 defaults remain the best-supported reading; both are flags, so the dataset is
rebuilt in minutes once the authors answer.

## Sweep of the unstated window and box arithmetic (same day, ~25 min)

Scripts `sweep_grid_variants.py`, `sweep_box_rules.py`; data `grid_sweep_2026-10-02.json`, `box_rule_sweep_2026-10-02.json`.

**Grid (72 variants: side rounding x stride rounding x border rule x duplicates).** Total tiles range from
300,936 to 785,087. Six variants fall in 375k–415k; scored with polygons (D7 rule), one stands out:

| grid | tiles | defect tiles | por | inc | crk | und | LoF | LoP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **round / round / window ending on the border counts** | **412,726** | **79,957 (−0.9 %)** | +0.1 % | +2.9 % | −1.2 % | +0.9 % | −7.5 % | −12.3 % |
| floor / round / same (our previous default) | 441,304 | 84,874 (+5.2 %) | +6.4 % | +8.8 % | +4.2 % | +5.1 % | +0.5 % | −7.7 % |
| paper | > 380,000 | 80,648 | | | | | | |

Adopted as default D10 (`_common.tile_side` now rounds to nearest).

**Box rule (18 rules on that grid).** No rule matches all six classes:

| rule | defect tiles | por | inc | crk | und | LoF | LoP |
|---|---:|---:|---:|---:|---:|---:|---:|
| D7 (polygon piece ≥ 4 px, ≥ 10 % visible or ≥ 0.5 % tile) | −0.9 % | +0.1 | +2.9 | −1.2 | +0.9 | −7.5 | −12.3 |
| polygon, any visible piece | +4.1 % | +5.6 | +11.3 | +8.3 | +10.2 | −0.5 | −8.7 |
| bbox clipped, ≥ 6 px | +4.6 % | +0.8 | +7.6 | +9.7 | +8.8 | +0.7 | −0.3 |
| bbox clipped, ≥ 12 px | −0.1 % | −6.1 | +0.5 | +4.8 | +4.2 | −3.9 | −4.4 |

Polygon-based rules undercount the two long classes; bounding-box rules fix those but overcount cracks,
undercut and inclusions by 5–10 %. The authors' rule is therefore not a single threshold of either family, or
their Table 2 was counted at a different stage. **Decision: keep D7.** It matches four classes within 3 % and
the positive total within 1 %, and it is the only family that never writes a box into a window the defect's
pixels do not enter (bounding-box clipping does that for sloped lines). The exact rule stays a question for
the authors (asked in the issue/e-mail drafts).
