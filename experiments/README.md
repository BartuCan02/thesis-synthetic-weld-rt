# Experiments

One folder per work package. Each folder has its own README with: goal, data version, deeplify framework
commit, ClearML task ids, the number that came out, and what was concluded. Update it the same day the
run finishes.

## Reference baseline (not ours, do not rerun)
SegFormer mit_b4, 7 classes, SWRD v7 only, March 2026. Macro IoU 0.4838. Ids in `../CLAUDE.md`.

## Work packages
| WP | folder | milestone | status |
|---|---|---|---|
| WP1 | `wp1_benchmark/` | M1 23 Oct 2026 | not started |
| WP2 | `wp2_physics/` | M2 4 Dec 2026 | not started |
| WP3 | `wp3_inpainting/` | M3 15 Jan 2027 | not started |
| WP4 | `wp4_ab_physical_vs_naive/` | M4 29 Jan 2027 | not started |
| WP5 | `wp5_downstream_utility/` | M5 26 Feb 2027 | not started |
| WP6 | `wp6_hybrid/` (designated cut) | M4 29 Jan 2027 | not started |
| WP7 | `wp7_realism_study/` | M5 26 Feb 2027 | not started |

## Pinned dependency
deeplify commit used for training: _not pinned yet_ (set when WP1 starts).
