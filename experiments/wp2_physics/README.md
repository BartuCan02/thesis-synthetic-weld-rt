# wp2_physics

Goal:

Data version:

deeplify commit:

## Runs
| date | ClearML task | config | result | conclusion |
|---|---|---|---|---|
| 2026-10-08 | none (laptop + data box) | `trial_log_compositing/`, pore-dip test on 600 films; 6 defects into 6 defect-free films | pore-dip elasticity 0.23 [0.09, 0.38]; physical insertions plausible by eye on all 6 classes; naive arm visibly wrong | SWRD raw16 is already near-additive, so Mery's log step is not needed; cross-film contrast scaling and profile-based placement are required; classical inpainting removal is too weak for clean hosts. To review with Felix. |
| 2026-10-17/18 | dataset `4e72b695729840018da24ca311013edd` (physical 3.0.0), run D `1fd29082b4cc4022971a2cdc24742604`; naive dataset `4dadc0ed…`, run E `16cd8721…` (running) | version 3 (`../../swrd_synthetic_physical/`): geometry placement, varied defects, depth compensation, null implants, 3x budget (31,153 positive + 2,272 null tiles), 50 epochs | run D best.pt mAP50 0.424 / 0.229 vs A 0.452 / 0.240; AP50 undercut 0.041 (A 0.101), crack 0.544 (0.573), LoF 0.224 (0.263), inclusion 0.240 (0.260) | no gain, a loss on every boosted class; run A's detector finds the synthetic defects as often as real ones (recall 0.41/0.58/0.14/0.37 vs 0.28/0.55/0.08/0.35), null tiles not learned as defects (17/300 vs 10/300 real clean); so the examples are plausible but add no usable appearance; 50-epoch reference A' not run; physical-vs-naive A/B pending on run E |
| 2026-10-12 | dataset `646e699eaa3246468b7f345c271c94c8` (2.0.1; 2.0.0 `dc3907f3…` broken, run C `67ab6191…` failed on it before training); run C `7e2c9089…` completed 2026-10-13 | `../../swrd_synthetic_physical/`: real inclusion, crack, undercut, lack-of-fusion defects inserted physically into other training films (version 2: never on a spot the baseline trains on), tiles matched to RFS t=0.1 | mAP50 0.437 vs A 0.452 / B 0.450; AP50 undercut 0.070 (A 0.101), crack 0.535 (A 0.573), lack of fusion 0.243 (A 0.263), inclusion 0.266 (A 0.260) | no gain from moved real defects at this budget; one seed per arm, so differences under ~0.02 are not interpretable; synthetic cracks/inclusions ~20 % fainter than real in the 8-bit tiles (1 px smoothing) |
