# wp2_physics

Goal:

Data version:

deeplify commit:

## Runs
| date | ClearML task | config | result | conclusion |
|---|---|---|---|---|
| 2026-10-08 | none (laptop + data box) | `trial_log_compositing/`, pore-dip test on 600 films; 6 defects into 6 defect-free films | pore-dip elasticity 0.23 [0.09, 0.38]; physical insertions plausible by eye on all 6 classes; naive arm visibly wrong | SWRD raw16 is already near-additive, so Mery's log step is not needed; cross-film contrast scaling and profile-based placement are required; classical inpainting removal is too weak for clean hosts. To review with Felix. |
| 2026-10-09 | dataset `dc3907f320474cd1ab034ea7689b89b1`; run C not launched | `../../swrd_synthetic_physical/`: real inclusion, crack, undercut, lack-of-fusion defects inserted physically into other training films (version 2: never on a spot the baseline trains on), tiles matched to RFS t=0.1 | 2,525 defects on 1,193 films, 11,256 tiles (100 % of target); 0 leakage findings | run C to be compared with run A (film baseline) and run B (RFS) per class |
