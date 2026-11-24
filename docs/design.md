# Design decisions

Locked before the main experiments. Rationale in brief; the statistical protocol is in
`preregistration.md`, the loss in `loss_math.md`. **Everything is simulation-only.**

## Task and signals
* **Task:** scripted pick-lift-hold of one cube (approach, descend, grasp, lift, hold). A scripted
  policy keeps control out of scope (no RL, no planning).
* **Target:** cube (x, y, z, yaw). The cube is square, so yaw is defined modulo 90 degrees and the
  angle is represented as `phi = 4*yaw` (tangent-space error, wrapped). Reported yaw error is the
  absolute error modulo 90 degrees.
* **Sensors:** one RGB camera (96x96, 20 Hz), 7 joint positions + velocities (+ fingers), wrist F/T.
* **Per-frame models:** each frame is estimated independently; no temporal filtering (keeps the
  comparison about fusion, not filtering).

## Data
* 600 train / 150 val / 150 test episodes (about 166 frames each) plus 100 `test_ood` episodes whose
  cube starts in a strip never used for the others.
* Splits are by episode. A guard rejects val/test starts closer than 3 mm (xy) and 3 degrees (yaw
  mod 90) to any training start.
* Episode k is fully determined by (configs, seed, k).

## Stages and what each modality can know
Frames are grouped in five stages plus `all`. Proprioception is **gated to the holding phase**:
before the hand is physically coupled to the cube, joint angles only reflect the scripted policy,
which can leak the cube pose (measured by `check_leakage.py`). The pre-registered primary group is
stage `coupled`, the only one in which vision, proprioception and contact are all informative.
The contact expert is conditioned on proprioception (wrench interpretation depends on the pose).

## Degradation grammar
Three families (vision, proprioception, contact), each with calibrated severity levels
(`calibrate_degradation.py`). Scenarios: clean; single-family failures at several severities;
combinations. Two family pairs are **forbidden in mixed training** and used as unseen test
combinations (set C); `allcombos` is the control that trains with them.

## Architecture
* Vision CNN, proprioception MLP, contact MLP: each outputs mean + Cholesky factor
  (softplus diagonal + epsilon) -> positive-definite covariance.
* Fusion in information form: `poe`, `tpoe`, `ci` (covariance intersection; valid under unknown
  correlation, hence no conditional-independence assumption). **F-CI is the primary method.**
* Non-uncertainty-aware fusion baselines: F0 concat, F1 learned with availability masks, F2 gated.
* Missing modalities are masked, never imputed with fake values.

## Training recipe
* Heteroscedastic Gaussian NLL; short MSE warm-up in which the covariance is trained on a detached
  mean (weight 0.1) so early NLL does not collapse the mean.
* Validation split for model selection (`best.pt`); test only after the freeze.
* Mixed training samples degradations per sample (unseen pairs held out).
* 3 seeds; 14 runs per seed (`scripts/run_protocol.py`), resumable with a DONE marker per run and a
  config fingerprint.

## Evaluation
Per-episode sufficient statistics (n, squared 3D error, xy error, |z|, yaw error, Mahalanobis d2,
NLL, 68%/95% coverage) per frame group are stored, so all statistics (episode-level paired cluster
bootstrap, Holm correction, seed sign-consistency) are computed offline without re-running models.

## Out of scope
ROS 2, reinforcement learning, motion planning, multiple objects, sim-to-real.
