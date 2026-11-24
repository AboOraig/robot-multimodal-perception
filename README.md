# Uncertainty-Aware Multimodal Perception for Robotic Manipulation

**Visual, proprioceptive and contact sensor fusion under sensor degradation.**

> **SIMULATION ONLY.** Every number, figure and video in this repository comes from a MuJoCo
> simulation of a Franka Panda. Nothing here has been tested on a real robot, and no claim is made
> about sim-to-real transfer.

A Franka Panda (MuJoCo) picks, lifts and holds one cube. From one RGB camera (96x96), joint
positions/velocities and the wrist force/torque sensor, models estimate the cube pose
(x, y, z, yaw) **with a full covariance**. The question is whether fusing the three modalities with
explicit uncertainty helps when sensors degrade, and whether training with degradations
generalises to unseen combinations of failures.

## Hypotheses (pre-registered, see `docs/preregistration.md`)

| | Statement | Test-split verdict |
|---|---|---|
| H1 | Fusion beats vision-only, especially under visual occlusion | **Supported** (-2.3 / -6.6 / -44.7 mm on the three occlusion scenarios) |
| H2 | Explicit uncertainty improves fusion reliability when modalities fail | **Supported** (better NLL in 9/9 scenarios vs both non-uncertainty fusions, RMSE never significantly worse) |
| H3 | Degradation-aware training generalises to unseen failure combinations | **Partially supported** (H3a holds; non-inferiority to the all-combinations control, H3b, is **not** met) |

Caveats that belong next to these verdicts (full discussion in the report):
* Pooled over *all* frames under severe occlusion, the fused model is **worse** than vision-only
  (41.2 vs 36.2 mm); the gain is in the pre-registered `coupled` stage, where every modality is informative.
* A clean-trained model is catastrophically overconfident under proprioceptive faults.
* Proprioception carries some information about the cube from the scripted policy (policy leakage);
  `scripts/check_leakage.py` quantifies it and it is reported with every proprioception result.
* Verdicts on the `test_ood` split (secondary) agree in direction but are weaker (H2 at threshold, H3 partial).

## Method in brief

* Three encoders (vision CNN, proprioception MLP, contact MLP conditioned on proprioception).
* Each expert outputs a mean and a Cholesky-parameterised positive-definite covariance
  (softplus diagonal + epsilon); trained with a heteroscedastic Gaussian NLL
  (yaw enters through the symmetric-cube angle `phi = 4*yaw`). See `docs/loss_math.md`.
* Fusion in information form: `poe`, `tpoe`, and **`ci` (covariance intersection, primary)**, which
  does not assume conditional independence between modalities.
* Baselines: classical estimators (B1/B3/B6 families, with oracle-noise variants), unimodal learned
  models, and three non-uncertainty-aware fusions (concat, learned-with-masks, gated).
* Degradations: vision (occlusion, noise, ...), proprioception (dropout, bias), contact (noise, bias),
  at several severities; train/test combinations clean/degraded, plus mixed training evaluated on
  held-out family pairs. Details in `docs/design.md`.

## Installation

Python 3.11, tested on Windows 11 with an NVIDIA RTX 3050 Ti (4 GB); CPU also works for tests.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# install PyTorch for your CUDA version first, e.g.:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"          # numpy, scipy, mujoco, matplotlib, pyyaml, tqdm, h5py, pytest
.\scripts\fetch_panda.ps1        # downloads the Panda model at a pinned Menagerie commit
```

On Linux/macOS use `source .venv/bin/activate`; for headless rendering set `MUJOCO_GL=egl`.
The robot model folder is git-ignored; its provenance is in `assets/robots/PROVENANCE.md`.

## Tests

```powershell
pytest -q
```

## Reproducing the pipeline

Paths are always arguments or config entries; heavy outputs can go on another drive.

```powershell
# 1. data: 600/150/150 episodes + 100 out-of-distribution test episodes (resumable)
python scripts/generate_dataset.py --workers 4
python scripts/check_leakage.py            # policy-leakage diagnostic
python scripts/calibrate_degradation.py    # severity calibration + example images

# 2. train the full protocol: 14 runs x 3 seeds (resumable; ~12 h on one RTX 3050 Ti)
python scripts/run_protocol.py --seeds 0 1 2 --out-root runs/protocol --dry-run   # check the plan
python scripts/run_protocol.py --seeds 0 1 2 --out-root runs/protocol

# 3. evaluate (validation for development; test only after the pre-registration freeze)
python scripts/evaluate_all.py --runs-root runs/protocol --split val      --out results/final
python scripts/evaluate_all.py --runs-root runs/protocol --split test     --out results/final
python scripts/evaluate_all.py --runs-root runs/protocol --split test_ood --out results/final

# 4. hypothesis tests and figures
python scripts/analyze.py      --results results/final --split test --out results/final/analysis_test
python scripts/make_figures.py --results results/final --split test \
       --analysis results/final/analysis_test --out results/final/figures_test
```

`analyze.py` writes `decisions.json` (machine-readable verdicts) and `tables.md`.

Single runs: `python scripts/train.py --kind fusion --mode ci --seeds 0` (see `--help`).
Classical baselines alone: `python scripts/run_baselines.py --split val`.

## Pre-registration and the test-set freeze

`docs/preregistration.md` fixes models, scenarios, statistics (paired episode-level cluster
bootstrap, 5000 resamples, Holm correction, sign-consistency over seeds, a 1 mm negligibility
margin) and decision rules **before** the test split was evaluated. Its SHA-256 was recorded in
`docs/FREEZE.txt` on 2026-11-23. Validation observations made before the freeze are listed in
section 7b; deviations (section 9) are to be listed there if any occur.

## Repository layout

```
configs/      simulation, dataset, degradation, training (YAML)
src/mmperc/
  simulation/ scene, camera, scripted pick-lift-hold episode, kinematics
  sensors/    degradation grammar (vision / proprio / contact)
  data/       generation, episode-level splits, leakage guard, storage
  models/     encoders, Gaussian head (Cholesky), unimodal and fusion models
  training/   engine (train/predict), features, degradation mixing, run naming
  baselines/  classical estimators
  evaluation/ metrics, sufficient statistics, scenarios, bootstrap stats, hypotheses
scripts/      CLI entry points (one per pipeline step)
tests/        unit and end-to-end tests
docs/         loss_math.md, design.md, preregistration.md
assets/       robot model provenance (model itself is fetched)
```

## Outputs and where they live

| What | Default location | Git |
|---|---|---|
| Dataset | `data/v1` (set `out_dir`) | ignored |
| Checkpoints and logs | `--out-root` (e.g. `runs/protocol/<run>/seed<k>/`) | ignored |
| Evaluation sufficient statistics | `--out` (e.g. `results/final/<split>/`) | ignored |
| Verdicts, tables, figures | `<out>/analysis_<split>`, `figures_<split>` | small files may be committed |

## Reproducibility notes

* Data are a deterministic function of (configs, seed, episode index).
* Training is seeded, but GPU kernels are not bit-exact across machines or driver versions, so
  expect small numerical differences; the protocol fingerprint refuses to resume with changed configs.
* Episode-level splits: no frame of a test episode is ever seen in training; a guard rejects
  val/test episodes whose initial cube pose is too close to a training episode.

## Limitations

One object, one camera, a scripted policy, a symmetric cube, 96x96 images, per-frame estimators
(no temporal filtering), three seeds, and **simulation only**. See report Section 8 and
`docs/preregistration.md` Section 8.

## Licence and third-party assets

Franka Panda model: Apache-2.0, from MuJoCo Menagerie (pinned commit in `scripts/fetch_panda.ps1`).
