# Pre-registration of the hypothesis tests (simulation only)

**Status: FROZEN when the test split is first evaluated.** Everything below was fixed using the *validation* split
(model design, hyper-parameters, scenario list) and the project brief (hypotheses H1–H3). The test and test_ood splits
have not been used for any decision. After the freeze, changes are only allowed as dated entries in section 9
(*Deviations*); the original text is never edited.

All numbers in this project come from a MuJoCo simulation of a Franka Panda with one cube. Nothing here is a claim
about real robots.

## 1. What is compared

| name (run folder) | modalities | training data | role |
|---|---|---|---|
| `vision_mixed` (B2m) | vision | mixed degradations | vision-only reference for H1 |
| `vision` (B2) | vision | clean | H3 (secondary) |
| `proprio_gated_mixed` (B4gm), `proprio_gated` (B4g) | proprioception, valid only while holding | mixed / clean | unimodal references |
| `fusion_concat_mixed` (F0) | vision + proprio + contact, concatenated, no masks | mixed | non-UA fusion |
| `fusion_learned_mixed` (F1) | same, explicit availability masks | mixed | non-UA fusion (**H2 comparator 1**) |
| `fusion_gated_mixed` (F2) | same, sigmoid reliability gates | mixed | non-UA fusion (**H2 comparator 2**) |
| `fusion_poe_mixed`, `fusion_tpoe_mixed` | 3 Gaussian experts, product of experts / tempered | mixed | UA fusion variants |
| `fusion_ci_mixed` (**F-CI**) | 3 Gaussian experts, covariance-intersection weights | mixed | **designated UA method** |
| `fusion_ci` | as F-CI | clean | H3 comparator |
| `fusion_ci_mixed_allcombos` | as F-CI, *forbidden pairs allowed in training* | mixed | H3 control |
| `fusion_ci_mixed_nocontact` | as F-CI, contact expert has no data | mixed | ablation |
| classical: B1a/B1b vision, B3/B3c kinematics, B6/B6c WLS fusion (+ `[oracle]` noise levels) | | none (calibrated on train) | classical baselines |

F-CI is designated the primary UA method *before* test evaluation because it was the best-calibrated of the three
fusion rules on validation. The other rules are reported, but H2 is decided on F-CI only (no selection among variants
on the test split).

**Important about "uncertainty".** F0–F2 also output a covariance trained with the heteroscedastic Gaussian NLL, so they
are not point estimators. What F-CI adds, and what H2 therefore tests, is *per-modality* uncertainty (each expert
reports its own covariance) and *uncertainty-weighted combination under unknown correlation*. H2 is a statement about
that structure, not about "having a covariance".

## 2. Data, seeds, scenarios

* Dataset `data/v1`: 600 train / 150 val / 150 test / 100 test_ood episodes (episode-level splits; test_ood starts the
  cube in a region never seen in training). Primary split: **test**. test_ood is secondary.
* **3 training seeds** (0, 1, 2) for every learned model. If fewer finish, the report states the actual number.
  Seeds are not added after seeing results (that would be optional stopping).
* Every scenario is degraded once with a fixed seed and shared by all models (`scripts/evaluate_all.py`).
* **Primary frame group: stage `coupled`** (cube lifted and held, the only stage in which every modality carries
  information about the cube). Reasons: in `free`, `hover`, `post` proprioception is gated off by design, so no fusion
  method can differ from the vision expert there; pooling these frames would dilute the comparison to a non-test. All
  other stages and the pooled "all frames" number are reported as secondary. This choice is made before test use.
* Scenario sets (names as in `src/mmperc/evaluation/scenarios.py`):
  * **O** (occlusion, H1): `occlusion L2`, `occlusion L4`, `random occl L4`.
  * **U** (an individual modality unreliable, H2): `occlusion L4`, `random occl L4`, `vis noise L3`, `illumination L3`,
    `blur L3`, `vision dropout L3`, `proprio noise L3`, `proprio bias L3`, `proprio dropout L3`.
  * **C** (unseen combinations, H3): `occl L3 + proprio noise L2`, `occl L4 + proprio bias L3`. Mixed-trained models
    never saw `vision_occlusion` together with `proprio_noise` or `proprio_bias` in training.

## 3. Statistics

* Unit of replication: the **episode** (frames within an episode are strongly correlated).
* Metrics per (scenario, model, frame group): xy RMSE (mm), yaw MAE (deg; chance = 22.5), NLL (4-D), mean Mahalanobis
  d² (4 = calibrated), |d² − 4|, 68 % / 95 % coverage.
* Seeds are combined by averaging the per-episode sums (= pooled metric of the seed-averaged squared errors).
  A comparison additionally has to be **sign-consistent in every seed** (`seed_sign_consistency`), because the episode
  bootstrap does not capture training-seed variance.
* Two methods are compared on the same episodes with a **paired cluster bootstrap** (5000 resamples, 95 % percentile CI,
  one-sided p = P\*(difference ≥ 0)). Error-type metrics: negative difference = candidate better.
* **Holm–Bonferroni** correction at α = 0.05 over the tests listed in each hypothesis (family-wise).
* "Significant" always means Holm-adjusted p < 0.05 *and* sign-consistent over seeds.
* Statistical significance is not practical significance: a difference below **1 mm** xy RMSE (≈ one rendered pixel
  at the camera's scale on the table) is reported as "negligible" whatever its p-value.

## 4. H1 — fusion improves state estimation over vision-only, particularly under occlusion

Candidate: F-CI. Reference: `vision_mixed`. Metric: xy RMSE, group `coupled`, split test.

* **Supported** iff (a) in at least 2 of the 3 scenarios of **O** the RMSE difference is significant and ≥ 1 mm in favour
  of F-CI (Holm over 3 tests), and (b) the *relative* gain ρ = 1 − RMSE(F-CI)/RMSE(vision) under `occlusion L4` is larger
  than under `clean` with a 95 % CI of the contrast excluding 0 (`relative_gain_contrast`; "particularly under occlusion").
* **Falsified** if (a) fails, i.e. fewer than 2 scenarios of **O** show a significant gain, **or** the gain is not
  larger under occlusion than clean (then fusion helps uniformly, or not at all, and the "particularly" part is wrong).
* **Partially supported**: (a) holds, (b) does not.
* Also reported, not part of the decision: `vision dropout L3` (vision absent in whole intervals: any proprioceptive
  expert beats the prior, so a gain there is expected and trivial); `clean` (non-inferiority: does fusion cost accuracy
  when nothing is wrong?); and the **stress scenario `occl L4 + proprio bias L3`**.
* **Prediction made on validation, stated so that a surprise is recognisable**: in the stress scenario F-CI is *not*
  better than vision-only (val seed 0: 12.8 mm vs 9.3 mm), because a constant proprioceptive bias is not removed by fusion,
  only reported as uncertainty. If this holds on the test split it is a limitation of H1 that the report must state, not
  hide.

## 5. H2 — explicit per-modality uncertainty improves fusion reliability when modalities become unreliable

Candidate: F-CI. Comparators: **F1** (`fusion_learned_mixed`) and **F2** (`fusion_gated_mixed`). Scenarios: set **U**
(9). Frame group `coupled`, split test. "Reliability" is measured by the **NLL** (primary: it rewards both accuracy and an
honest covariance); RMSE and |d² − 4| are co-reported.

* **Supported** iff, for *both* comparators, F-CI has a significantly lower NLL in at least **6 of the 9** scenarios
  (Holm over the 18 tests), **and** F-CI's RMSE is significantly worse (by ≥ 1 mm) than the comparator in at most 2
  scenarios.
* **Falsified** if for either comparator fewer than 6 of 9 scenarios reach a significant NLL gain.
* If NLL improves but RMSE is *not* better, the report says so explicitly: the benefit is then calibration (honest
  uncertainty), not accuracy.
* **H2b (independence assumption, secondary)**: F-CI vs F-PoE vs F-tPoE on |d² − 4| and NLL over **U**. Prediction from
  validation: |d² − 4| ordering CI ≤ tPoE ≤ PoE in the majority of **U**. This is the empirical test of the brief's
  requirement not to assume conditional independence between modality uncertainties.
* **H2c (classical comparison, secondary)**: F-CI vs `B6 WLS` and vs `B6c WLS+contact [oracle]` (classical fusion
  that is even told the true noise levels). The oracle row is an upper bound for the classical family, not a fair
  deployment baseline, and is labelled so.

## 6. H3 — training with controlled degradation improves generalisation to unseen combinations of failures

Candidate: `fusion_ci_mixed`. Reference: `fusion_ci` (identical architecture trained on clean data). Scenarios: set **C**.
Metric: NLL (primary), xy RMSE, |d² − 4|; group `coupled`; split test.

* **H3a (supported)** iff mixed training has a significantly lower NLL than clean training in **both** scenarios of **C**
  (Holm over 2).
* **H3b (is it really *combination* generalisation?)** — mixed training also teaches each single failure, so a gain on
  **C** can come from the singles alone. Control: `fusion_ci_mixed_allcombos` (same, but the held-out pairs are present in
  training). H3b holds iff F-CI-mixed is *non-inferior* to the control on **C**: upper 95 % CI bound of ΔNLL ≤ **0.5 nats**
  and of the RMSE ratio ≤ **1.25** (margins fixed here, before the test split).
* **Verdict**: *supported* = H3a and H3b; *partially supported* = H3a only ("mixed training helps, but seeing the
  combination helps further", the report quantifies by how much); *falsified* = H3a fails.
* Also reported: the cost of mixed training on `clean`; the same comparison for the unimodal models; test_ood.

## 7. Secondary analyses (descriptive, no pass/fail)

* Reliability diagrams (empirical vs nominal coverage), coverage 68/95 %, error vs predicted σ binned (correlation of
  predicted σ and realised error), per stage and scenario.
* Clean/degraded × train/test matrix for `vision`, `vision_mixed`, `proprio_gated(_mixed)`, `fusion_ci`, `fusion_ci_mixed`,
  `fusion_learned`, `fusion_learned_mixed`.
* Contact expert: `fusion_ci_mixed` vs `fusion_ci_mixed_nocontact` (paired), and test-time `--ablate contact|proprio|vision`
  on F-CI. The first prediction (from one validation seed: marginal value small) was **wrong** on the full 3-seed
  validation run: without the contact expert xy RMSE under `occlusion L4` is 5.9 mm vs 3.3 mm (NLL −15.6 vs −16.8). The
  revised expectation, fixed before the test split, is that the contact expert helps mainly when vision is degraded and
  is close to neutral on `clean` (1.7 vs 1.4 mm). Reported either way.
* Expert weights of F-CI per scenario and stage (`w_sum`, `avail_sum` in the result files).
* Per-stage tables (`free hover grasp coupled post`): stages where nothing is observable are prior-level *by
  construction*; the report does not present them as failures of any model.

## 7b. Validation observations recorded before the freeze (3 seeds, val split; nothing here was tuned on test)

The pre-registered rules were applied to the validation results as a dry run of the pipeline. They behaved as designed
and **no rule, scenario or threshold was changed afterwards**. Observations that the test split should confirm or refute:

* H1 rules pass on val, but the gain is almost uniform: relative gain 0.68 under `occlusion L4` vs 0.60 on `clean`
  (contrast CI [0.04, 0.13]): fusion is better than vision-only even when nothing is wrong, so "particularly under
  occlusion" is at best weakly supported. The stress scenario behaves as predicted in section 4 (13.5 vs 10.4 mm).
* **Pooled over all frames, F-CI is *not* better than vision-only under `occlusion L4`** (39.4 vs 34.0 mm): outside the
  `coupled` stage only the vision expert is informative, and the vision expert inside the fusion model is slightly worse
  than the stand-alone vision model. This is a limitation of the fused model, reported alongside the `coupled` result.
* H3a is trivially large on NLL: the clean-trained model is catastrophically over-confident under proprioceptive
  degradation (mean d² of 10^4–10^5). On accuracy the gap is much smaller (e.g. 5.1 vs 4.2 mm on `occl L3 + proprio noise L2`),
  and the clean-trained model is *more* accurate on `clean` (0.6 vs 1.4 mm). Mixed training buys calibration and
  robustness against proprioceptive faults at a cost in clean accuracy.
* H3b (margins 0.5 nats and ratio 1.25) fails narrowly on val (RMSE ratio upper bounds 1.26 and 1.29). The margins are
  kept as registered.
* Ordering of calibration on val: |d² − 4| of F-CI ≤ tPoE ≤ PoE in every scenario inspected (H2b prediction holds).

## 8. Known limitations fixed before the test (will be repeated in the report)

* Per-frame estimators, no temporal filter; the scripted policy makes the hand hover above the cube, so unrestricted
  proprioception leaks the cube position (handled by gating proprioception to the holding phase and by per-stage
  reporting; B4 ungated is not used for any claim).
* Under occlusion the CNN keeps a weak cue from the arm and the occluder patch (cube-invisible control on validation:
  7–11 mm after the hand approaches). `vision dropout` was added to measure fusion when vision is truly absent.
* Contact features exclude TCP x,y (leak found and removed in M7).
* One cube, one camera, simulation only; no claim about sim-to-real.
* Bit-exact reproducibility of *trained models* across machines is not guaranteed (GPU nondeterminism);
  data generation and degradations are deterministic given the seeds.

## 9. Deviations (append-only; date, what, why, effect on conclusions)

_none yet_

## 10. How the protocol is executed

```
python scripts/run_protocol.py --seeds 0 1 2 --out-root F:/runs/protocol        # resumable, seed-major
python scripts/evaluate_all.py --runs-root F:/runs/protocol --split test     --out F:/results/final
python scripts/evaluate_all.py --runs-root F:/runs/protocol --split test_ood --out F:/results/final
```
The test split is evaluated once, after the freeze of this file. If a bug is found afterwards the fix and the re-run are
recorded in section 9.
