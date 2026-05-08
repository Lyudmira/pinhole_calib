# Paper Alignment Log

## Scope

This log records the recent paper-alignment work on `pinhole_calib`, in the order it was implemented and re-tested. The goal was to move the code closer to the workflow described in `论文.md` sections 8.3-8.7 without hiding regressions behind "paper-sounding" shortcuts.

All tests and smoke runs below used:

- `PYTHONPATH=python /data/users/mia/conda_envs/dc_nksr/bin/python`

## Phase 1: Stabilize Stage A uncertainty and gating inputs

Implemented:

- Added covariance/conditioning diagnostics to `SeparableSharedIntrinsicSolver`.
- Added `PriorStateConfig` / `PriorStateAssessment`.
- Added unitless intrinsics error for deterministic gating.
- Added shared-K Schur response utilities in `python/pinhole_calib/local_ba.py`.

Why:

- `论文.md` requires classification of priors based on current `K` reliability, not just a raw principal-point delta.
- The earlier gate was too close to a placeholder and could saturate for the wrong reasons.

Verification:

- Unit tests passed after this phase.
- Smoke outputs before later BA experiments showed:
  - `python/output/4/courtroom_stage_ab_smoke.json`
  - `prior_state.state = "debiased"`
  - `cx/cy` estimation error stayed near zero (`~0.01 px` scale)
  - aggregate gate `~0.84946`

Conclusion:

- Good change.
- This phase improved diagnostics and fixed the earlier "gate collapses for the wrong reason" behavior.

## Phase 2: Add shared-K BA probe

Implemented:

- Added `SharedKBundleAdjuster` and `SharedKBundleAdjustmentResult`.
- Initially wired its output directly into `run_courtroom_stage_ab.py` as the final estimated intrinsics.

What went wrong:

- First attempt used unshifted COLMAP observations inside BA while `stage_ab` was solving a shifted-principal-point problem.
- That produced a clear regression:
  - `python/output/5/courtroom_stage_ab_smoke.json`
  - `cx` error `~4.97 px`
  - `cy` error `~32.84 px`
- This was a real bug, not noise.

Fix:

- Added `LocalBAProblem.with_observation_offset(...)`.
- Shifted BA observations by `[delta_cx, delta_cy]` just like the separable solver.
- Added regression coverage in `python/tests/test_local_ba.py`.

Result after fixing the observation path:

- `python/output/6/courtroom_stage_ab_smoke.json`
- Catastrophic drift disappeared, but BA still degraded the final intrinsics versus the separable estimate:
  - `cx` error `~-5.82 px`
  - `cy` error `~2.30 px`
- BA reprojection RMS improved slightly, but the recovered `K` got worse.

Interpretation:

- A free shared-K BA probe is useful as a local response/conditioning tool.
- In the current codebase it is **not** safe to treat that BA result as the final `K`.
- Doing so would be pretending we have the paper's full joint objective when we do not.

## Phase 3: Reposition BA as probe, not final estimator

Implemented:

- Kept BA in `stage_ab`, but only as a probe/diagnostic.
- Restored `separable.intrinsics` as the final estimated `K`.
- Kept BA outputs under `shared_k_ba_probe`.
- Kept Schur-response stability metrics, but evaluated them around the trusted separable estimate.

Verification:

- Unit tests passed.
- Smoke run:
  - `python/output/7/courtroom_stage_ab_smoke.json`
- Metrics returned to the stable pre-BA behavior:
  - `cx` error `~0.0096 px`
  - `cy` error `~-0.0019 px`
  - aggregate metrics matched the stable earlier run

Conclusion:

- Good change.
- This is currently the honest alignment with the paper: use BA linearization/response information, but do not substitute a weaker optimizer for the full joint solver.

## Phase 4: Add per-channel geometry diagnostics and gates

Implemented:

- Extended `prior_adapter.py` with:
  - per-channel gating signals for `pointmap`, `depth`, and `normals`
  - `normal_pointmap_consistency_residual(...)`
- Extended `stage_ab` to compute:
  - pointmap correspondence residual
  - depth-derived pointmap correspondence residual
  - normal-vs-pointmap local consistency residual
  - channelwise gates
- Added/updated tests in:
  - `python/tests/test_prior_adapter.py`

Verification:

- `23` tests passed.
- Smoke run:
  - `python/output/8/courtroom_stage_ab_smoke.json`
- Aggregate metrics did not regress.
- New diagnostics were numerically sensible:
  - `pointmap_gate ~ 0.84946`
  - `depth_gate ~ 0.85092`
  - `normal_gate ~ 0.82651`

Interpretation:

- The script can now distinguish that the normal channel is less trustworthy than the pointmap/depth channels on this smoke run.
- This is much closer to the paper than a single scalar gate.

## Phase 5: Make prior-state output actionable in the experiment summary

Implemented:

- Added `recommended_prior_mode` to `stage_ab`.
- Added `selected_metrics` based on the current recommended mode:
  - `corrected`
  - `rerun`
  - or `filtered` (returns `null` metrics)

Verification:

- `23` tests passed again.
- Smoke run:
  - `python/output/9/courtroom_stage_ab_smoke.json`
- Current recommendation:
  - `recommended_prior_mode = "corrected"`
  - `selected_metrics = {point_rmse: 1.4279, depth_mae: 1.2947, normal_deg: 23.3715}`

Interpretation:

- On the current 3-image smoke case, the code does **not** recommend rerunning the FM model on recentered inputs.
- That is consistent with:
  - `prior_state.state = "debiased"`
  - `rerun_recommended = false`

Superseded by Phase 6:

- This interpretation was based on a flawed synthetic-image sign convention and an overly conservative rerun classifier.
- Keep the run as historical evidence, not as a current conclusion about the paper.

## Phase 6: Fix synthetic principal-point sign and rerun classification

Implemented:

- Fixed `run_courtroom_stage_ab.py` so the synthetic shifted image and shifted COLMAP observations represent the same camera:
  - `shift_image_principal_point(image, delta)` applies the inverse image warp for principal point `-delta`.
  - The experiment now passes `-delta_cx, -delta_cy` when synthesizing images whose physical camera is `base + delta`.
- Added a regression test in `python/tests/test_image_ops.py`.
- Changed `PriorStateAssessment` classification so a large but reliably estimated intrinsics mismatch recommends recenter-and-rerun when rerun is available.
- Changed high Schur condition to block raw/debiased response use, not the safer recenter-and-rerun path.

Why:

- The old setup compared MoGe outputs from an image shifted one way against geometry shifted the other way.
- The old classifier treated large principal-point correction as a reason not to rerun, but the paper's safest path is exactly to rerun once `K` is reliable.

Verification:

- `24` tests passed.
- 3-image smoke:
  - `python/output/14/courtroom_stage_ab_smoke.json`
  - `recommended_prior_mode = "rerun"`
  - `rerun_point_rmse = 0.6243`
  - `rerun_depth_mae = 0.5383`
  - `rerun_normal_deg = 4.6794`
- 6-image smoke:
  - `python/output/16/courtroom_stage_ab_smoke6.json`
  - `recommended_prior_mode = "rerun"`
  - `rerun_point_rmse = 0.5115`
  - `rerun_depth_mae = 0.4462`
  - `rerun_normal_deg = 4.2867`

Interpretation:

- The earlier "rerun does not help" conclusion was an implementation artifact.
- Recenter-and-rerun is now strongly supported in the smoke experiments.
- Direct geometric correction of raw shifted MoGe outputs remains weak, which is consistent with the paper's warning that model outputs generated under the wrong camera convention are biased responses, not simple unbiased measurements.

## Phase 7: Inject full intrinsics into MoGe inference

Implemented outside this public repo, in the sibling MoGe checkout:

- Added `intrinsics=` to `MoGeModel.infer(...)` in `/data/users/mia/current/MoGe/moge/model/v2.py`.
- Added `recover_shift_from_rays(...)` in `/data/users/mia/current/MoGe/moge/utils/geometry_torch.py`.
- When full normalized `K` is supplied, MoGe no longer:
  - estimates focal length from the point map,
  - forces `cx = cy = 0.5`,
  - assumes centered/isometric rays in its post-processing.
- Instead, it uses the supplied full `K` to construct per-pixel rays, recovers only the z-shift, and recomputes the point map under that `K`.
- Updated `run_courtroom_stage_ab.py` to pass:
  - `base_shared` for baseline images,
  - `estimated` for shifted images,
  - `base_shared` for recentered/rerun images.
- Added `moge_intrinsics_injected = true` to the experiment summary.

Why:

- MoGe's default `infer()` post-processing always returned centered-principal-point intrinsics and self-estimated per-frame focal length.
- That violated the paper's requirement that foundation-model priors enter under an explicit camera convention.
- External image recentering alone was not enough; the FM post-processing also had to obey the supplied camera model.

Verification:

- MoGe files compile with `py_compile`.
- Synthetic ray/shift check recovered a known z-shift with error about `6e-8`.
- `24` public repo tests passed.
- A direct check confirmed MoGe now returns the injected pixel intrinsics exactly in the wrapper.
- 3-image intrinsics-injected smoke:
  - `python/output/17/courtroom_stage_ab_intrinsics.json`
  - `recommended_prior_mode = "rerun"`
  - `rerun_point_rmse = 0.2601`
  - `rerun_depth_mae = 0.1222`
  - `rerun_normal_deg = 4.6794`
- 6-image intrinsics-injected smoke:
  - `python/output/18/courtroom_stage_ab_intrinsics6.json`
  - `recommended_prior_mode = "rerun"`
  - `rerun_point_rmse = 0.3508`
  - `rerun_depth_mae = 0.1983`
  - `rerun_normal_deg = 4.2780`

Interpretation:

- This is the closest current implementation to the paper's safe path:
  1. estimate reliable shared `K`,
  2. recenter images to the target camera,
  3. run the foundation model with the target full intrinsics injected,
  4. use the resulting outputs as corrected priors.
- The large gain from Phase 6 to Phase 7, especially in point/depth metrics, shows that MoGe's post-processing camera convention was a real blocker.
- Direct correction of raw shifted MoGe outputs is still not competitive with rerun, so it should remain weak/gated unless a model-specific response calibration is added.

## Tests Run

Repeated after each chunk:

- `PYTHONPATH=python /data/users/mia/conda_envs/dc_nksr/bin/python -m unittest python.tests.test_prior_adapter python.tests.test_local_ba python.tests.test_separable_solver python.tests.test_image_ops`

Observed status at the end of this log:

- `24 tests`
- `OK`

## Current State

What is now implemented and behaving sensibly:

- shared-intrinsics separable solve with uncertainty diagnostics
- prior-state classification
- deterministic scalar gating with unitless intrinsics error
- shared-K BA response/probe utilities
- per-channel geometry residuals and gates for `pointmap/depth/normals`
- synthetic principal-point image generation with the correct sign convention
- safe rerun recommendation when `K` is reliable
- MoGe v2 inference post-processing with injected full normalized intrinsics in the sibling MoGe checkout
- non-overwriting numbered `stage_ab` work dirs and output dirs
- experiment summaries that record the recommended prior mode

What is still **not** done, relative to `论文.md`:

- the full joint objective combining correspondence terms, reprojection terms, shared intrinsics, and corrected FM priors
- using BA Schur response inside an actual joint optimizer, rather than only as a probe
- a real estimated source for per-frame depth affine `(a_t, b_t)` inside the optimization loop
- pose-prior correction driven by real FM pose outputs in `stage_ab`
- a true joint "corrected vs filtered vs rerun" optimization schedule beyond summary-time recommendation and the Stage A/B smoke path

## Recommended Next Order

If continuing from here, the next steps should be:

1. Treat recenter-and-rerun with injected full MoGe intrinsics as the primary prior path.
2. Preserve direct raw-output correction only as a weak/gated fallback unless MoGe-specific response calibration proves it reliable.
3. Build a real joint optimization core that keeps the reliable separable `K` initialization but adds corrected FM prior residuals as weak terms.
4. Feed Schur response matrices from the local BA linearization into pose-prior correction where pose FM outputs are available.
5. Estimate or optimize depth affine parameters `(a_t, b_t)` from geometry-consistent signals instead of keeping them as an API-only capability.
6. Turn `recommended_prior_mode` from a reporting label into an actual optimization schedule.

The key lesson so far:

- "more BA" is not automatically "more paper-aligned".
- The paper's promise comes from the **right** camera convention, joint objective, and response-aware prior handling, not from replacing a trustworthy calibration estimate with an underconstrained BA refinement.
