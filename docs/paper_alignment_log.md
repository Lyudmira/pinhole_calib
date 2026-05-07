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

## Tests Run

Repeated after each chunk:

- `PYTHONPATH=python /data/users/mia/conda_envs/dc_nksr/bin/python -m unittest python.tests.test_prior_adapter python.tests.test_local_ba python.tests.test_separable_solver python.tests.test_image_ops`

Observed status at the end of this log:

- `23 tests`
- `OK`

## Current State

What is now implemented and behaving sensibly:

- shared-intrinsics separable solve with uncertainty diagnostics
- prior-state classification
- deterministic scalar gating with unitless intrinsics error
- shared-K BA response/probe utilities
- per-channel geometry residuals and gates for `pointmap/depth/normals`
- non-overwriting numbered `stage_ab` work dirs and output dirs
- experiment summaries that record the recommended prior mode

What is still **not** done, relative to `论文.md`:

- the full joint objective combining correspondence terms, reprojection terms, shared intrinsics, and corrected FM priors
- using BA Schur response inside an actual joint optimizer, rather than only as a probe
- a real estimated source for per-frame depth affine `(a_t, b_t)` inside the optimization loop
- pose-prior correction driven by real FM pose outputs in `stage_ab`
- a true "corrected vs filtered vs rerun" optimization schedule, instead of only summary-time recommendation

## Recommended Next Order

If continuing from here, the next steps should be:

1. Build a real joint optimization core that keeps the reliable separable `K` initialization but adds corrected FM prior residuals as weak terms instead of swapping in BA outputs directly.
2. Feed Schur response matrices from the local BA linearization into pose-prior correction where pose FM outputs are available.
3. Estimate or optimize depth affine parameters `(a_t, b_t)` from geometry-consistent signals instead of keeping them as an API-only capability.
4. Turn `recommended_prior_mode` from a reporting label into an actual optimization schedule.

The key lesson so far:

- "more BA" is not automatically "more paper-aligned".
- The paper's promise comes from the **right** joint objective and response-aware prior handling, not from replacing a trustworthy calibration estimate with an underconstrained BA refinement.
