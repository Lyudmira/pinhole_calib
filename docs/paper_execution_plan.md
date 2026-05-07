# Paper Execution Plan

## Goal

The goal is not to make the code "sound like the paper". The goal is to satisfy the paper's method claims in the repository, in dependency order, with explicit stop conditions when a stage fails.

This plan is strict:

- We do **not** advance to the next stage if the current stage fails its acceptance checks.
- If a stage regresses metrics or fails to deliver the intended behavior, we stop, diagnose, and revise that stage.
- We only treat a stage as complete when both unit tests and targeted smoke checks support it.

## Non-Negotiable Evaluation Rules

Every stage must define:

- `deliverable`: what code exists after the stage
- `acceptance`: what must be true to move on
- `failure action`: what to inspect if acceptance is not met

Baseline commands:

- Unit tests:
  - `PYTHONPATH=python /data/users/mia/conda_envs/dc_nksr/bin/python -m unittest python.tests.test_prior_adapter python.tests.test_local_ba python.tests.test_separable_solver python.tests.test_image_ops`
- Courtroom smoke:
  - `PYTHONPATH=python /data/users/mia/conda_envs/dc_nksr/bin/python python/run_courtroom_stage_ab.py --num-images 3 --seed 7 --use-fp16 --output python/output/courtroom_stage_ab_smoke.json`

Baseline reference before this plan:

- Stable smoke behavior is represented by `python/output/9/courtroom_stage_ab_smoke.json`
- Known bad regression examples are `python/output/5/...` and `python/output/6/...`

## Stage 0: Freeze the Evaluation Contract

### Deliverable

- A stable definition of what counts as progress for the current codebase.
- For `stage_ab`, every run must report:
  - geometry-only estimate
  - any new joint estimate
  - aggregate corrected/rerun metrics
  - prior-state and gate diagnostics

### Acceptance

- The script remains reproducible under the same seed.
- The reporting surface is rich enough to compare new stages against the stable baseline.

### Failure Action

- If a later stage cannot be judged fairly because the reporting is insufficient, expand reporting before changing the optimizer.

Status:

- Already largely satisfied.

## Stage 1: Minimal Joint Objective Over Shared K With Corrected Priors

### Why this is first

The paper's missing core is not "more helper functions". It is the absence of an actual joint objective that combines geometry with corrected FM priors. The first executable step is therefore a minimal joint optimization loop.

### Deliverable

- A new shared-intrinsics optimizer that minimizes a combined objective:
  - geometry reprojection term
  - corrected pointmap intrinsics-consistency term
  - corrected depth-derived pointmap intrinsics-consistency term
  - corrected normal/pointmap consistency term
- This stage keeps COLMAP poses/structure fixed.
- This stage does **not** yet claim to be the full paper solver.

### Acceptance

- Synthetic unit test: when priors are consistent and geometry is biased/noisy, the joint optimizer improves shared-`K` error versus geometry-only solve.
- Courtroom smoke: the joint estimate must not clearly degrade the stable baseline.
  - If it is worse than the geometry-only/separable estimate in the synthetic-shift experiment, it fails.
- The optimizer must report per-term residual breakdowns so failures are diagnosable.

### Failure Action

- Inspect term scaling first.
- Inspect whether the prior terms are actually informative or only restating the same geometry.
- Inspect whether the optimizer is overfitting priors or drifting because of weak conditioning.
- Do **not** proceed to Stage 2 until the joint objective is either:
  - accepted, or
  - explicitly redesigned and re-tested.

## Stage 2: Use BA Schur Response for Pose-Prior Correction

### Deliverable

- Replace near-axis pose correction fallback, when possible, with Schur-response-driven pose correction.
- Wire the response matrices into a real pose-prior path, not just diagnostics.

### Acceptance

- Unit tests verify that response-matrix-based correction behaves correctly on synthetic local BA setups.
- Smoke runs show pose-prior correction is numerically stable and does not destabilize the shared-`K` estimate.

### Failure Action

- Check sign conventions in SE(3) updates.
- Check anchor handling in the local BA system.
- Check whether the Schur response is being used outside its local-validity regime.

## Stage 3: Add Depth Affine Parameters as Optimized Variables

### Deliverable

- Depth affine correction `(a_t, b_t)` is no longer API-only.
- It becomes a real estimated variable or alternating-update block inside the optimization process.

### Acceptance

- Synthetic tests show depth priors with affine ambiguity are corrected better than with pure pixel-domain shift only.
- Courtroom smoke does not regress corrected-depth metrics relative to the stable baseline.

### Failure Action

- Check whether affine depth is being estimated from a geometrically meaningful signal.
- Check whether the scale/bias block is underconstrained and needs regularization or alternation.

## Stage 4: Turn Prior-State Classification Into Real Scheduling

### Deliverable

- `corrected`, `debiased`, `filtered`, `rerun` become actual control flow in optimization, not summary labels.
- Different prior channels can be admitted, downweighted, or excluded based on the current state.

### Acceptance

- The script actually changes behavior when the state changes.
- Unit tests verify that scheduling decisions match the configured thresholds and available data.
- Smoke runs show the schedule chooses the same path the diagnostics recommend.

### Failure Action

- Check whether state thresholds are physically meaningful.
- Check whether scheduling logic is hiding optimizer failures instead of exposing them.

## Stage 5: Expand From Fixed Geometry to a Real Joint Solve

### Deliverable

- Shared intrinsics, corrected prior terms, and geometry terms live in the same optimization loop.
- The code begins moving from "diagnostic scaffold" to the paper's actual joint method.
- If feasible in the repo, this stage should start relaxing at least part of the fixed geometry assumption.

### Acceptance

- The joint solve improves or matches the earlier fixed-geometry version on the synthetic-shift experiment.
- The optimization remains numerically stable across repeated smoke runs.

### Failure Action

- Reduce scope and identify which blocks are destabilizing the solve:
  - intrinsics
  - pose
  - structure
  - prior weights

## Stage 6: Only Then Revisit Rerun-Based Replacement

### Deliverable

- Recenter-and-rerun FM priors are only promoted when the estimated `K` is actually reliable.
- This is used as a justified late-stage path, not as an early shortcut.

### Acceptance

- `rerun` is recommended only when the state is truly stable.
- On smoke tests where rerun is chosen, its selected metrics should justify the choice.

### Failure Action

- Check whether rerun quality is being overpredicted by state heuristics.
- Check whether recentering is incomplete for focal mismatches.

## Hard Stop Rule

If Stage 1 does not pass, the project is not allowed to claim paper alignment and is not allowed to proceed to Stage 2.

That is the current execution rule from this point onward.
