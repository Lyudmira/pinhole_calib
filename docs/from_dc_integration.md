# Integration with private `gaussian-splatting-dc`

This repo imports **`dc_reality`** from a separate checkout of **gaussian-splatting-dc** (same codebase as the internal dConstruct Gaussian splatting project). That dependency is **private** and must not be vendored in the public `pinhole_calib` tree.

## Setup

1. Clone the private `gaussian-splatting-dc` repository (SSH recommended).
2. Either:
   - **Editable install (preferred):** `pip install -e /path/to/gaussian-splatting-dc`
   - **Environment variable:** `export PINHOLE_CALIB_GAUSSIAN_SPLATTING_DC=/path/to/gaussian-splatting-dc`
   - **Optional mirror:** clone into `pinhole_calib/from_dc/gaussian-splatting-dc` (this path is gitignored).

Then install and run tests for `pinhole_calib` with `PYTHONPATH` including `pinhole_calib/python` or `pip install -e python` if you add a package config.

## Why this split

`pinhole_calib` (public) hosts intrinsics-aware helpers, COLMAP wrappers, prior adapters, and Lean formalization. **gaussian-splatting-dc** (private) contains training pipelines, licensing, and proprietary assets referenced by `dc_reality`.

## Notes migrated from the old `from_dc/guide.md`

Paths below are relative to the **gaussian-splatting-dc** repository root.

**Relevant `dc_reality` pieces**

- Full off-center pinhole: `dc_reality/splatting/cameras/base.py`, `dc_reality/splatting/utils/graphics_utils.py`
- Differentiable intrinsics: `dc_reality/splatting/cameras/delta.py`, `dc_reality/splatting/cameras/optimizers/intrinsics.py`
- Shared K: `dc_reality/splatting/cameras/converters/vanilla.py`, `dc_reality/splatting/cameras/converters/rig.py`
- Projection / PGSR rays: `dc_reality/splatting/utils/projection.py`, `dc_reality/splatting/renderers/gsplat_renderer/pgsr_renderer.py`
- COLMAP BA refine principal point: `dc_reality/toolbox/run_colmap.py`
- Undistort + recenter: `dc_reality/toolbox/undistort_image.py`, tests under `dc_reality/tests/`

**Closest prototypes inside the private repo**

- MiniBA: `dc_reality/contrib/ontheflynvs_dc/mini_ba.py`
- BA with `fx,fy,cx,cy`: `dc_reality/contrib/ontheflynvs_dc/Levenberg_Marquardt/ba_optimizer.py`
- MoGe pointmap → intrinsics: `dc_reality/splatting/utils/moge/_utils.py`
- Curvature / Hessian tooling: `dc_reality/contrib/fundamental/second_order_gradient/hessian_utils.py`

**Still missing for the paper pipeline (to build in `pinhole_calib` or upstream)**

- Larsson-style P4.5Pfuv / P5Pfuva / P7Pfruv solvers
- Linear separable `cx,cy` sub-step when pose/structure are fixed (beyond the current nonlinear `SeparableSharedIntrinsicSolver`)
- Full geometry-aware correction of foundation outputs in the trainer (beyond `MiscalibrationPriorAdapter`)
- Depth prior that flags systematic bending under wrong `K` (see `dc_reality/splatting/trainer/criteria/depths.py` today)

**Roadmap sketch**

1. Bootstrap with shared-K cameras + COLMAP / rig BA from `dc_reality`.
2. Harden LM/Schur layer with shared `fx,fy,cx,cy` (extend `BAOptimizer` / MiniBA).
3. Add separable least-squares substeps for principal point where conditioning allows.
4. Apply miscalibration-aware correction or gating before FM losses.

Optional reference: `ref/Camera Pose Estimation with Unknown Principal Point.md` for solver ideas.
