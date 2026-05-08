# pinhole_calib

Intrinsics-aware helpers around shared pinhole calibration, COLMAP bootstrap, separable refinement, and **bias-aware prior adapters** (see `论文.md` for the manuscript in this repo).

This repository is intended to stay **public**. It does **not** contain the **gaussian-splatting-dc** training codebase; that lives in a **separate private** repository.

## Current paper-alignment status

The current Stage A/B path is:

1. estimate a reliable shared pinhole `K` from COLMAP geometry plus separable refinement,
2. synthesize or recenter images with the correct principal-point sign convention,
3. run the foundation model on the appropriate camera convention,
4. prefer recenter-and-rerun priors when `K` is reliable, while keeping raw-output correction weak/gated.

Important: the smoke experiments now rely on a sibling MoGe checkout patch at
`/data/users/mia/current/MoGe`:

- `moge/model/v2.py`: `MoGeModel.infer(..., intrinsics=...)` accepts full normalized intrinsics, including off-center principal points.
- `moge/utils/geometry_torch.py`: `recover_shift_from_rays(...)` lets MoGe recover only z-shift when rays are known.

Without that MoGe-side injection, MoGe's default post-processing forces a centered principal point and self-estimates focal length per frame, which is not the camera convention required by `论文.md`.

Latest documented smoke outputs:

- 3-image injected-K run: `python/output/17/courtroom_stage_ab_intrinsics.json`
- 6-image injected-K run: `python/output/18/courtroom_stage_ab_intrinsics6.json`

See [docs/paper_alignment_log.md](docs/paper_alignment_log.md) for the implementation history and remaining gaps.

## Private dependency: `dc_reality`

`python/pinhole_calib` imports symbols from the **`dc_reality`** package shipped with **gaussian-splatting-dc**. Install one of:

```bash
pip install -e /path/to/private/gaussian-splatting-dc
```

or set the repo root:

```bash
export PINHOLE_CALIB_GAUSSIAN_SPLATTING_DC=/path/to/private/gaussian-splatting-dc
```

Optional local mirror (ignored by git): `from_dc/gaussian-splatting-dc/` with a `dc_reality/` directory inside.

Details: [docs/from_dc_integration.md](docs/from_dc_integration.md).

## Layout

- `python/pinhole_calib/` — library (`PriorAdapter`, intrinsics, COLMAP stage A, separable solver, …)
- `python/tests/` — unit tests (require `dc_reality` installed)
- `lean/` — Mathlib proofs around principal-point / BA sub-models
- `论文.md` — draft paper (Chinese)

## Tests

From the repo root, with `dc_reality` available:

```bash
pip install pytest  # if needed
PYTHONPATH=python pytest -q python/tests
```

The current unittest command used during paper-alignment work is:

```bash
PYTHONPATH=python /data/users/mia/conda_envs/dc_nksr/bin/python -m unittest python.tests.test_prior_adapter python.tests.test_local_ba python.tests.test_separable_solver python.tests.test_image_ops
```

## Lean

See [lean/README.md](lean/README.md).
