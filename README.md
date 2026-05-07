# pinhole_calib

Intrinsics-aware helpers around shared pinhole calibration, COLMAP bootstrap, separable refinement, and **bias-aware prior adapters** (see `论文.md` for the manuscript in this repo).

This repository is intended to stay **public**. It does **not** contain the **gaussian-splatting-dc** training codebase; that lives in a **separate private** repository.

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

## Lean

See [lean/README.md](lean/README.md).
