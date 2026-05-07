from __future__ import annotations

import os
import sys
from pathlib import Path


_ENV_ROOT = "PINHOLE_CALIB_GAUSSIAN_SPLATTING_DC"


def ensure_from_dc_on_path() -> Path:
    """Ensure the private `gaussian-splatting-dc` checkout is importable as `dc_reality`.

    Resolution order:
    1. `dc_reality` already on PYTHONPATH / installed (`pip install -e ...`).
    2. Directory from env ``PINHOLE_CALIB_GAUSSIAN_SPLATTING_DC`` (repo root containing `dc_reality/`).
    3. Optional local mirror: ``<this-repo>/from_dc/gaussian-splatting-dc`` (gitignored in the public tree).

    Raises ImportError with setup hints if nothing resolves.
    """
    try:
        import dc_reality  # type: ignore  # noqa: PLC0415

        return Path(dc_reality.__file__).resolve().parents[1]
    except ImportError:
        pass

    env = os.environ.get(_ENV_ROOT)
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = (
            Path(__file__).resolve().parents[2]
            / "from_dc"
            / "gaussian-splatting-dc"
        )

    if not (root / "dc_reality").is_dir():
        raise ImportError(
            "The `dc_reality` package is required but was not found. "
            "Install the private gaussian-splatting-dc dependency, for example:\n"
            "  pip install -e /path/to/gaussian-splatting-dc\n"
            "or set the repository root explicitly:\n"
            f"  export {_ENV_ROOT}=/path/to/gaussian-splatting-dc\n"
            "See README.md for the public/private repo split."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
