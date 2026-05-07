from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import typing as t

import numpy as np

from .intrinsics import PinholeIntrinsics

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover
    least_squares = None


@dataclass(slots=True)
class SeparableSolveResult:
    intrinsics: PinholeIntrinsics
    residual_rms: float
    num_observations: int


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def load_colmap_tracks(txt_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points3d = {}
    with open(txt_dir / "points3D.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            xyz = np.array(list(map(float, parts[1:4])), dtype=np.float64)
            points3d[point_id] = xyz

    poses = {}
    observations_xyz = []
    observations_uv = []
    with open(txt_dir / "images.txt") as f:
        lines = [line.rstrip("\n") for line in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        poses[image_id] = (_qvec_to_rotmat(qvec), tvec)
        if i >= len(lines):
            break
        points_line = lines[i].strip()
        i += 1
        if not points_line:
            continue
        vals = points_line.split()
        triples = [vals[j : j + 3] for j in range(0, len(vals), 3)]
        R, t = poses[image_id]
        for x_s, y_s, pid_s in triples:
            pid = int(pid_s)
            if pid < 0 or pid not in points3d:
                continue
            xyz_world = points3d[pid]
            xyz_cam = R @ xyz_world + t
            if xyz_cam[2] <= 1e-6:
                continue
            observations_xyz.append(xyz_cam)
            observations_uv.append([float(x_s), float(y_s)])

    return (
        np.asarray(observations_xyz, dtype=np.float64),
        np.asarray(observations_uv, dtype=np.float64),
        np.asarray([len(observations_uv)], dtype=np.int64),
    )


class SeparableSharedIntrinsicSolver:
    def fit(
        self,
        *,
        camera_xyz: np.ndarray,
        observed_uv: np.ndarray,
        init_intrinsics: PinholeIntrinsics,
    ) -> SeparableSolveResult:
        if least_squares is None:
            raise RuntimeError("scipy is required for SeparableSharedIntrinsicSolver")
        x0 = np.array(
            [
                init_intrinsics.focal_x,
                init_intrinsics.focal_y,
                init_intrinsics.cx,
                init_intrinsics.cy,
            ],
            dtype=np.float64,
        )
        result = least_squares(
            fun=self._residual,
            x0=x0,
            args=(camera_xyz, observed_uv),
            method="trf",
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=200,
        )
        fx, fy, cx, cy = result.x
        return SeparableSolveResult(
            intrinsics=PinholeIntrinsics(focal_x=float(fx), focal_y=float(fy), cx=float(cx), cy=float(cy)),
            residual_rms=float(math.sqrt(np.mean(result.fun ** 2))),
            num_observations=int(len(observed_uv)),
        )

    @staticmethod
    def _residual(params: np.ndarray, camera_xyz: np.ndarray, observed_uv: np.ndarray) -> np.ndarray:
        fx, fy, cx, cy = params
        z = camera_xyz[:, 2]
        pred_u = fx * camera_xyz[:, 0] / z + cx
        pred_v = fy * camera_xyz[:, 1] / z + cy
        return np.concatenate([pred_u - observed_uv[:, 0], pred_v - observed_uv[:, 1]], axis=0)
