from __future__ import annotations

from dataclasses import dataclass
import typing as t

import numpy as np
import torch
from torch import Tensor

from ._external import ensure_from_dc_on_path
from .intrinsics import PinholeIntrinsics

ensure_from_dc_on_path()

from dc_reality.splatting.utils.moge._utils import recover_intrinsic_with_shift  # noqa: E402
from dc_reality.splatting.utils.pixel_grid import pixel_grid_hw2  # noqa: E402


try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover
    least_squares = None


@dataclass(slots=True)
class SharedIntrinsicEstimate:
    intrinsics: PinholeIntrinsics
    depth_shifts: list[float]
    per_frame: list[dict[str, float]]
    residual_rms: float
    frames_used: int


class SharedIntrinsicEstimator:
    def __init__(self, *, max_points_per_frame: int = 30000, random_seed: int = 0) -> None:
        self.max_points_per_frame = int(max_points_per_frame)
        self.random_seed = int(random_seed)

    def fit(
        self,
        *,
        pointmaps: list[Tensor],
        masks: list[Tensor | None] | None = None,
    ) -> SharedIntrinsicEstimate:
        if not pointmaps:
            raise ValueError("Need at least one pointmap.")
        if masks is None:
            masks = [None] * len(pointmaps)
        if len(pointmaps) != len(masks):
            raise ValueError("pointmaps and masks must have the same length.")

        per_frame = [
            self._recover_single(pointmap=pointmap, mask=mask)
            for pointmap, mask in zip(pointmaps, masks)
        ]
        init = np.array(
            [
                np.median([r["focal_x"] for r in per_frame]),
                np.median([r["focal_y"] for r in per_frame]),
                np.median([r["cx"] for r in per_frame]),
                np.median([r["cy"] for r in per_frame]),
            ],
            dtype=np.float64,
        )
        t0 = np.array([r["t"] for r in per_frame], dtype=np.float64)

        if least_squares is None:
            params = init
            shifts = t0
            residual_rms = float("nan")
        else:
            sample_data = [
                self._sample_frame_data(pointmap=pointmap, mask=mask)
                for pointmap, mask in zip(pointmaps, masks)
            ]
            x0 = np.concatenate([init, t0], axis=0)
            result = least_squares(
                fun=self._shared_residual,
                x0=x0,
                args=(sample_data,),
                method="trf",
                loss="soft_l1",
                f_scale=5.0,
                max_nfev=200,
            )
            params = result.x[:4]
            shifts = result.x[4:]
            residual_rms = float(np.sqrt(np.mean(result.fun**2)))

        intrinsics = PinholeIntrinsics(
            focal_x=float(params[0]),
            focal_y=float(params[1]),
            cx=float(params[2]),
            cy=float(params[3]),
        )
        return SharedIntrinsicEstimate(
            intrinsics=intrinsics,
            depth_shifts=[float(x) for x in shifts],
            per_frame=per_frame,
            residual_rms=residual_rms,
            frames_used=len(pointmaps),
        )

    @staticmethod
    def _recover_single(*, pointmap: Tensor, mask: Tensor | None) -> dict[str, float]:
        _, result = recover_intrinsic_with_shift(points=pointmap, mask=mask)
        return {k: float(v) for k, v in result.items()}

    def _sample_frame_data(self, *, pointmap: Tensor, mask: Tensor | None) -> tuple[np.ndarray, ...]:
        height, width = pointmap.shape[:2]
        uv = pixel_grid_hw2(width=width, height=height, device=pointmap.device, dtype=pointmap.dtype)
        if mask is None:
            valid = torch.ones((height, width), dtype=torch.bool, device=pointmap.device)
        else:
            valid = mask.bool()
        pts = pointmap[valid]
        uvs = uv[valid]
        if pts.shape[0] > self.max_points_per_frame:
            generator = torch.Generator(device=pts.device)
            generator.manual_seed(self.random_seed)
            idx = torch.randperm(pts.shape[0], generator=generator, device=pts.device)[: self.max_points_per_frame]
            pts = pts[idx]
            uvs = uvs[idx]
        return (
            pts[:, 0].detach().cpu().numpy().astype(np.float64),
            pts[:, 1].detach().cpu().numpy().astype(np.float64),
            pts[:, 2].detach().cpu().numpy().astype(np.float64),
            uvs[:, 0].detach().cpu().numpy().astype(np.float64),
            uvs[:, 1].detach().cpu().numpy().astype(np.float64),
        )

    @staticmethod
    def _shared_residual(x: np.ndarray, sample_data: list[tuple[np.ndarray, ...]]) -> np.ndarray:
        fx, fy, cx, cy = x[:4]
        shifts = x[4:]
        residuals = []
        for t_i, (px, py, pz, u, v) in zip(shifts, sample_data):
            denom = pz + t_i
            valid = np.abs(denom) > 1e-6
            denom = denom[valid]
            px = px[valid]
            py = py[valid]
            u = u[valid]
            v = v[valid]
            residuals.append(fx * px / denom + cx - u)
            residuals.append(fy * py / denom + cy - v)
        if not residuals:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(residuals, axis=0)
