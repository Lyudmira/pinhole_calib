from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor

from .intrinsics import PinholeIntrinsics
from .prior_adapter import MiscalibrationPriorAdapter
from .separable_solver import SeparableSharedIntrinsicSolver

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover
    least_squares = None


@dataclass(slots=True)
class JointFramePriors:
    image_name: str
    pointmap: Tensor
    depth: Tensor
    normals: Tensor
    mask: Tensor | None = None
    observed_uv: np.ndarray | None = None
    camera_xyz: np.ndarray | None = None
    pointmap_weight_scale: float = 1.0
    depth_weight_scale: float = 1.0
    normal_weight_scale: float = 1.0


@dataclass(slots=True)
class JointIntrinsicProblem:
    camera_xyz: np.ndarray
    observed_uv: np.ndarray
    assumed_intrinsics: PinholeIntrinsics
    frame_priors: list[JointFramePriors]


@dataclass(slots=True)
class JointIntrinsicSolveResult:
    intrinsics: PinholeIntrinsics
    residual_rms: float
    geometry_rms: float
    pointmap_prior_rms: float
    depth_prior_rms: float
    normal_prior_rms: float
    iterations: int
    converged: bool


class JointSharedIntrinsicOptimizer:
    def __init__(
        self,
        *,
        geometry_weight: float = 1.0,
        pointmap_weight: float = 0.1,
        depth_weight: float = 0.05,
        normal_weight: float = 0.02,
        focal_prior_weight: float = 1.0,
        max_nfev: int = 60,
        loss: str = "soft_l1",
        f_scale: float = 1.0,
        diff_step: float = 1.0,
    ) -> None:
        self.geometry_weight = float(geometry_weight)
        self.pointmap_weight = float(pointmap_weight)
        self.depth_weight = float(depth_weight)
        self.normal_weight = float(normal_weight)
        self.focal_prior_weight = float(focal_prior_weight)
        self.max_nfev = int(max_nfev)
        self.loss = str(loss)
        self.f_scale = float(f_scale)
        self.diff_step = float(diff_step)

    def fit(
        self,
        *,
        problem: JointIntrinsicProblem,
        init_intrinsics: PinholeIntrinsics,
    ) -> JointIntrinsicSolveResult:
        if least_squares is None:
            raise RuntimeError("scipy is required for JointSharedIntrinsicOptimizer")

        x0 = np.array(
            [
                float(init_intrinsics.focal_x),
                float(init_intrinsics.focal_y),
                float(init_intrinsics.cx),
                float(init_intrinsics.cy),
            ],
            dtype=np.float64,
        )
        result = least_squares(
            fun=self._residual,
            x0=x0,
            args=(problem, x0.copy()),
            method="trf",
            loss=self.loss,
            f_scale=self.f_scale,
            max_nfev=self.max_nfev,
            diff_step=self.diff_step,
            bounds=(
                np.array([1e-6, 1e-6, -np.inf, -np.inf], dtype=np.float64),
                np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64),
            ),
        )
        intrinsics = PinholeIntrinsics(
            focal_x=float(result.x[0]),
            focal_y=float(result.x[1]),
            cx=float(result.x[2]),
            cy=float(result.x[3]),
        )
        term_breakdown = self.evaluate_terms(problem=problem, intrinsics=intrinsics)
        return JointIntrinsicSolveResult(
            intrinsics=intrinsics,
            residual_rms=float(math.sqrt(np.mean(result.fun ** 2))) if result.fun.size else 0.0,
            geometry_rms=term_breakdown["geometry_rms"],
            pointmap_prior_rms=term_breakdown["pointmap_prior_rms"],
            depth_prior_rms=term_breakdown["depth_prior_rms"],
            normal_prior_rms=term_breakdown["normal_prior_rms"],
            iterations=int(result.nfev),
            converged=bool(result.success),
        )

    def evaluate_terms(
        self,
        *,
        problem: JointIntrinsicProblem,
        intrinsics: PinholeIntrinsics,
    ) -> dict[str, float]:
        adapter = MiscalibrationPriorAdapter(assumed_intrinsics=problem.assumed_intrinsics)
        geometry_residual = self._normalized_geometry_residual(
            intrinsics=intrinsics,
            camera_xyz=problem.camera_xyz,
            observed_uv=problem.observed_uv,
        )
        pointmap_residuals = []
        depth_residuals = []
        normal_residuals = []
        for frame in problem.frame_priors:
            corrected = adapter.correct_priors(
                current_intrinsics=intrinsics,
                pointmap=frame.pointmap,
                depth=frame.depth,
                normals=frame.normals,
            )
            depth_pointmap = adapter.corrected_pointmap_from_depth(
                frame.depth,
                current_intrinsics=intrinsics,
            )
            if frame.observed_uv is not None and frame.camera_xyz is not None:
                if frame.pointmap_weight_scale > 0.0:
                    pointmap_residuals.append(
                        float(
                            frame.pointmap_weight_scale
                            * adapter.pointmap_correspondence_residual(
                                corrected.pointmap,
                                observed_uv=frame.observed_uv,
                                camera_xyz=frame.camera_xyz,
                                mask=frame.mask,
                            ).item()
                        )
                    )
                if frame.depth_weight_scale > 0.0:
                    depth_residuals.append(
                        float(
                            frame.depth_weight_scale
                            * adapter.pointmap_correspondence_residual(
                                depth_pointmap,
                                observed_uv=frame.observed_uv,
                                camera_xyz=frame.camera_xyz,
                                mask=frame.mask,
                            ).item()
                        )
                    )
            normal_residuals.append(
                float(
                    frame.normal_weight_scale
                    * adapter.normal_pointmap_consistency_residual(
                        corrected.normals,
                        pointmap=corrected.pointmap,
                        mask=frame.mask,
                    ).item()
                )
            )

        return {
            "geometry_rms": float(math.sqrt(np.mean(geometry_residual ** 2)))
            if geometry_residual.size
            else 0.0,
            "pointmap_prior_rms": float(math.sqrt(np.mean(np.square(pointmap_residuals))))
            if pointmap_residuals
            else 0.0,
            "depth_prior_rms": float(math.sqrt(np.mean(np.square(depth_residuals))))
            if depth_residuals
            else 0.0,
            "normal_prior_rms": float(math.sqrt(np.mean(np.square(normal_residuals))))
            if normal_residuals
            else 0.0,
        }

    def _residual(
        self,
        params: np.ndarray,
        problem: JointIntrinsicProblem,
        init_params: np.ndarray,
    ) -> np.ndarray:
        intrinsics = PinholeIntrinsics(
            focal_x=float(params[0]),
            focal_y=float(params[1]),
            cx=float(params[2]),
            cy=float(params[3]),
        )
        adapter = MiscalibrationPriorAdapter(assumed_intrinsics=problem.assumed_intrinsics)

        residuals: list[np.ndarray] = []
        geometry_residual = SeparableSharedIntrinsicSolver._residual(
            params,
            problem.camera_xyz,
            problem.observed_uv,
        )
        if geometry_residual.size:
            focal_scale = math.sqrt(max(float(params[0]) * float(params[1]), 1e-12))
            residuals.append(
                math.sqrt(self.geometry_weight)
                * geometry_residual
                / (focal_scale * math.sqrt(float(geometry_residual.size)))
            )
        if self.focal_prior_weight > 0.0:
            residuals.append(
                np.asarray(
                    [
                        math.sqrt(self.focal_prior_weight)
                        * (float(params[0]) - float(init_params[0]))
                        / max(float(init_params[0]), 1e-12),
                        math.sqrt(self.focal_prior_weight)
                        * (float(params[1]) - float(init_params[1]))
                        / max(float(init_params[1]), 1e-12),
                    ],
                    dtype=np.float64,
                )
            )

        for frame in problem.frame_priors:
            corrected = adapter.correct_priors(
                current_intrinsics=intrinsics,
                pointmap=frame.pointmap,
                depth=frame.depth,
                normals=frame.normals,
            )
            if frame.observed_uv is not None and frame.camera_xyz is not None:
                if self.pointmap_weight > 0.0 and frame.pointmap_weight_scale > 0.0:
                    pointmap_term = adapter.pointmap_correspondence_residual(
                        corrected.pointmap,
                        observed_uv=frame.observed_uv,
                        camera_xyz=frame.camera_xyz,
                        mask=frame.mask,
                    )
                    residuals.append(
                        np.asarray(
                            [
                                math.sqrt(self.pointmap_weight * frame.pointmap_weight_scale)
                                * float(pointmap_term.item())
                            ],
                            dtype=np.float64,
                        )
                    )
                if self.depth_weight > 0.0 and frame.depth_weight_scale > 0.0:
                    depth_pointmap = adapter.corrected_pointmap_from_depth(
                        frame.depth,
                        current_intrinsics=intrinsics,
                    )
                    depth_term = adapter.pointmap_correspondence_residual(
                        depth_pointmap,
                        observed_uv=frame.observed_uv,
                        camera_xyz=frame.camera_xyz,
                        mask=frame.mask,
                    )
                    residuals.append(
                        np.asarray(
                            [
                                math.sqrt(self.depth_weight * frame.depth_weight_scale)
                                * float(depth_term.item())
                            ],
                            dtype=np.float64,
                        )
                    )
            if self.normal_weight > 0.0 and frame.normal_weight_scale > 0.0:
                normal_term = adapter.normal_pointmap_consistency_residual(
                    corrected.normals,
                    pointmap=corrected.pointmap,
                    mask=frame.mask,
                )
                residuals.append(
                    np.asarray(
                        [
                            math.sqrt(self.normal_weight * frame.normal_weight_scale)
                            * float(normal_term.item())
                        ],
                        dtype=np.float64,
                    )
                )

        if not residuals:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(residuals, axis=0)

    @staticmethod
    def _normalized_geometry_residual(
        *,
        intrinsics: PinholeIntrinsics,
        camera_xyz: np.ndarray,
        observed_uv: np.ndarray,
    ) -> np.ndarray:
        residual = SeparableSharedIntrinsicSolver._residual(
            np.array(
                [
                    float(intrinsics.focal_x),
                    float(intrinsics.focal_y),
                    float(intrinsics.cx),
                    float(intrinsics.cy),
                ],
                dtype=np.float64,
            ),
            camera_xyz,
            observed_uv,
        )
        focal_scale = math.sqrt(
            max(float(intrinsics.focal_x) * float(intrinsics.focal_y), 1e-12)
        )
        if residual.size == 0:
            return residual
        return residual / (focal_scale * math.sqrt(float(residual.size)))
