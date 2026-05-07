from __future__ import annotations

from dataclasses import dataclass
import typing as t

import torch
import torch.nn.functional as F
from torch import Tensor

from ._external import ensure_from_dc_on_path
from .intrinsics import PinholeIntrinsics
from .se3 import se3_exp

ensure_from_dc_on_path()

from dc_reality.splatting.utils.pixel_grid import (  # noqa: E402
    normalize_uv_hw2_for_grid_sample,
    pixel_grid_hw2,
)


def _recover_intrinsic_with_shift(*, points: Tensor, mask: Tensor | None):
    from dc_reality.splatting.utils.moge._utils import recover_intrinsic_with_shift

    return recover_intrinsic_with_shift(points=points, mask=mask)


def _unproject_grid_points(
    *,
    depths: Tensor,
    intrinsics: Tensor,
    image_width: int,
    image_height: int,
) -> Tensor:
    from dc_reality.splatting.utils.projection import unproject_grid_points

    return unproject_grid_points(
        depths=depths,
        intrinsics=intrinsics,
        image_width=image_width,
        image_height=image_height,
    )


@dataclass(slots=True)
class CorrectedPriors:
    pose: Tensor | None = None
    depth: Tensor | None = None
    normals: Tensor | None = None
    pointmap: Tensor | None = None


@dataclass(slots=True)
class GatingSignals:
    intrinsics_error: Tensor
    geometry_residual: Tensor
    bias_alignment: Tensor


@dataclass(slots=True)
class GatingConfig:
    tau: float = 1.0
    intrinsics_weight: float = 1.0
    geometry_weight: float = 1.0
    bias_weight: float = 1.0


class MiscalibrationPriorAdapter:
    def __init__(
        self,
        *,
        assumed_intrinsics: PinholeIntrinsics,
        align_corners: bool = False,
        padding_mode: str = "border",
    ) -> None:
        self.assumed_intrinsics = assumed_intrinsics
        self.align_corners = align_corners
        self.padding_mode = padding_mode

    def correction_state(self, current_intrinsics: PinholeIntrinsics) -> dict[str, Tensor]:
        mismatch = current_intrinsics.mismatch_matrix(self.assumed_intrinsics)
        return {
            "H": mismatch,
            "H_inv": torch.linalg.inv(mismatch),
            "beta": current_intrinsics.normalized_principal_point_shift(
                self.assumed_intrinsics
            ),
            "delta_pixels": current_intrinsics.delta_pixels(self.assumed_intrinsics),
            "delta_theta": current_intrinsics.delta_theta(self.assumed_intrinsics),
        }

    def pose_delta_near_axis(self, current_intrinsics: PinholeIntrinsics) -> Tensor:
        beta_x, beta_y = self.correction_state(current_intrinsics)["beta"]
        xi = torch.zeros(6, device=beta_x.device, dtype=beta_x.dtype)
        xi[3:] = torch.stack(
            [
                -beta_y,
                beta_x,
                torch.zeros_like(beta_x),
            ]
        )
        return xi

    def pose_delta_from_jacobian(
        self,
        *,
        current_intrinsics: PinholeIntrinsics,
        response_matrix: Tensor,
    ) -> Tensor:
        delta_theta = self.correction_state(current_intrinsics)["delta_theta"]
        assert response_matrix.shape[-2:] == (6, 4), response_matrix.shape
        return torch.einsum("...ab,b->...a", response_matrix, delta_theta)

    def correct_pose(
        self,
        pose: Tensor,
        *,
        current_intrinsics: PinholeIntrinsics,
        pose_type: t.Literal["world_to_camera", "camera_to_world"] = "world_to_camera",
        response_matrix: Tensor | None = None,
    ) -> Tensor:
        assert pose.shape[-2:] == (4, 4), pose.shape
        xi = (
            self.pose_delta_from_jacobian(
                current_intrinsics=current_intrinsics,
                response_matrix=response_matrix,
            )
            if response_matrix is not None
            else self.pose_delta_near_axis(current_intrinsics)
        )
        bias = se3_exp(xi)
        if pose_type == "world_to_camera":
            return torch.linalg.inv(bias) @ pose
        if pose_type == "camera_to_world":
            return pose @ bias
        raise ValueError(f"Unknown pose_type: {pose_type}")

    def correct_depth(
        self,
        depth: Tensor,
        *,
        current_intrinsics: PinholeIntrinsics,
        mode: str = "bilinear",
    ) -> Tensor:
        depth_nchw, squeezed = self._to_nchw(depth)
        _, _, height, width = depth_nchw.shape
        delta_pixels = self.correction_state(current_intrinsics)["delta_pixels"]
        base_uv = pixel_grid_hw2(
            width=width,
            height=height,
            device=depth_nchw.device,
            dtype=depth_nchw.dtype,
            pixel_offset=0.5,
        )
        source_uv = base_uv - delta_pixels.view(1, 1, 2)
        grid = normalize_uv_hw2_for_grid_sample(
            source_uv,
            width=width,
            height=height,
            align_corners=self.align_corners,
        ).unsqueeze(0).expand(depth_nchw.shape[0], -1, -1, -1)
        corrected = F.grid_sample(
            depth_nchw,
            grid,
            mode=mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return self._restore_depth_shape(corrected, squeezed)

    def correct_normals(
        self,
        normals: Tensor,
        *,
        current_intrinsics: PinholeIntrinsics,
    ) -> Tensor:
        assert normals.shape[-1] == 3, normals.shape
        H = self.correction_state(current_intrinsics)["H"]
        corrected = torch.einsum("ij,...j->...i", H.transpose(0, 1), normals)
        return F.normalize(corrected, dim=-1)

    def correct_pointmap(
        self,
        pointmap: Tensor,
        *,
        current_intrinsics: PinholeIntrinsics,
    ) -> Tensor:
        assert pointmap.shape[-1] == 3, pointmap.shape
        H_inv = self.correction_state(current_intrinsics)["H_inv"]
        return torch.einsum("ij,...j->...i", H_inv, pointmap)

    def corrected_pointmap_from_depth(
        self,
        depth: Tensor,
        *,
        current_intrinsics: PinholeIntrinsics,
    ) -> Tensor:
        corrected_depth = self.correct_depth(depth, current_intrinsics=current_intrinsics)
        depth_hw1 = self._to_hw1(corrected_depth)
        return _unproject_grid_points(
            depths=depth_hw1,
            intrinsics=current_intrinsics.matrix(
                device=depth_hw1.device, dtype=depth_hw1.dtype
            ),
            image_width=depth_hw1.shape[1],
            image_height=depth_hw1.shape[0],
        )

    def pointmap_intrinsics_consistency(
        self,
        pointmap: Tensor,
        *,
        mask: Tensor | None = None,
        target_intrinsics: PinholeIntrinsics | None = None,
    ) -> dict[str, t.Any]:
        if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
            raise ValueError(f"Expected pointmap [H, W, 3], got {pointmap.shape}")
        _, result = _recover_intrinsic_with_shift(points=pointmap, mask=mask)
        if target_intrinsics is None:
            return result

        target = target_intrinsics.matrix(
            device=pointmap.device, dtype=pointmap.dtype
        )
        estimated = torch.tensor(
            [
                result["focal_x"],
                result["focal_y"],
                result["cx"],
                result["cy"],
            ],
            device=pointmap.device,
            dtype=pointmap.dtype,
        )
        target_vec = torch.tensor(
            [
                target[0, 0],
                target[1, 1],
                target[0, 2],
                target[1, 2],
            ],
            device=pointmap.device,
            dtype=pointmap.dtype,
        )
        result["target_l2"] = torch.linalg.norm(estimated - target_vec)
        return result

    def gate(
        self,
        *,
        signals: GatingSignals,
        config: GatingConfig | None = None,
    ) -> Tensor:
        config = config or GatingConfig()
        logits = (
            config.tau
            - config.intrinsics_weight * signals.intrinsics_error
            - config.geometry_weight * signals.geometry_residual
            - config.bias_weight * signals.bias_alignment
        )
        return torch.sigmoid(logits)

    def build_default_signals(
        self,
        *,
        current_intrinsics: PinholeIntrinsics,
        geometry_residual: Tensor | float = 0.0,
        bias_alignment: Tensor | float = 0.0,
    ) -> GatingSignals:
        delta_theta = self.correction_state(current_intrinsics)["delta_theta"]
        intrinsics_error = torch.linalg.norm(delta_theta)
        return GatingSignals(
            intrinsics_error=intrinsics_error,
            geometry_residual=torch.as_tensor(
                geometry_residual,
                device=intrinsics_error.device,
                dtype=intrinsics_error.dtype,
            ),
            bias_alignment=torch.as_tensor(
                bias_alignment,
                device=intrinsics_error.device,
                dtype=intrinsics_error.dtype,
            ),
        )

    def correct_priors(
        self,
        *,
        current_intrinsics: PinholeIntrinsics,
        pose: Tensor | None = None,
        pose_type: t.Literal["world_to_camera", "camera_to_world"] = "world_to_camera",
        pose_response_matrix: Tensor | None = None,
        depth: Tensor | None = None,
        normals: Tensor | None = None,
        pointmap: Tensor | None = None,
    ) -> CorrectedPriors:
        corrected = CorrectedPriors()
        if pose is not None:
            corrected.pose = self.correct_pose(
                pose,
                current_intrinsics=current_intrinsics,
                pose_type=pose_type,
                response_matrix=pose_response_matrix,
            )
        if depth is not None:
            corrected.depth = self.correct_depth(
                depth,
                current_intrinsics=current_intrinsics,
            )
        if normals is not None:
            corrected.normals = self.correct_normals(
                normals,
                current_intrinsics=current_intrinsics,
            )
        if pointmap is not None:
            corrected.pointmap = self.correct_pointmap(
                pointmap,
                current_intrinsics=current_intrinsics,
            )
        return corrected

    @staticmethod
    def _to_nchw(depth: Tensor) -> tuple[Tensor, bool]:
        if depth.ndim == 2:
            return depth.unsqueeze(0).unsqueeze(0), True
        if depth.ndim == 3:
            if depth.shape[0] == 1:
                return depth.unsqueeze(0), True
            if depth.shape[-1] == 1:
                return depth.permute(2, 0, 1).unsqueeze(0), True
        if depth.ndim == 4:
            return depth, False
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    @staticmethod
    def _restore_depth_shape(depth_nchw: Tensor, squeezed: bool) -> Tensor:
        if not squeezed:
            return depth_nchw
        if depth_nchw.shape[0] == 1 and depth_nchw.shape[1] == 1:
            return depth_nchw.squeeze(0).squeeze(0)
        return depth_nchw

    @staticmethod
    def _to_hw1(depth: Tensor) -> Tensor:
        if depth.ndim == 2:
            return depth.unsqueeze(-1)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            return depth
        if depth.ndim == 3 and depth.shape[0] == 1:
            return depth.squeeze(0).unsqueeze(-1)
        raise ValueError(f"Unsupported depth shape for pointmap conversion: {depth.shape}")
