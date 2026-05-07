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
    normalize_uv_n2_for_grid_sample,
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
class DepthAffineCorrection:
    scale: float
    bias: float


@dataclass(slots=True)
class GatingSignals:
    intrinsics_error: Tensor
    geometry_residual: Tensor


@dataclass(slots=True)
class ChannelGatingSignals:
    intrinsics_error: Tensor
    pointmap_geometry_residual: Tensor
    pose_geometry_residual: Tensor
    depth_geometry_residual: Tensor
    normal_geometry_residual: Tensor


@dataclass(slots=True)
class ChannelGates:
    pointmap: Tensor
    pose: Tensor
    depth: Tensor
    normals: Tensor


@dataclass(slots=True)
class PriorStateConfig:
    corrected_intrinsics_error_max: float = 0.04
    corrected_principal_point_std_px_max: float = 2.0
    filtered_intrinsics_error_max: float = 0.10
    filtered_principal_point_std_px_max: float = 10.0
    filtered_schur_condition_max: float = 1e14


@dataclass(slots=True)
class PriorStateAssessment:
    state: t.Literal["corrected", "debiased", "filtered"]
    intrinsics_error: float
    principal_point_std_px: float
    principal_point_rotation_sensitivity: float
    schur_condition_number: float
    rerun_recommended: bool


@dataclass(init=False, slots=True)
class GatingConfig:
    tau: float
    eta_k: float
    eta_r: float

    def __init__(
        self,
        tau: float = 1.0,
        eta_k: float = 1.0,
        eta_r: float = 1.0,
        *,
        intrinsics_weight: float | None = None,
        geometry_weight: float | None = None,
    ) -> None:
        if intrinsics_weight is not None:
            eta_k = intrinsics_weight
        if geometry_weight is not None:
            eta_r = geometry_weight
        self.tau = float(tau)
        self.eta_k = float(eta_k)
        self.eta_r = float(eta_r)

    @property
    def intrinsics_weight(self) -> float:
        return self.eta_k

    @property
    def geometry_weight(self) -> float:
        return self.eta_r


@dataclass(init=False, slots=True)
class ChannelGatingConfig:
    tau_pointmap: float
    tau_pose: float
    tau_depth: float
    tau_normal: float
    eta_k: float
    eta_pointmap_r: float
    eta_pose_r: float
    eta_depth_r: float
    eta_normal_r: float

    def __init__(
        self,
        tau_pointmap: float = 1.0,
        tau_pose: float = 1.0,
        tau_depth: float = 1.0,
        tau_normal: float = 1.0,
        eta_k: float = 1.0,
        eta_pointmap_r: float = 1.0,
        eta_pose_r: float = 1.0,
        eta_depth_r: float = 1.0,
        eta_normal_r: float = 1.0,
    ) -> None:
        self.tau_pointmap = float(tau_pointmap)
        self.tau_pose = float(tau_pose)
        self.tau_depth = float(tau_depth)
        self.tau_normal = float(tau_normal)
        self.eta_k = float(eta_k)
        self.eta_pointmap_r = float(eta_pointmap_r)
        self.eta_pose_r = float(eta_pose_r)
        self.eta_depth_r = float(eta_depth_r)
        self.eta_normal_r = float(eta_normal_r)


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
        affine_correction: DepthAffineCorrection | None = None,
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
        if affine_correction is not None:
            corrected = corrected * float(affine_correction.scale) + float(affine_correction.bias)
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
        affine_correction: DepthAffineCorrection | None = None,
    ) -> Tensor:
        corrected_depth = self.correct_depth(
            depth,
            current_intrinsics=current_intrinsics,
            affine_correction=affine_correction,
        )
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
        estimated_intrinsics = PinholeIntrinsics(
            focal_x=float(result["focal_x"]),
            focal_y=float(result["focal_y"]),
            cx=float(result["cx"]),
            cy=float(result["cy"]),
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
        result["target_l2"] = float(torch.linalg.norm(estimated - target_vec).item())
        result["target_delta_theta_unitless_l2"] = float(
            torch.linalg.norm(
                estimated_intrinsics.delta_theta_unitless(target_intrinsics)
            ).item()
        )
        return result

    def pointmap_geometry_residual(
        self,
        pointmap: Tensor,
        *,
        target_intrinsics: PinholeIntrinsics,
        mask: Tensor | None = None,
    ) -> Tensor:
        consistency = self.pointmap_intrinsics_consistency(
            pointmap,
            mask=mask,
            target_intrinsics=target_intrinsics,
        )
        return torch.as_tensor(
            consistency["target_delta_theta_unitless_l2"],
            device=pointmap.device,
            dtype=pointmap.dtype,
        )

    def pointmap_correspondence_residual(
        self,
        pointmap: Tensor,
        *,
        observed_uv: Tensor | t.Any,
        camera_xyz: Tensor | t.Any,
        mask: Tensor | None = None,
    ) -> Tensor:
        if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
            raise ValueError(f"Expected pointmap [H, W, 3], got {pointmap.shape}")

        observed_uv_t = torch.as_tensor(
            observed_uv,
            device=pointmap.device,
            dtype=pointmap.dtype,
        )
        camera_xyz_t = torch.as_tensor(
            camera_xyz,
            device=pointmap.device,
            dtype=pointmap.dtype,
        )
        if observed_uv_t.ndim != 2 or observed_uv_t.shape[-1] != 2:
            raise ValueError(f"Expected observed_uv [N, 2], got {observed_uv_t.shape}")
        if camera_xyz_t.ndim != 2 or camera_xyz_t.shape[-1] != 3:
            raise ValueError(f"Expected camera_xyz [N, 3], got {camera_xyz_t.shape}")
        if observed_uv_t.shape[0] != camera_xyz_t.shape[0]:
            raise ValueError("observed_uv and camera_xyz must have the same number of rows")
        if observed_uv_t.shape[0] == 0:
            return torch.full((), float("inf"), device=pointmap.device, dtype=pointmap.dtype)

        height, width = pointmap.shape[:2]
        valid = torch.isfinite(observed_uv_t).all(dim=-1) & torch.isfinite(camera_xyz_t).all(dim=-1)
        valid = valid & (camera_xyz_t[:, 2] > 1e-6)
        valid = valid & (observed_uv_t[:, 0] >= 0.0) & (observed_uv_t[:, 0] < width)
        valid = valid & (observed_uv_t[:, 1] >= 0.0) & (observed_uv_t[:, 1] < height)
        if not torch.any(valid):
            return torch.full((), float("inf"), device=pointmap.device, dtype=pointmap.dtype)

        observed_uv_t = observed_uv_t[valid]
        camera_xyz_t = camera_xyz_t[valid]

        sampled_points = self._sample_hwc_at_uv(pointmap, observed_uv_t, mode="bilinear")
        valid = torch.isfinite(sampled_points).all(dim=-1)

        if mask is not None:
            sampled_mask = self._sample_mask_at_uv(mask, observed_uv_t)
            valid = valid & sampled_mask

        if not torch.any(valid):
            return torch.full((), float("inf"), device=pointmap.device, dtype=pointmap.dtype)

        sampled_points = sampled_points[valid]
        camera_xyz_t = camera_xyz_t[valid]
        depth_scale = camera_xyz_t[:, 2].abs().median().clamp_min(1e-6)
        diff = (sampled_points - camera_xyz_t) / depth_scale
        return torch.sqrt(torch.mean(torch.sum(diff * diff, dim=-1)))

    def pointmap_multiview_correspondence_residual(
        self,
        pointmap: Tensor,
        *,
        source_uv: Tensor | t.Any,
        target_uv: Tensor | t.Any,
        source_rotation: Tensor | t.Any,
        source_translation: Tensor | t.Any,
        target_rotations: Tensor | t.Any,
        target_translations: Tensor | t.Any,
        target_intrinsics: PinholeIntrinsics,
        mask: Tensor | None = None,
    ) -> Tensor:
        if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
            raise ValueError(f"Expected pointmap [H, W, 3], got {pointmap.shape}")

        source_uv_t = torch.as_tensor(source_uv, device=pointmap.device, dtype=pointmap.dtype)
        target_uv_t = torch.as_tensor(target_uv, device=pointmap.device, dtype=pointmap.dtype)
        source_rotation_t = torch.as_tensor(
            source_rotation, device=pointmap.device, dtype=pointmap.dtype
        )
        source_translation_t = torch.as_tensor(
            source_translation, device=pointmap.device, dtype=pointmap.dtype
        )
        target_rotations_t = torch.as_tensor(
            target_rotations, device=pointmap.device, dtype=pointmap.dtype
        )
        target_translations_t = torch.as_tensor(
            target_translations, device=pointmap.device, dtype=pointmap.dtype
        )

        if source_uv_t.ndim != 2 or source_uv_t.shape[-1] != 2:
            raise ValueError(f"Expected source_uv [N, 2], got {source_uv_t.shape}")
        if target_uv_t.ndim != 2 or target_uv_t.shape[-1] != 2:
            raise ValueError(f"Expected target_uv [N, 2], got {target_uv_t.shape}")
        if source_uv_t.shape[0] != target_uv_t.shape[0]:
            raise ValueError("source_uv and target_uv must have the same number of rows")
        if target_rotations_t.ndim != 3 or target_rotations_t.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected target_rotations [N, 3, 3], got {target_rotations_t.shape}"
            )
        if target_translations_t.ndim != 2 or target_translations_t.shape[-1] != 3:
            raise ValueError(
                f"Expected target_translations [N, 3], got {target_translations_t.shape}"
            )
        if target_rotations_t.shape[0] != source_uv_t.shape[0]:
            raise ValueError("target_rotations and source_uv must have the same number of rows")
        if target_translations_t.shape[0] != source_uv_t.shape[0]:
            raise ValueError("target_translations and source_uv must have the same number of rows")
        if source_rotation_t.shape != (3, 3):
            raise ValueError(f"Expected source_rotation [3, 3], got {source_rotation_t.shape}")
        if source_translation_t.shape != (3,):
            raise ValueError(f"Expected source_translation [3], got {source_translation_t.shape}")
        if source_uv_t.shape[0] == 0:
            return torch.full((), float("inf"), device=pointmap.device, dtype=pointmap.dtype)

        height, width = pointmap.shape[:2]
        valid = torch.isfinite(source_uv_t).all(dim=-1) & torch.isfinite(target_uv_t).all(dim=-1)
        valid = valid & torch.isfinite(target_rotations_t).all(dim=(-1, -2))
        valid = valid & torch.isfinite(target_translations_t).all(dim=-1)
        valid = valid & (source_uv_t[:, 0] >= 0.0) & (source_uv_t[:, 0] < width)
        valid = valid & (source_uv_t[:, 1] >= 0.0) & (source_uv_t[:, 1] < height)
        if not torch.any(valid):
            return torch.full((), float("inf"), device=pointmap.device, dtype=pointmap.dtype)

        source_uv_t = source_uv_t[valid]
        target_uv_t = target_uv_t[valid]
        target_rotations_t = target_rotations_t[valid]
        target_translations_t = target_translations_t[valid]

        sampled_points = self._sample_hwc_at_uv(pointmap, source_uv_t, mode="bilinear")
        valid = torch.isfinite(sampled_points).all(dim=-1) & (sampled_points[:, 2] > 1e-6)
        if mask is not None:
            valid = valid & self._sample_mask_at_uv(mask, source_uv_t)
        if not torch.any(valid):
            return torch.full((), float("inf"), device=pointmap.device, dtype=pointmap.dtype)

        sampled_points = sampled_points[valid]
        target_uv_t = target_uv_t[valid]
        target_rotations_t = target_rotations_t[valid]
        target_translations_t = target_translations_t[valid]

        world_points = torch.einsum(
            "ij,nj->ni", source_rotation_t.transpose(0, 1), sampled_points - source_translation_t
        )
        target_camera_points = torch.einsum("nij,nj->ni", target_rotations_t, world_points)
        target_camera_points = target_camera_points + target_translations_t
        valid = torch.isfinite(target_camera_points).all(dim=-1) & (target_camera_points[:, 2] > 1e-6)
        if not torch.any(valid):
            return torch.full((), float("inf"), device=pointmap.device, dtype=pointmap.dtype)

        target_camera_points = target_camera_points[valid]
        target_uv_t = target_uv_t[valid]
        fx = torch.as_tensor(
            target_intrinsics.focal_x, device=pointmap.device, dtype=pointmap.dtype
        )
        fy = torch.as_tensor(
            target_intrinsics.focal_y, device=pointmap.device, dtype=pointmap.dtype
        )
        cx = torch.as_tensor(target_intrinsics.cx, device=pointmap.device, dtype=pointmap.dtype)
        cy = torch.as_tensor(target_intrinsics.cy, device=pointmap.device, dtype=pointmap.dtype)
        pred_u = fx * target_camera_points[:, 0] / target_camera_points[:, 2] + cx
        pred_v = fy * target_camera_points[:, 1] / target_camera_points[:, 2] + cy
        predicted_uv = torch.stack([pred_u, pred_v], dim=-1)
        focal_scale = torch.sqrt(fx * fy).clamp_min(1e-6)
        diff = (predicted_uv - target_uv_t) / focal_scale
        return torch.sqrt(torch.mean(torch.sum(diff * diff, dim=-1)))

    def gate(
        self,
        *,
        signals: GatingSignals,
        config: GatingConfig | None = None,
    ) -> Tensor:
        config = config or GatingConfig()
        logits = (
            config.tau
            - config.eta_k * signals.intrinsics_error
            - config.eta_r * signals.geometry_residual
        )
        return torch.sigmoid(logits)

    def build_channel_signals(
        self,
        *,
        current_intrinsics: PinholeIntrinsics,
        pointmap_geometry_residual: Tensor | float = 0.0,
        pose_geometry_residual: Tensor | float = 0.0,
        depth_geometry_residual: Tensor | float = 0.0,
        normal_geometry_residual: Tensor | float = 0.0,
    ) -> ChannelGatingSignals:
        intrinsics_error = torch.linalg.norm(
            current_intrinsics.delta_theta_unitless(self.assumed_intrinsics)
        )
        return ChannelGatingSignals(
            intrinsics_error=intrinsics_error,
            pointmap_geometry_residual=torch.as_tensor(
                pointmap_geometry_residual,
                device=intrinsics_error.device,
                dtype=intrinsics_error.dtype,
            ),
            pose_geometry_residual=torch.as_tensor(
                pose_geometry_residual,
                device=intrinsics_error.device,
                dtype=intrinsics_error.dtype,
            ),
            depth_geometry_residual=torch.as_tensor(
                depth_geometry_residual,
                device=intrinsics_error.device,
                dtype=intrinsics_error.dtype,
            ),
            normal_geometry_residual=torch.as_tensor(
                normal_geometry_residual,
                device=intrinsics_error.device,
                dtype=intrinsics_error.dtype,
            ),
        )

    def gate_channels(
        self,
        *,
        signals: ChannelGatingSignals,
        config: ChannelGatingConfig | None = None,
    ) -> ChannelGates:
        config = config or ChannelGatingConfig()
        intrinsics_term = config.eta_k * signals.intrinsics_error
        return ChannelGates(
            pointmap=torch.sigmoid(
                config.tau_pointmap
                - intrinsics_term
                - config.eta_pointmap_r * signals.pointmap_geometry_residual
            ),
            pose=torch.sigmoid(config.tau_pose - intrinsics_term - config.eta_pose_r * signals.pose_geometry_residual),
            depth=torch.sigmoid(config.tau_depth - intrinsics_term - config.eta_depth_r * signals.depth_geometry_residual),
            normals=torch.sigmoid(config.tau_normal - intrinsics_term - config.eta_normal_r * signals.normal_geometry_residual),
        )

    def normal_pointmap_consistency_residual(
        self,
        normals: Tensor,
        *,
        pointmap: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if normals.ndim != 3 or normals.shape[-1] != 3:
            raise ValueError(f"Expected normals [H, W, 3], got {normals.shape}")
        if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
            raise ValueError(f"Expected pointmap [H, W, 3], got {pointmap.shape}")
        if normals.shape[:2] != pointmap.shape[:2]:
            raise ValueError("normals and pointmap must share image dimensions")
        if pointmap.shape[0] < 3 or pointmap.shape[1] < 3:
            return torch.full((), float("inf"), device=pointmap.device, dtype=pointmap.dtype)

        dx = pointmap[1:-1, 2:, :] - pointmap[1:-1, :-2, :]
        dy = pointmap[2:, 1:-1, :] - pointmap[:-2, 1:-1, :]
        predicted_normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        observed_normals = F.normalize(normals[1:-1, 1:-1, :], dim=-1)
        valid = torch.isfinite(predicted_normals).all(dim=-1) & torch.isfinite(observed_normals).all(dim=-1)
        valid = valid & (torch.linalg.norm(dx, dim=-1) > 1e-6) & (torch.linalg.norm(dy, dim=-1) > 1e-6)
        if mask is not None:
            valid = valid & mask[1:-1, 1:-1].bool()
        if not torch.any(valid):
            return torch.full((), float("inf"), device=pointmap.device, dtype=pointmap.dtype)

        cos = (predicted_normals[valid] * observed_normals[valid]).sum(dim=-1).abs().clamp(0.0, 1.0)
        return torch.mean(torch.acos(cos))

    def build_default_signals(
        self,
        *,
        current_intrinsics: PinholeIntrinsics,
        geometry_residual: Tensor | float = 0.0,
    ) -> GatingSignals:
        delta_theta = current_intrinsics.delta_theta_unitless(self.assumed_intrinsics)
        intrinsics_error = torch.linalg.norm(delta_theta)
        return GatingSignals(
            intrinsics_error=intrinsics_error,
            geometry_residual=torch.as_tensor(
                geometry_residual,
                device=intrinsics_error.device,
                dtype=intrinsics_error.dtype,
            ),
        )

    def classify_prior_state(
        self,
        *,
        current_intrinsics: PinholeIntrinsics,
        principal_point_std_px: float | None = None,
        principal_point_rotation_sensitivity: float | None = None,
        schur_condition_number: float | None = None,
        rerun_available: bool = False,
        config: PriorStateConfig | None = None,
    ) -> PriorStateAssessment:
        config = config or PriorStateConfig()
        intrinsics_error = float(
            torch.linalg.norm(
                current_intrinsics.delta_theta_unitless(self.assumed_intrinsics)
            ).item()
        )
        principal_point_std_px = float("inf") if principal_point_std_px is None else float(principal_point_std_px)
        principal_point_rotation_sensitivity = (
            float("inf")
            if principal_point_rotation_sensitivity is None
            else float(principal_point_rotation_sensitivity)
        )
        schur_condition_number = (
            float("inf") if schur_condition_number is None else float(schur_condition_number)
        )

        if (
            intrinsics_error >= config.filtered_intrinsics_error_max
            or principal_point_std_px >= config.filtered_principal_point_std_px_max
            or schur_condition_number >= config.filtered_schur_condition_max
        ):
            state: t.Literal["corrected", "debiased", "filtered"] = "filtered"
        elif (
            rerun_available
            and intrinsics_error <= config.corrected_intrinsics_error_max
            and principal_point_std_px <= config.corrected_principal_point_std_px_max
        ):
            state = "corrected"
        else:
            state = "debiased"

        return PriorStateAssessment(
            state=state,
            intrinsics_error=intrinsics_error,
            principal_point_std_px=principal_point_std_px,
            principal_point_rotation_sensitivity=principal_point_rotation_sensitivity,
            schur_condition_number=schur_condition_number,
            rerun_recommended=(state == "corrected" and rerun_available),
        )

    def correct_priors(
        self,
        *,
        current_intrinsics: PinholeIntrinsics,
        pose: Tensor | None = None,
        pose_type: t.Literal["world_to_camera", "camera_to_world"] = "world_to_camera",
        pose_response_matrix: Tensor | None = None,
        depth: Tensor | None = None,
        depth_affine_correction: DepthAffineCorrection | None = None,
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
                affine_correction=depth_affine_correction,
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

    def fit_depth_affine(
        self,
        *,
        source_depth: Tensor,
        target_depth: Tensor,
        mask: Tensor | None = None,
    ) -> DepthAffineCorrection:
        source = self._to_hw1(source_depth).squeeze(-1)
        target = self._to_hw1(target_depth).squeeze(-1)
        valid = torch.isfinite(source) & torch.isfinite(target)
        if mask is not None:
            valid = valid & mask.bool()
        if not torch.any(valid):
            return DepthAffineCorrection(scale=1.0, bias=0.0)
        x = source[valid].reshape(-1)
        y = target[valid].reshape(-1)
        if x.numel() == 1:
            return DepthAffineCorrection(scale=1.0, bias=float((y - x).item()))
        ones = torch.ones_like(x)
        design = torch.stack([x, ones], dim=-1)
        params = torch.linalg.lstsq(design, y.unsqueeze(-1)).solution.squeeze(-1)
        return DepthAffineCorrection(scale=float(params[0].item()), bias=float(params[1].item()))

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

    def _sample_hwc_at_uv(self, image: Tensor, uv: Tensor, *, mode: str) -> Tensor:
        if image.ndim != 3:
            raise ValueError(f"Expected image [H, W, C], got {image.shape}")
        height, width, _ = image.shape
        grid = normalize_uv_n2_for_grid_sample(
            uv,
            width=width,
            height=height,
            align_corners=self.align_corners,
        ).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            image.permute(2, 0, 1).unsqueeze(0),
            grid,
            mode=mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return sampled.squeeze(0).squeeze(-1).transpose(0, 1)

    def _sample_mask_at_uv(self, mask: Tensor, uv: Tensor) -> Tensor:
        if mask.ndim != 2:
            raise ValueError(f"Expected mask [H, W], got {mask.shape}")
        sampled = self._sample_hwc_at_uv(mask.unsqueeze(-1).to(dtype=uv.dtype), uv, mode="nearest")
        return sampled.squeeze(-1) > 0.5
