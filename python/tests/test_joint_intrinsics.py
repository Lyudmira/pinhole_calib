import unittest

import numpy as np
import torch

from pinhole_calib import (
    JointFramePriors,
    JointIntrinsicProblem,
    JointSharedIntrinsicOptimizer,
    PinholeIntrinsics,
    SeparableSharedIntrinsicSolver,
)


def _make_true_pointmap(intrinsics: PinholeIntrinsics, *, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    depth = 3.0 + 0.3 * (u - intrinsics.cx) / intrinsics.focal_x - 0.2 * (v - intrinsics.cy) / intrinsics.focal_y
    x_cam = (u - intrinsics.cx) / intrinsics.focal_x * depth
    y_cam = (v - intrinsics.cy) / intrinsics.focal_y * depth
    pointmap = torch.stack([x_cam, y_cam, depth], dim=-1)
    return pointmap, depth


def _pointmap_normals(pointmap: torch.Tensor) -> torch.Tensor:
    dx = pointmap[1:-1, 2:, :] - pointmap[1:-1, :-2, :]
    dy = pointmap[2:, 1:-1, :] - pointmap[:-2, 1:-1, :]
    interior = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    normals = torch.zeros_like(pointmap)
    normals[1:-1, 1:-1, :] = interior
    normals[0, :, :] = normals[1, :, :]
    normals[-1, :, :] = normals[-2, :, :]
    normals[:, 0, :] = normals[:, 1, :]
    normals[:, -1, :] = normals[:, -2, :]
    return normals


def _shift_depth_like_adapter(true_depth: torch.Tensor, *, delta_cx: float, delta_cy: float) -> torch.Tensor:
    shifted = torch.zeros_like(true_depth)
    height, width = true_depth.shape
    for y in range(height):
        for x in range(width):
            src_x = int(round(x - delta_cx))
            src_y = int(round(y - delta_cy))
            if 0 <= src_x < width and 0 <= src_y < height:
                shifted[src_y, src_x] = true_depth[y, x]
    return shifted


class JointIntrinsicOptimizerTest(unittest.TestCase):
    def test_joint_optimizer_improves_over_geometry_when_priors_are_consistent(self) -> None:
        assumed = PinholeIntrinsics(focal_x=400.0, focal_y=420.0, cx=16.0, cy=16.0)
        true_intrinsics = PinholeIntrinsics(focal_x=400.0, focal_y=420.0, cx=28.0, cy=10.0)
        width = 32
        height = 32

        true_pointmap, true_depth = _make_true_pointmap(true_intrinsics, width=width, height=height)
        true_normals = _pointmap_normals(true_pointmap)
        H = true_intrinsics.mismatch_matrix(assumed)
        raw_pointmap = torch.einsum("ij,hwj->hwi", H, true_pointmap)
        raw_normals = torch.einsum(
            "ij,hwj->hwi",
            torch.linalg.inv(H.transpose(0, 1)),
            true_normals,
        )
        raw_normals = torch.nn.functional.normalize(raw_normals, dim=-1)
        raw_depth = _shift_depth_like_adapter(
            true_depth,
            delta_cx=float(true_intrinsics.cx - assumed.cx),
            delta_cy=float(true_intrinsics.cy - assumed.cy),
        )
        mask = torch.ones((height, width), dtype=torch.bool)
        mask[:2, :] = False
        mask[-2:, :] = False
        mask[:, :2] = False
        mask[:, -2:] = False

        sample_uv = []
        sample_xyz = []
        for y in range(3, height - 3, 4):
            for x in range(3, width - 3, 4):
                sample_uv.append([x + 0.5, y + 0.5])
                sample_xyz.append(true_pointmap[y, x].numpy())
        observed_uv = np.asarray(sample_uv, dtype=np.float64)
        camera_xyz = np.asarray(sample_xyz, dtype=np.float64)
        noisy_observed_uv = observed_uv + np.array([2.5, -1.5], dtype=np.float64)

        geometry_only = SeparableSharedIntrinsicSolver().fit(
            camera_xyz=camera_xyz,
            observed_uv=noisy_observed_uv,
            init_intrinsics=assumed,
        )
        problem = JointIntrinsicProblem(
            camera_xyz=camera_xyz,
            observed_uv=noisy_observed_uv,
            assumed_intrinsics=assumed,
            frame_priors=[
                JointFramePriors(
                    image_name="synthetic",
                    pointmap=raw_pointmap,
                    depth=raw_depth,
                    normals=raw_normals,
                    mask=mask,
                    observed_uv=observed_uv,
                    camera_xyz=camera_xyz,
                )
            ],
        )
        joint = JointSharedIntrinsicOptimizer(
            geometry_weight=1.0,
            pointmap_weight=8.0,
            depth_weight=4.0,
            normal_weight=0.5,
            max_nfev=80,
        ).fit(
            problem=problem,
            init_intrinsics=geometry_only.intrinsics,
        )

        geometry_error = np.linalg.norm(
            np.array(
                [
                    float(geometry_only.intrinsics.focal_x) - float(true_intrinsics.focal_x),
                    float(geometry_only.intrinsics.focal_y) - float(true_intrinsics.focal_y),
                    float(geometry_only.intrinsics.cx) - float(true_intrinsics.cx),
                    float(geometry_only.intrinsics.cy) - float(true_intrinsics.cy),
                ],
                dtype=np.float64,
            )
        )
        joint_error = np.linalg.norm(
            np.array(
                [
                    float(joint.intrinsics.focal_x) - float(true_intrinsics.focal_x),
                    float(joint.intrinsics.focal_y) - float(true_intrinsics.focal_y),
                    float(joint.intrinsics.cx) - float(true_intrinsics.cx),
                    float(joint.intrinsics.cy) - float(true_intrinsics.cy),
                ],
                dtype=np.float64,
            )
        )

        self.assertLess(joint_error, geometry_error)
        self.assertLess(joint.pointmap_prior_rms, 0.2)
        self.assertLess(joint.depth_prior_rms, 0.2)
        self.assertTrue(np.isfinite(joint.normal_prior_rms))


if __name__ == "__main__":
    unittest.main()
