import unittest

import torch

from pinhole_calib import (
    ChannelGatingConfig,
    GatingConfig,
    MiscalibrationPriorAdapter,
    PinholeIntrinsics,
    PriorStateConfig,
)
from pinhole_calib.se3 import se3_exp


class PriorAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assumed = PinholeIntrinsics(400.0, 420.0, 320.0, 240.0)
        self.current = PinholeIntrinsics(400.0, 420.0, 332.0, 234.0)
        self.adapter = MiscalibrationPriorAdapter(assumed_intrinsics=self.assumed)

    def test_correction_state_matches_expected_beta_and_mismatch(self) -> None:
        state = self.adapter.correction_state(self.current)
        expected_beta = torch.tensor([12.0 / 400.0, -6.0 / 420.0])
        self.assertTrue(torch.allclose(state["beta"], expected_beta))
        self.assertTrue(torch.allclose(state["H"], self.current.mismatch_matrix(self.assumed)))

    def test_pointmap_correction_inverts_intrinsic_shear(self) -> None:
        true_points = torch.tensor(
            [
                [[0.2, -0.1, 3.0], [0.3, 0.2, 4.0]],
                [[-0.4, 0.1, 2.0], [0.0, 0.0, 5.0]],
            ],
            dtype=torch.float32,
        )
        H = self.current.mismatch_matrix(self.assumed)
        wrong_points = torch.einsum("ij,hwj->hwi", H, true_points)
        corrected = self.adapter.correct_pointmap(
            wrong_points,
            current_intrinsics=self.current,
        )
        self.assertTrue(torch.allclose(corrected, true_points, atol=1e-5))

    def test_normal_correction_inverts_plane_normal_shear(self) -> None:
        true_normal = torch.tensor([0.6, 0.0, 0.8], dtype=torch.float32)
        H = self.current.mismatch_matrix(self.assumed)
        wrong_normal = torch.linalg.solve(H.transpose(0, 1), true_normal)
        wrong_normal = torch.nn.functional.normalize(wrong_normal, dim=0)
        corrected = self.adapter.correct_normals(
            wrong_normal.view(1, 1, 3),
            current_intrinsics=self.current,
        ).view(3)
        self.assertTrue(
            torch.allclose(
                corrected,
                torch.nn.functional.normalize(true_normal, dim=0),
                atol=1e-5,
            )
        )

    def test_depth_correction_shifts_back_to_target_pixel_grid(self) -> None:
        assumed = PinholeIntrinsics(400.0, 400.0, 2.0, 2.0)
        current = PinholeIntrinsics(400.0, 400.0, 3.0, 1.0)
        adapter = MiscalibrationPriorAdapter(assumed_intrinsics=assumed)

        true_depth = torch.arange(25, dtype=torch.float32).view(5, 5)
        wrong_depth = torch.zeros_like(true_depth)
        delta = current.delta_pixels(assumed)
        for y in range(true_depth.shape[0]):
            for x in range(true_depth.shape[1]):
                src_x = int(x - delta[0].item())
                src_y = int(y - delta[1].item())
                if 0 <= src_x < true_depth.shape[1] and 0 <= src_y < true_depth.shape[0]:
                    wrong_depth[src_y, src_x] = true_depth[y, x]
        corrected = adapter.correct_depth(wrong_depth, current_intrinsics=current, mode="nearest")
        self.assertTrue(torch.allclose(corrected[:-1, 1:], true_depth[:-1, 1:]))

    def test_pose_correction_is_inverse_for_world_to_camera_and_camera_to_world(self) -> None:
        pose_w2c = torch.eye(4)
        xi = self.adapter.pose_delta_near_axis(self.current)
        bias = se3_exp(xi)
        biased_w2c = bias @ pose_w2c
        corrected_w2c = self.adapter.correct_pose(
            biased_w2c,
            current_intrinsics=self.current,
            pose_type="world_to_camera",
        )
        self.assertTrue(torch.allclose(corrected_w2c, pose_w2c, atol=1e-5))

        pose_c2w = torch.eye(4)
        biased_c2w = pose_c2w @ torch.linalg.inv(bias)
        corrected_c2w = self.adapter.correct_pose(
            biased_c2w,
            current_intrinsics=self.current,
            pose_type="camera_to_world",
        )
        self.assertTrue(torch.allclose(corrected_c2w, pose_c2w, atol=1e-5))

    def test_gating_decreases_with_larger_error(self) -> None:
        small = self.adapter.build_default_signals(
            current_intrinsics=self.current,
            geometry_residual=0.1,
        )
        large = self.adapter.build_default_signals(
            current_intrinsics=PinholeIntrinsics(400.0, 420.0, 360.0, 210.0),
            geometry_residual=1.0,
        )
        config = GatingConfig(tau=2.0)
        self.assertGreater(
            self.adapter.gate(signals=small, config=config).item(),
            self.adapter.gate(signals=large, config=config).item(),
        )

    def test_default_gating_uses_unitless_intrinsics_error(self) -> None:
        signals = self.adapter.build_default_signals(current_intrinsics=self.current)
        expected = torch.linalg.norm(
            torch.tensor([0.0, 0.0, 12.0 / 400.0, -6.0 / 420.0], dtype=torch.float32)
        )
        self.assertAlmostEqual(signals.intrinsics_error.item(), expected.item(), places=6)

    def test_gate_matches_paper_deterministic_formula(self) -> None:
        signals = self.adapter.build_default_signals(
            current_intrinsics=self.current,
            geometry_residual=0.4,
        )
        config = GatingConfig(tau=1.0, eta_k=2.0, eta_r=0.5)
        expected = torch.sigmoid(
            torch.tensor(1.0)
            - 2.0 * signals.intrinsics_error
            - 0.5 * signals.geometry_residual
        )
        self.assertAlmostEqual(
            self.adapter.gate(signals=signals, config=config).item(),
            expected.item(),
            places=6,
        )

    def test_gating_config_accepts_legacy_weight_names(self) -> None:
        config = GatingConfig(tau=1.0, intrinsics_weight=2.0, geometry_weight=0.5)
        self.assertEqual(config.eta_k, 2.0)
        self.assertEqual(config.eta_r, 0.5)

    def test_pointmap_correspondence_residual_uses_track_geometry(self) -> None:
        pointmap = torch.tensor(
            [
                [[1.0, 0.0, 4.0], [2.0, 0.0, 4.0]],
                [[1.0, 1.0, 4.0], [2.0, 1.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        observed_uv = torch.tensor([[0.5, 0.5], [1.5, 1.5]], dtype=torch.float32)
        camera_xyz = torch.tensor([[1.0, 0.0, 4.0], [2.0, 1.0, 4.0]], dtype=torch.float32)
        residual = self.adapter.pointmap_correspondence_residual(
            pointmap,
            observed_uv=observed_uv,
            camera_xyz=camera_xyz,
        )
        self.assertLess(residual.item(), 1e-6)

        wrong_camera_xyz = camera_xyz.clone()
        wrong_camera_xyz[1, 0] += 1.0
        wrong_residual = self.adapter.pointmap_correspondence_residual(
            pointmap,
            observed_uv=observed_uv,
            camera_xyz=wrong_camera_xyz,
        )
        self.assertGreater(wrong_residual.item(), 0.1)

    def test_pointmap_multiview_correspondence_residual_uses_target_reprojection(self) -> None:
        pointmap = torch.tensor(
            [
                [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]],
                [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]],
            ],
            dtype=torch.float32,
        )
        intrinsics = PinholeIntrinsics(focal_x=4.0, focal_y=4.0, cx=1.0, cy=1.0)
        residual = self.adapter.pointmap_multiview_correspondence_residual(
            pointmap,
            source_uv=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            target_uv=torch.tensor([[3.0, 1.0]], dtype=torch.float32),
            source_rotation=torch.eye(3, dtype=torch.float32),
            source_translation=torch.zeros(3, dtype=torch.float32),
            target_rotations=torch.eye(3, dtype=torch.float32).unsqueeze(0),
            target_translations=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            target_intrinsics=intrinsics,
        )
        self.assertLess(residual.item(), 1e-6)

        wrong_residual = self.adapter.pointmap_multiview_correspondence_residual(
            pointmap,
            source_uv=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            target_uv=torch.tensor([[2.0, 1.0]], dtype=torch.float32),
            source_rotation=torch.eye(3, dtype=torch.float32),
            source_translation=torch.zeros(3, dtype=torch.float32),
            target_rotations=torch.eye(3, dtype=torch.float32).unsqueeze(0),
            target_translations=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            target_intrinsics=intrinsics,
        )
        self.assertGreater(wrong_residual.item(), 0.1)

    def test_prior_state_prefers_corrected_only_when_intrinsics_are_stable(self) -> None:
        stable = self.adapter.classify_prior_state(
            current_intrinsics=self.current,
            principal_point_std_px=1.0,
            principal_point_rotation_sensitivity=1e-3,
            schur_condition_number=1e4,
            rerun_available=True,
        )
        self.assertEqual(stable.state, "corrected")
        self.assertTrue(stable.rerun_recommended)

        unstable = self.adapter.classify_prior_state(
            current_intrinsics=self.current,
            principal_point_std_px=20.0,
            principal_point_rotation_sensitivity=1e-3,
            schur_condition_number=1e4,
            rerun_available=True,
        )
        self.assertEqual(unstable.state, "filtered")

        debiased = self.adapter.classify_prior_state(
            current_intrinsics=self.current,
            principal_point_std_px=4.0,
            principal_point_rotation_sensitivity=1e-3,
            schur_condition_number=1e4,
            rerun_available=True,
            config=PriorStateConfig(corrected_principal_point_std_px_max=2.0),
        )
        self.assertEqual(debiased.state, "debiased")

    def test_depth_affine_fit_and_application(self) -> None:
        source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
        )
        target = source * 1.5 + 0.25
        affine = self.adapter.fit_depth_affine(source_depth=source, target_depth=target)
        self.assertAlmostEqual(affine.scale, 1.5, places=5)
        self.assertAlmostEqual(affine.bias, 0.25, places=5)
        corrected = self.adapter.correct_depth(
            source,
            current_intrinsics=self.assumed,
            mode="nearest",
            affine_correction=affine,
        )
        self.assertTrue(torch.allclose(corrected, target))

    def test_normal_pointmap_consistency_residual_detects_mismatch(self) -> None:
        x = torch.linspace(-1.0, 1.0, 5)
        y = torch.linspace(-1.0, 1.0, 5)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        zz = 2.0 + 0.2 * xx - 0.1 * yy
        pointmap = torch.stack([xx, yy, zz], dim=-1).float()
        normal = torch.tensor([-0.2, 0.1, 1.0], dtype=torch.float32)
        normal = torch.nn.functional.normalize(normal, dim=0)
        normals = normal.view(1, 1, 3).expand(5, 5, 3).clone()

        residual = self.adapter.normal_pointmap_consistency_residual(normals, pointmap=pointmap)
        self.assertLess(residual.item(), 0.1)

        wrong_normals = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).view(1, 1, 3).expand(5, 5, 3)
        wrong_residual = self.adapter.normal_pointmap_consistency_residual(
            wrong_normals,
            pointmap=pointmap,
        )
        self.assertGreater(wrong_residual.item(), 0.5)

    def test_channel_gates_are_per_prior(self) -> None:
        signals = self.adapter.build_channel_signals(
            current_intrinsics=self.current,
            pointmap_geometry_residual=0.1,
            pose_geometry_residual=0.1,
            depth_geometry_residual=1.0,
            normal_geometry_residual=0.2,
        )
        gates = self.adapter.gate_channels(
            signals=signals,
            config=ChannelGatingConfig(
                tau_pointmap=1.0,
                tau_pose=1.0,
                tau_depth=1.0,
                tau_normal=1.0,
                eta_k=1.0,
                eta_pointmap_r=0.5,
                eta_pose_r=0.5,
                eta_depth_r=2.0,
                eta_normal_r=0.5,
            ),
        )
        self.assertGreater(gates.pointmap.item(), gates.depth.item())
        self.assertGreater(gates.pose.item(), gates.depth.item())
        self.assertGreater(gates.normals.item(), gates.depth.item())


if __name__ == "__main__":
    unittest.main()
