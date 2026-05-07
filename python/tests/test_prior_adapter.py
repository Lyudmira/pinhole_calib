import unittest

import torch

from pinhole_calib import GatingConfig, MiscalibrationPriorAdapter, PinholeIntrinsics
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
            bias_alignment=0.1,
        )
        large = self.adapter.build_default_signals(
            current_intrinsics=PinholeIntrinsics(400.0, 420.0, 360.0, 210.0),
            geometry_residual=1.0,
            bias_alignment=1.0,
        )
        config = GatingConfig(tau=2.0)
        self.assertGreater(
            self.adapter.gate(signals=small, config=config).item(),
            self.adapter.gate(signals=large, config=config).item(),
        )


if __name__ == "__main__":
    unittest.main()
