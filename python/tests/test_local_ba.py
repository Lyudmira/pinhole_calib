import tempfile
import unittest
from pathlib import Path

import numpy as np

from pinhole_calib import (
    PinholeIntrinsics,
    SharedKBundleAdjuster,
    compute_shared_k_schur_response,
    load_colmap_local_ba_problem,
    principal_point_stability_metrics,
)
from pinhole_calib.local_ba import LocalBAImage, LocalBAProblem


class LocalBATest(unittest.TestCase):
    def test_load_colmap_local_ba_problem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_dir = Path(tmp_dir)
            (txt_dir / "images.txt").write_text(
                "\n".join(
                    [
                        "# Image list with two lines of data per image:",
                        "1 1 0 0 0 0 0 0 1 image1.jpg",
                        "10 20 1",
                        "2 1 0 0 0 1 0 0 1 image2.jpg",
                        "12 20 1",
                    ]
                )
                + "\n"
            )
            (txt_dir / "points3D.txt").write_text(
                "\n".join(
                    [
                        "# 3D point list with one line of data per point:",
                        "1 5 0 10 0 0 0 0 1 0 2 0",
                    ]
                )
                + "\n"
            )
            problem = load_colmap_local_ba_problem(txt_dir)

        self.assertEqual(len(problem.images), 2)
        self.assertEqual(problem.points.shape, (1, 3))
        self.assertEqual(problem.observed_uv.shape, (2, 2))
        np.testing.assert_array_equal(problem.observation_image_indices, np.array([0, 1], dtype=np.int64))

    def test_schur_response_is_finite_and_anchor_aware(self) -> None:
        intrinsics = PinholeIntrinsics(focal_x=800.0, focal_y=820.0, cx=320.0, cy=240.0)
        points = []
        for z in (2.0, 3.0, 4.5, 6.0, 8.0):
            for x in (-0.2, -0.05, 0.1, 0.25):
                for y in (-0.1, 0.0, 0.12):
                    points.append([x, y, z])
        points = np.asarray(points, dtype=np.float64)
        problem = LocalBAProblem(
            images=[
                LocalBAImage(
                    image_id=1,
                    image_name="anchor.jpg",
                    rotation=np.eye(3, dtype=np.float64),
                    translation=np.zeros(3, dtype=np.float64),
                ),
                LocalBAImage(
                    image_id=2,
                    image_name="target_x.jpg",
                    rotation=np.eye(3, dtype=np.float64),
                    translation=np.array([0.4, 0.0, 0.0], dtype=np.float64),
                ),
                LocalBAImage(
                    image_id=3,
                    image_name="target_y.jpg",
                    rotation=np.eye(3, dtype=np.float64),
                    translation=np.array([0.0, 0.25, 0.0], dtype=np.float64),
                ),
            ],
            points=points,
            point_ids=np.arange(len(points), dtype=np.int64),
            observation_image_indices=np.concatenate(
                [
                    np.full((len(points),), 0, dtype=np.int64),
                    np.full((len(points),), 1, dtype=np.int64),
                    np.full((len(points),), 2, dtype=np.int64),
                ]
            ),
            observation_point_indices=np.concatenate(
                [
                    np.arange(len(points), dtype=np.int64),
                    np.arange(len(points), dtype=np.int64),
                    np.arange(len(points), dtype=np.int64),
                ]
            ),
            observed_uv=np.zeros((3 * len(points), 2), dtype=np.float64),
        )
        observed_uv = []
        for image_index, point_index in zip(
            problem.observation_image_indices,
            problem.observation_point_indices,
        ):
            image = problem.images[int(image_index)]
            point = problem.points[int(point_index)]
            xyz = image.rotation @ point + image.translation
            observed_uv.append(
                [
                    intrinsics.focal_x * xyz[0] / xyz[2] + intrinsics.cx,
                    intrinsics.focal_y * xyz[1] / xyz[2] + intrinsics.cy,
                ]
            )
        problem.observed_uv = np.asarray(observed_uv, dtype=np.float64)

        response = compute_shared_k_schur_response(problem, intrinsics=intrinsics, anchor_image_index=0)
        response_by_name = response.response_by_name()
        self.assertIn("anchor.jpg", response_by_name)
        self.assertIn("target_x.jpg", response_by_name)
        self.assertIn("target_y.jpg", response_by_name)
        anchor_norm = np.linalg.norm(response_by_name["anchor.jpg"])
        target_x_norm = np.linalg.norm(response_by_name["target_x.jpg"])
        target_y_norm = np.linalg.norm(response_by_name["target_y.jpg"])
        self.assertLess(anchor_norm, 1e-10)
        self.assertGreater(target_x_norm, anchor_norm * 1000.0)
        self.assertGreater(target_y_norm, anchor_norm * 1000.0)
        self.assertTrue(np.isfinite(response.response_matrices).all())

        metrics = principal_point_stability_metrics(response, intrinsics=intrinsics)
        self.assertTrue(np.isfinite(metrics["principal_point_response_sensitivity_px"]))
        self.assertGreater(metrics["principal_point_rotation_sensitivity"], 0.0)

    def test_shared_k_bundle_adjuster_reduces_intrinsics_error(self) -> None:
        true_intrinsics = PinholeIntrinsics(
            focal_x=800.0,
            focal_y=820.0,
            cx=320.0,
            cy=240.0,
        )
        points = []
        for z in (2.0, 3.5, 5.0, 7.0):
            for x in (-0.3, -0.05, 0.2):
                for y in (-0.15, 0.05):
                    points.append([x, y, z])
        points = np.asarray(points, dtype=np.float64)
        images = [
            LocalBAImage(1, "a.jpg", np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
            LocalBAImage(2, "b.jpg", np.eye(3, dtype=np.float64), np.array([0.4, 0.0, 0.0], dtype=np.float64)),
            LocalBAImage(3, "c.jpg", np.eye(3, dtype=np.float64), np.array([0.0, 0.3, 0.0], dtype=np.float64)),
        ]
        obs_img = []
        obs_pt = []
        obs_uv = []
        for image_index, image in enumerate(images):
            for point_index, point in enumerate(points):
                point_cam = image.rotation @ point + image.translation
                obs_img.append(image_index)
                obs_pt.append(point_index)
                obs_uv.append(
                    [
                        true_intrinsics.focal_x * point_cam[0] / point_cam[2] + true_intrinsics.cx,
                        true_intrinsics.focal_y * point_cam[1] / point_cam[2] + true_intrinsics.cy,
                    ]
                )
        problem = LocalBAProblem(
            images=images,
            points=points.copy(),
            point_ids=np.arange(len(points), dtype=np.int64),
            observation_image_indices=np.asarray(obs_img, dtype=np.int64),
            observation_point_indices=np.asarray(obs_pt, dtype=np.int64),
            observed_uv=np.asarray(obs_uv, dtype=np.float64),
        )
        init_intrinsics = PinholeIntrinsics(
            focal_x=790.0,
            focal_y=815.0,
            cx=327.0,
            cy=235.0,
        )
        result = SharedKBundleAdjuster(anchor_image_index=0, max_iterations=10).fit(
            problem=problem,
            init_intrinsics=init_intrinsics,
        )

        init_error = np.linalg.norm(
            np.array(
                [
                    float(init_intrinsics.focal_x) - float(true_intrinsics.focal_x),
                    float(init_intrinsics.focal_y) - float(true_intrinsics.focal_y),
                    float(init_intrinsics.cx) - float(true_intrinsics.cx),
                    float(init_intrinsics.cy) - float(true_intrinsics.cy),
                ],
                dtype=np.float64,
            )
        )
        final_error = np.linalg.norm(
            np.array(
                [
                    float(result.intrinsics.focal_x) - float(true_intrinsics.focal_x),
                    float(result.intrinsics.focal_y) - float(true_intrinsics.focal_y),
                    float(result.intrinsics.cx) - float(true_intrinsics.cx),
                    float(result.intrinsics.cy) - float(true_intrinsics.cy),
                ],
                dtype=np.float64,
            )
        )
        self.assertLess(result.residual_rms, result.initial_residual_rms)
        self.assertLess(final_error, init_error)

    def test_observation_offset_matches_shifted_principal_point_problem(self) -> None:
        base_intrinsics = PinholeIntrinsics(
            focal_x=800.0,
            focal_y=820.0,
            cx=320.0,
            cy=240.0,
        )
        target_intrinsics = PinholeIntrinsics(
            focal_x=800.0,
            focal_y=820.0,
            cx=332.0,
            cy=234.0,
        )
        points = np.asarray(
            [
                [-0.3, -0.1, 2.0],
                [-0.2, 0.2, 2.7],
                [0.1, -0.15, 3.5],
                [0.25, 0.1, 4.2],
                [0.3, 0.25, 5.0],
            ],
            dtype=np.float64,
        )
        images = [
            LocalBAImage(1, "a.jpg", np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
            LocalBAImage(2, "b.jpg", np.eye(3, dtype=np.float64), np.array([0.4, 0.0, 0.0], dtype=np.float64)),
            LocalBAImage(3, "c.jpg", np.eye(3, dtype=np.float64), np.array([0.0, 0.3, 0.0], dtype=np.float64)),
        ]
        obs_img = []
        obs_pt = []
        obs_uv = []
        for image_index, image in enumerate(images):
            for point_index, point in enumerate(points):
                point_cam = image.rotation @ point + image.translation
                obs_img.append(image_index)
                obs_pt.append(point_index)
                obs_uv.append(
                    [
                        base_intrinsics.focal_x * point_cam[0] / point_cam[2] + base_intrinsics.cx,
                        base_intrinsics.focal_y * point_cam[1] / point_cam[2] + base_intrinsics.cy,
                    ]
                )
        problem = LocalBAProblem(
            images=images,
            points=points.copy(),
            point_ids=np.arange(len(points), dtype=np.int64),
            observation_image_indices=np.asarray(obs_img, dtype=np.int64),
            observation_point_indices=np.asarray(obs_pt, dtype=np.int64),
            observed_uv=np.asarray(obs_uv, dtype=np.float64),
        )

        shifted_problem = problem.with_observation_offset(
            np.array(
                [
                    target_intrinsics.cx - base_intrinsics.cx,
                    target_intrinsics.cy - base_intrinsics.cy,
                ],
                dtype=np.float64,
            )
        )
        shifted_result = SharedKBundleAdjuster(anchor_image_index=0, max_iterations=1).fit(
            problem=shifted_problem,
            init_intrinsics=target_intrinsics,
        )
        base_result = SharedKBundleAdjuster(anchor_image_index=0, max_iterations=1).fit(
            problem=problem,
            init_intrinsics=target_intrinsics,
        )

        self.assertLess(shifted_result.initial_residual_rms, 1e-6)
        self.assertGreater(base_result.initial_residual_rms, 1.0)


if __name__ == "__main__":
    unittest.main()
