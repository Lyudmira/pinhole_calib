import unittest
from pathlib import Path
import tempfile

import numpy as np

from pinhole_calib import PinholeIntrinsics
from pinhole_calib.separable_solver import (
    SeparableSharedIntrinsicSolver,
    load_colmap_correspondences_by_image,
    load_colmap_tracks_by_image,
)


class SeparableSolverTest(unittest.TestCase):
    def test_recovers_shared_intrinsics_from_shifted_observations(self) -> None:
        intrinsics = PinholeIntrinsics(
            focal_x=900.0,
            focal_y=880.0,
            cx=320.0,
            cy=240.0,
        )
        xyz = np.array(
            [
                [0.1, 0.0, 2.0],
                [0.2, -0.1, 2.5],
                [-0.3, 0.15, 3.0],
                [0.05, 0.2, 1.5],
                [-0.15, -0.25, 2.2],
                [0.3, 0.1, 4.0],
            ],
            dtype=np.float64,
        )
        uv = np.stack(
            [
                intrinsics.focal_x * xyz[:, 0] / xyz[:, 2] + intrinsics.cx,
                intrinsics.focal_y * xyz[:, 1] / xyz[:, 2] + intrinsics.cy,
            ],
            axis=-1,
        )
        target = PinholeIntrinsics(
            focal_x=intrinsics.focal_x,
            focal_y=intrinsics.focal_y,
            cx=intrinsics.cx + 17.0,
            cy=intrinsics.cy - 11.0,
        )
        uv_shifted = np.stack(
            [
                target.focal_x * xyz[:, 0] / xyz[:, 2] + target.cx,
                target.focal_y * xyz[:, 1] / xyz[:, 2] + target.cy,
            ],
            axis=-1,
        )
        result = SeparableSharedIntrinsicSolver().fit(
            camera_xyz=xyz,
            observed_uv=uv_shifted,
            init_intrinsics=intrinsics,
        )
        self.assertAlmostEqual(result.intrinsics.focal_x, target.focal_x, places=5)
        self.assertAlmostEqual(result.intrinsics.focal_y, target.focal_y, places=5)
        self.assertAlmostEqual(result.intrinsics.cx, target.cx, places=5)
        self.assertAlmostEqual(result.intrinsics.cy, target.cy, places=5)
        self.assertLess(result.residual_rms, 1e-6)
        self.assertTrue(np.isfinite(result.cx_std))
        self.assertTrue(np.isfinite(result.cy_std))
        self.assertGreater(result.normal_matrix_condition, 0.0)

    def test_load_colmap_correspondences_by_image(self) -> None:
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

            observations = load_colmap_tracks_by_image(txt_dir)
            correspondences = load_colmap_correspondences_by_image(txt_dir)

        self.assertIn("image1.jpg", observations)
        self.assertIn("image2.jpg", observations)
        np.testing.assert_array_equal(
            observations["image1.jpg"].translation,
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            observations["image2.jpg"].translation,
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            correspondences["image1.jpg"].source_uv,
            np.array([[10.0, 20.0]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            correspondences["image1.jpg"].target_image_ids,
            np.array([2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            correspondences["image1.jpg"].target_uv,
            np.array([[12.0, 20.0]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            correspondences["image2.jpg"].source_uv,
            np.array([[12.0, 20.0]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            correspondences["image2.jpg"].target_image_ids,
            np.array([1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            correspondences["image2.jpg"].target_uv,
            np.array([[10.0, 20.0]], dtype=np.float64),
        )


if __name__ == "__main__":
    unittest.main()
