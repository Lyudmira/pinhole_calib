import unittest

import numpy as np

from pinhole_calib import PinholeIntrinsics
from pinhole_calib.separable_solver import SeparableSharedIntrinsicSolver


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


if __name__ == "__main__":
    unittest.main()
