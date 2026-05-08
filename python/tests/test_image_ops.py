import unittest

import cv2
import numpy as np

from pinhole_calib.image_ops import (
    recenter_image_from_estimated_intrinsics,
    shift_image_principal_point,
)


class ImageOpsTest(unittest.TestCase):
    def test_synthetic_principal_point_shift_uses_inverse_warp_sign(self) -> None:
        image = np.arange(99, dtype=np.uint8).reshape(9, 11)
        target_cx = 5.0
        target_cy = 4.0
        principal_delta_cx = 2.0
        principal_delta_cy = -1.0

        shifted = shift_image_principal_point(
            image,
            delta_cx=-principal_delta_cx,
            delta_cy=-principal_delta_cy,
            border_mode=cv2.BORDER_CONSTANT,
        )
        recentered = recenter_image_from_estimated_intrinsics(
            shifted,
            estimated_cx=target_cx + principal_delta_cx,
            estimated_cy=target_cy + principal_delta_cy,
            target_cx=target_cx,
            target_cy=target_cy,
            border_mode=cv2.BORDER_CONSTANT,
        )

        np.testing.assert_array_equal(
            recentered[1:-1, 2:-2],
            image[1:-1, 2:-2],
        )

    def test_general_recenter_reduces_to_translation_when_focals_match(self) -> None:
        image = np.arange(30, dtype=np.uint8).reshape(5, 6)
        expected = shift_image_principal_point(
            image,
            delta_cx=1.0,
            delta_cy=-1.0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        actual = recenter_image_from_estimated_intrinsics(
            image,
            estimated_cx=3.0,
            estimated_cy=1.0,
            target_cx=2.0,
            target_cy=2.0,
            estimated_focal_x=4.0,
            estimated_focal_y=4.0,
            target_focal_x=4.0,
            target_focal_y=4.0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_general_recenter_is_identity_when_intrinsics_match(self) -> None:
        image = np.arange(45, dtype=np.uint8).reshape(5, 9)
        actual = recenter_image_from_estimated_intrinsics(
            image,
            estimated_cx=4.0,
            estimated_cy=2.0,
            target_cx=4.0,
            target_cy=2.0,
            estimated_focal_x=7.5,
            estimated_focal_y=6.0,
            target_focal_x=7.5,
            target_focal_y=6.0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        np.testing.assert_array_equal(actual, image)


if __name__ == "__main__":
    unittest.main()
