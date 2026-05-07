from __future__ import annotations

import cv2
import numpy as np


def shift_image_principal_point(
    image: np.ndarray,
    *,
    delta_cx: float,
    delta_cy: float,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.array([[1.0, 0.0, -delta_cx], [0.0, 1.0, -delta_cy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
    )


def recenter_image_from_estimated_intrinsics(
    image: np.ndarray,
    *,
    estimated_cx: float,
    estimated_cy: float,
    target_cx: float,
    target_cy: float,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    return shift_image_principal_point(
        image,
        delta_cx=estimated_cx - target_cx,
        delta_cy=estimated_cy - target_cy,
        border_mode=border_mode,
    )
