from __future__ import annotations

import cv2
import numpy as np


def _pinhole_matrix(*, focal_x: float, focal_y: float, cx: float, cy: float) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[0, 0] = float(focal_x)
    matrix[1, 1] = float(focal_y)
    matrix[0, 2] = float(cx)
    matrix[1, 2] = float(cy)
    return matrix


def warp_image_between_intrinsics(
    image: np.ndarray,
    *,
    source_focal_x: float,
    source_focal_y: float,
    source_cx: float,
    source_cy: float,
    target_focal_x: float,
    target_focal_y: float,
    target_cx: float,
    target_cy: float,
    output_size: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    height, width = image.shape[:2]
    output_width, output_height = output_size or (width, height)

    source_K = _pinhole_matrix(
        focal_x=source_focal_x,
        focal_y=source_focal_y,
        cx=source_cx,
        cy=source_cy,
    )
    target_K = _pinhole_matrix(
        focal_x=target_focal_x,
        focal_y=target_focal_y,
        cx=target_cx,
        cy=target_cy,
    )
    homography = source_K @ np.linalg.inv(target_K)

    target_u = np.arange(output_width, dtype=np.float32)
    target_v = np.arange(output_height, dtype=np.float32)
    target_u_grid, target_v_grid = np.meshgrid(target_u, target_v, indexing="xy")
    target_h = np.stack(
        [target_u_grid, target_v_grid, np.ones_like(target_u_grid)],
        axis=-1,
    )
    source_h = target_h @ homography.T
    source_u = (source_h[..., 0] / source_h[..., 2]).astype(np.float32)
    source_v = (source_h[..., 1] / source_h[..., 2]).astype(np.float32)
    return cv2.remap(
        image,
        source_u,
        source_v,
        interpolation=interpolation,
        borderMode=border_mode,
    )


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
    estimated_focal_x: float | None = None,
    estimated_focal_y: float | None = None,
    target_focal_x: float | None = None,
    target_focal_y: float | None = None,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    focal_args = (
        estimated_focal_x,
        estimated_focal_y,
        target_focal_x,
        target_focal_y,
    )
    if any(value is not None for value in focal_args):
        estimated_focal_x = target_focal_x if estimated_focal_x is None else estimated_focal_x
        estimated_focal_y = target_focal_y if estimated_focal_y is None else estimated_focal_y
        target_focal_x = estimated_focal_x if target_focal_x is None else target_focal_x
        target_focal_y = estimated_focal_y if target_focal_y is None else target_focal_y
        return warp_image_between_intrinsics(
            image,
            source_focal_x=estimated_focal_x,
            source_focal_y=estimated_focal_y,
            source_cx=estimated_cx,
            source_cy=estimated_cy,
            target_focal_x=target_focal_x,
            target_focal_y=target_focal_y,
            target_cx=target_cx,
            target_cy=target_cy,
            border_mode=border_mode,
        )
    return shift_image_principal_point(
        image,
        delta_cx=estimated_cx - target_cx,
        delta_cy=estimated_cy - target_cy,
        border_mode=border_mode,
    )
