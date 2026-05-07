from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from .intrinsics import PinholeIntrinsics


@dataclass(slots=True)
class LocalBAImage:
    image_id: int
    image_name: str
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(slots=True)
class LocalBAProblem:
    images: list[LocalBAImage]
    points: np.ndarray
    point_ids: np.ndarray
    observation_image_indices: np.ndarray
    observation_point_indices: np.ndarray
    observed_uv: np.ndarray

    def with_observation_offset(self, delta_uv: np.ndarray) -> "LocalBAProblem":
        delta_uv = np.asarray(delta_uv, dtype=np.float64).reshape(1, 2)
        return LocalBAProblem(
            images=self.images,
            points=self.points,
            point_ids=self.point_ids,
            observation_image_indices=self.observation_image_indices,
            observation_point_indices=self.observation_point_indices,
            observed_uv=self.observed_uv + delta_uv,
        )


@dataclass(slots=True)
class LocalBASchurResponse:
    image_names: list[str]
    image_ids: np.ndarray
    response_matrices: np.ndarray
    schur_matrix: np.ndarray
    gamma_matrix: np.ndarray
    covariance: np.ndarray
    anchor_image_index: int

    def response_by_name(self) -> dict[str, np.ndarray]:
        return {
            image_name: self.response_matrices[idx]
            for idx, image_name in enumerate(self.image_names)
        }


@dataclass(slots=True)
class SharedKBundleAdjustmentResult:
    intrinsics: PinholeIntrinsics
    images: list[LocalBAImage]
    points: np.ndarray
    residual_rms: float
    initial_residual_rms: float
    iterations: int
    converged: bool


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def load_colmap_local_ba_problem(txt_dir: Path) -> LocalBAProblem:
    txt_dir = Path(txt_dir)
    points: list[np.ndarray] = []
    point_ids: list[int] = []
    point_id_to_index: dict[int, int] = {}
    tracks: dict[int, list[tuple[int, int]]] = {}
    with open(txt_dir / "points3D.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            point_id_to_index[point_id] = len(points)
            point_ids.append(point_id)
            points.append(np.array(list(map(float, parts[1:4])), dtype=np.float64))
            track_tokens = parts[8:]
            tracks[point_id] = [
                (int(track_tokens[i]), int(track_tokens[i + 1]))
                for i in range(0, len(track_tokens), 2)
            ]

    images_by_id: dict[int, LocalBAImage] = {}
    observations_by_image: dict[int, list[tuple[np.ndarray, int]]] = {}
    with open(txt_dir / "images.txt") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        image_name = " ".join(parts[9:])
        images_by_id[image_id] = LocalBAImage(
            image_id=image_id,
            image_name=image_name,
            rotation=_qvec_to_rotmat(qvec),
            translation=tvec,
        )
        cur_observations: list[tuple[np.ndarray, int]] = []
        if i < len(lines):
            points_line = lines[i].strip()
            i += 1
            if points_line:
                vals = points_line.split()
                triples = [vals[j : j + 3] for j in range(0, len(vals), 3)]
                for x_s, y_s, pid_s in triples:
                    point_id = int(pid_s)
                    if point_id < 0 or point_id not in point_id_to_index:
                        continue
                    cur_observations.append(
                        (
                            np.array([float(x_s), float(y_s)], dtype=np.float64),
                            point_id_to_index[point_id],
                        )
                    )
        observations_by_image[image_id] = cur_observations

    image_ids_sorted = sorted(images_by_id)
    image_index_by_id = {image_id: idx for idx, image_id in enumerate(image_ids_sorted)}
    images = [images_by_id[image_id] for image_id in image_ids_sorted]

    observation_image_indices: list[int] = []
    observation_point_indices: list[int] = []
    observed_uv: list[np.ndarray] = []
    for image_id in image_ids_sorted:
        for uv, point_index in observations_by_image.get(image_id, []):
            observation_image_indices.append(image_index_by_id[image_id])
            observation_point_indices.append(point_index)
            observed_uv.append(uv)

    if not observed_uv:
        uv_array = np.zeros((0, 2), dtype=np.float64)
    else:
        uv_array = np.asarray(observed_uv, dtype=np.float64)

    return LocalBAProblem(
        images=images,
        points=np.asarray(points, dtype=np.float64) if points else np.zeros((0, 3), dtype=np.float64),
        point_ids=np.asarray(point_ids, dtype=np.int64) if point_ids else np.zeros((0,), dtype=np.int64),
        observation_image_indices=np.asarray(observation_image_indices, dtype=np.int64),
        observation_point_indices=np.asarray(observation_point_indices, dtype=np.int64),
        observed_uv=uv_array,
    )


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + _skew(omega)
    omega_hat = _skew(omega)
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3, dtype=np.float64) + a * omega_hat + b * (omega_hat @ omega_hat)


def _apply_left_pose_increment(image: LocalBAImage, delta_xi: np.ndarray) -> LocalBAImage:
    rotation_inc = _so3_exp(delta_xi[3:])
    translation_inc = delta_xi[:3]
    return LocalBAImage(
        image_id=image.image_id,
        image_name=image.image_name,
        rotation=rotation_inc @ image.rotation,
        translation=rotation_inc @ image.translation + translation_inc,
    )


def _compute_reprojection_residuals(
    *,
    problem: LocalBAProblem,
    intrinsics: PinholeIntrinsics,
    images: list[LocalBAImage],
    points: np.ndarray,
) -> np.ndarray:
    residuals = []
    fx = float(intrinsics.focal_x)
    fy = float(intrinsics.focal_y)
    cx = float(intrinsics.cx)
    cy = float(intrinsics.cy)
    for image_index, point_index, observed_uv in zip(
        problem.observation_image_indices,
        problem.observation_point_indices,
        problem.observed_uv,
    ):
        image = images[int(image_index)]
        point = points[int(point_index)]
        point_cam = image.rotation @ point + image.translation
        if point_cam[2] <= 1e-8:
            continue
        predicted_uv = np.array(
            [
                fx * point_cam[0] / point_cam[2] + cx,
                fy * point_cam[1] / point_cam[2] + cy,
            ],
            dtype=np.float64,
        )
        residuals.append(predicted_uv - observed_uv)
    if not residuals:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray(residuals, dtype=np.float64).reshape(-1)


class SharedKBundleAdjuster:
    def __init__(
        self,
        *,
        anchor_image_index: int = 0,
        max_iterations: int = 15,
        lm_damping: float = 1e-4,
        step_tolerance: float = 1e-9,
    ) -> None:
        self.anchor_image_index = int(anchor_image_index)
        self.max_iterations = int(max_iterations)
        self.lm_damping = float(lm_damping)
        self.step_tolerance = float(step_tolerance)

    def fit(
        self,
        *,
        problem: LocalBAProblem,
        init_intrinsics: PinholeIntrinsics,
    ) -> SharedKBundleAdjustmentResult:
        images = [
            LocalBAImage(
                image_id=image.image_id,
                image_name=image.image_name,
                rotation=image.rotation.copy(),
                translation=image.translation.copy(),
            )
            for image in problem.images
        ]
        points = problem.points.copy()
        intrinsics = PinholeIntrinsics(
            focal_x=float(init_intrinsics.focal_x),
            focal_y=float(init_intrinsics.focal_y),
            cx=float(init_intrinsics.cx),
            cy=float(init_intrinsics.cy),
        )
        pose_param_offsets: dict[int, int] = {}
        offset = 4
        for image_index in range(len(images)):
            if image_index == self.anchor_image_index:
                continue
            pose_param_offsets[image_index] = offset
            offset += 6
        param_dim = offset

        residuals = _compute_reprojection_residuals(
            problem=problem,
            intrinsics=intrinsics,
            images=images,
            points=points,
        )
        initial_residual_rms = float(math.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
        current_residual_rms = initial_residual_rms
        damping = self.lm_damping
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            U = np.zeros((param_dim, param_dim), dtype=np.float64)
            W = np.zeros((points.shape[0], param_dim, 3), dtype=np.float64)
            V = np.zeros((points.shape[0], 3, 3), dtype=np.float64)
            g_a = np.zeros((param_dim,), dtype=np.float64)
            g_b = np.zeros((points.shape[0], 3), dtype=np.float64)

            fx = float(intrinsics.focal_x)
            fy = float(intrinsics.focal_y)
            cx = float(intrinsics.cx)
            cy = float(intrinsics.cy)

            for image_index, point_index, observed_uv in zip(
                problem.observation_image_indices,
                problem.observation_point_indices,
                problem.observed_uv,
            ):
                image_index = int(image_index)
                point_index = int(point_index)
                image = images[image_index]
                point = points[point_index]
                point_cam = image.rotation @ point + image.translation
                x_c, y_c, z_c = point_cam
                if z_c <= 1e-8:
                    continue

                predicted_uv = np.array(
                    [
                        fx * x_c / z_c + cx,
                        fy * y_c / z_c + cy,
                    ],
                    dtype=np.float64,
                )
                residual = predicted_uv - observed_uv
                gp = np.array(
                    [
                        [fx / z_c, 0.0, -fx * x_c / (z_c * z_c)],
                        [0.0, fy / z_c, -fy * y_c / (z_c * z_c)],
                    ],
                    dtype=np.float64,
                )
                g_x = gp @ image.rotation
                A = np.zeros((2, param_dim), dtype=np.float64)
                A[:, :4] = np.array(
                    [
                        [x_c / z_c, 0.0, 1.0, 0.0],
                        [0.0, y_c / z_c, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                if image_index != self.anchor_image_index:
                    pose_offset = pose_param_offsets[image_index]
                    A[:, pose_offset : pose_offset + 6] = gp @ np.concatenate(
                        [np.eye(3), -_skew(point_cam)],
                        axis=1,
                    )

                U += A.T @ A
                W[point_index] += A.T @ g_x
                V[point_index] += g_x.T @ g_x
                g_a += A.T @ residual
                g_b[point_index] += g_x.T @ residual

            U += damping * np.eye(param_dim, dtype=np.float64)
            schur = U.copy()
            rhs = g_a.copy()
            point_inv = np.zeros_like(V)
            for point_index in range(points.shape[0]):
                point_inv[point_index] = np.linalg.pinv(
                    V[point_index] + damping * np.eye(3, dtype=np.float64)
                )
                schur -= W[point_index] @ point_inv[point_index] @ W[point_index].T
                rhs -= W[point_index] @ point_inv[point_index] @ g_b[point_index]

            delta_a = -np.linalg.pinv(schur) @ rhs
            delta_points = np.zeros_like(points)
            for point_index in range(points.shape[0]):
                delta_points[point_index] = -point_inv[point_index] @ (
                    g_b[point_index] + W[point_index].T @ delta_a
                )

            candidate_intrinsics = PinholeIntrinsics(
                focal_x=max(1e-6, float(intrinsics.focal_x) + float(delta_a[0])),
                focal_y=max(1e-6, float(intrinsics.focal_y) + float(delta_a[1])),
                cx=float(intrinsics.cx) + float(delta_a[2]),
                cy=float(intrinsics.cy) + float(delta_a[3]),
            )
            candidate_images = []
            for image_index, image in enumerate(images):
                if image_index == self.anchor_image_index:
                    candidate_images.append(image)
                    continue
                pose_offset = pose_param_offsets[image_index]
                candidate_images.append(
                    _apply_left_pose_increment(
                        image,
                        delta_a[pose_offset : pose_offset + 6],
                    )
                )
            candidate_points = points + delta_points
            candidate_residuals = _compute_reprojection_residuals(
                problem=problem,
                intrinsics=candidate_intrinsics,
                images=candidate_images,
                points=candidate_points,
            )
            candidate_rms = (
                float(math.sqrt(np.mean(candidate_residuals ** 2)))
                if candidate_residuals.size
                else 0.0
            )

            if candidate_rms <= current_residual_rms:
                intrinsics = candidate_intrinsics
                images = candidate_images
                points = candidate_points
                if abs(current_residual_rms - candidate_rms) < self.step_tolerance:
                    converged = True
                    current_residual_rms = candidate_rms
                    break
                current_residual_rms = candidate_rms
                damping = max(damping * 0.5, 1e-9)
            else:
                damping = min(damping * 10.0, 1e6)

        return SharedKBundleAdjustmentResult(
            intrinsics=intrinsics,
            images=images,
            points=points,
            residual_rms=current_residual_rms,
            initial_residual_rms=initial_residual_rms,
            iterations=iteration,
            converged=converged,
        )


def compute_shared_k_schur_response(
    problem: LocalBAProblem,
    *,
    intrinsics: PinholeIntrinsics,
    anchor_image_index: int = 0,
    damping: float = 1e-9,
    anchor_weight: float = 1e6,
) -> LocalBASchurResponse:
    num_images = len(problem.images)
    num_points = int(problem.points.shape[0])
    if num_images == 0:
        raise ValueError("LocalBAProblem must contain at least one image.")
    if not (0 <= anchor_image_index < num_images):
        raise ValueError(f"anchor_image_index {anchor_image_index} is out of range.")

    fx = float(intrinsics.focal_x)
    fy = float(intrinsics.focal_y)

    U = np.zeros((6 * num_images, 6 * num_images), dtype=np.float64)
    Gamma_pose = np.zeros((6 * num_images, 4), dtype=np.float64)
    V = np.zeros((num_points, 3, 3), dtype=np.float64)
    Gamma_points = np.zeros((num_points, 3, 4), dtype=np.float64)
    W_by_point: list[dict[int, np.ndarray]] = [dict() for _ in range(num_points)]

    for image_index, point_index, uv in zip(
        problem.observation_image_indices,
        problem.observation_point_indices,
        problem.observed_uv,
    ):
        image = problem.images[int(image_index)]
        point = problem.points[int(point_index)]
        point_cam = image.rotation @ point + image.translation
        x_c, y_c, z_c = point_cam
        if z_c <= 1e-8:
            continue

        gp = np.array(
            [
                [fx / z_c, 0.0, -fx * x_c / (z_c * z_c)],
                [0.0, fy / z_c, -fy * y_c / (z_c * z_c)],
            ],
            dtype=np.float64,
        )
        g_xi = gp @ np.concatenate([np.eye(3), -_skew(point_cam)], axis=1)
        g_x = gp @ image.rotation
        g_k = np.array(
            [
                [x_c / z_c, 0.0, 1.0, 0.0],
                [0.0, y_c / z_c, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        cam_slice = slice(6 * int(image_index), 6 * (int(image_index) + 1))
        U[cam_slice, cam_slice] += g_xi.T @ g_xi
        Gamma_pose[cam_slice] += g_xi.T @ g_k
        V[int(point_index)] += g_x.T @ g_x
        Gamma_points[int(point_index)] += g_x.T @ g_k
        W_by_point[int(point_index)].setdefault(int(image_index), np.zeros((6, 3), dtype=np.float64))
        W_by_point[int(point_index)][int(image_index)] += g_xi.T @ g_x

    schur = U.copy()
    gamma = Gamma_pose.copy()
    for point_index in range(num_points):
        if not W_by_point[point_index]:
            continue
        v_inv = np.linalg.pinv(V[point_index] + damping * np.eye(3, dtype=np.float64))
        point_gamma = Gamma_points[point_index]
        point_blocks = W_by_point[point_index]
        for image_i, w_i in point_blocks.items():
            cam_i = slice(6 * image_i, 6 * (image_i + 1))
            gamma[cam_i] -= w_i @ v_inv @ point_gamma
            for image_j, w_j in point_blocks.items():
                cam_j = slice(6 * image_j, 6 * (image_j + 1))
                schur[cam_i, cam_j] -= w_i @ v_inv @ w_j.T

    anchor_slice = slice(6 * anchor_image_index, 6 * (anchor_image_index + 1))
    schur[anchor_slice, anchor_slice] += anchor_weight * np.eye(6, dtype=np.float64)
    covariance = np.linalg.pinv(schur + damping * np.eye(schur.shape[0], dtype=np.float64))
    response = -(covariance @ gamma)

    return LocalBASchurResponse(
        image_names=[image.image_name for image in problem.images],
        image_ids=np.asarray([image.image_id for image in problem.images], dtype=np.int64),
        response_matrices=response.reshape(num_images, 6, 4),
        schur_matrix=schur,
        gamma_matrix=gamma,
        covariance=covariance,
        anchor_image_index=anchor_image_index,
    )


def principal_point_stability_metrics(
    response: LocalBASchurResponse,
    *,
    intrinsics: PinholeIntrinsics,
) -> dict[str, float]:
    if response.response_matrices.size == 0:
        return {
            "principal_point_response_sensitivity_px": math.inf,
            "principal_point_rotation_sensitivity": math.inf,
            "schur_condition_number": math.inf,
        }
    principal_point_rot = []
    principal_point_px = []
    focal_scale = math.sqrt(float(intrinsics.focal_x) * float(intrinsics.focal_y))
    for image_response in response.response_matrices:
        principal_point_rot.append(np.linalg.norm(image_response[3:5, 2:4]))
        principal_point_px.append(np.linalg.norm(image_response[:, 2:4], ord=2) * focal_scale)
    return {
        "principal_point_response_sensitivity_px": float(np.mean(principal_point_px)),
        "principal_point_rotation_sensitivity": float(np.mean(principal_point_rot)),
        "schur_condition_number": float(np.linalg.cond(response.schur_matrix)),
    }
