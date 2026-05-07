from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import typing as t

import numpy as np

from .intrinsics import PinholeIntrinsics

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover
    least_squares = None


@dataclass(slots=True)
class SeparableSolveResult:
    intrinsics: PinholeIntrinsics
    residual_rms: float
    num_observations: int


@dataclass(slots=True)
class ColmapImageObservations:
    image_id: int
    image_name: str
    rotation: np.ndarray
    translation: np.ndarray
    camera_xyz: np.ndarray
    observed_uv: np.ndarray
    point_ids: np.ndarray


@dataclass(slots=True)
class ColmapImageCorrespondences:
    image_id: int
    image_name: str
    source_uv: np.ndarray
    target_image_ids: np.ndarray
    target_uv: np.ndarray
    point_ids: np.ndarray


@dataclass(slots=True)
class _ParsedColmapImage:
    image_id: int
    image_name: str
    rotation: np.ndarray
    translation: np.ndarray
    point2d_uv: np.ndarray
    point2d_ids: np.ndarray


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


def _load_points3d(txt_dir: Path) -> tuple[dict[int, np.ndarray], dict[int, list[tuple[int, int]]]]:
    points3d = {}
    tracks = {}
    with open(txt_dir / "points3D.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            xyz = np.array(list(map(float, parts[1:4])), dtype=np.float64)
            points3d[point_id] = xyz
            track_tokens = parts[8:]
            tracks[point_id] = [
                (int(track_tokens[i]), int(track_tokens[i + 1]))
                for i in range(0, len(track_tokens), 2)
            ]
    return points3d, tracks


def _to_uv_array(values: list[list[float]]) -> np.ndarray:
    if not values:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _to_xyz_array(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _to_int_array(values: list[int]) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


def _load_colmap_images(txt_dir: Path) -> dict[int, _ParsedColmapImage]:
    images_by_id: dict[int, _ParsedColmapImage] = {}
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
        rotation = _qvec_to_rotmat(qvec)

        point2d_uv: list[list[float]] = []
        point2d_ids: list[int] = []
        if i < len(lines):
            points_line = lines[i].strip()
            i += 1
            if points_line:
                vals = points_line.split()
                triples = [vals[j : j + 3] for j in range(0, len(vals), 3)]
                for x_s, y_s, pid_s in triples:
                    point2d_uv.append([float(x_s), float(y_s)])
                    point2d_ids.append(int(pid_s))

        images_by_id[image_id] = _ParsedColmapImage(
            image_id=image_id,
            image_name=image_name,
            rotation=rotation,
            translation=tvec,
            point2d_uv=_to_uv_array(point2d_uv),
            point2d_ids=_to_int_array(point2d_ids),
        )

    return images_by_id


def load_colmap_tracks_by_image(txt_dir: Path) -> dict[str, ColmapImageObservations]:
    points3d, _ = _load_points3d(txt_dir)
    images_by_id = _load_colmap_images(txt_dir)
    observations_by_image: dict[str, ColmapImageObservations] = {}
    for parsed_image in images_by_id.values():
        camera_xyz = []
        observed_uv = []
        point_ids = []
        for uv, point_id in zip(parsed_image.point2d_uv, parsed_image.point2d_ids):
            if point_id < 0 or point_id not in points3d:
                continue
            xyz_world = points3d[point_id]
            xyz_cam = parsed_image.rotation @ xyz_world + parsed_image.translation
            if xyz_cam[2] <= 1e-6:
                continue
            camera_xyz.append(xyz_cam)
            observed_uv.append([float(uv[0]), float(uv[1])])
            point_ids.append(int(point_id))

        observations_by_image[parsed_image.image_name] = ColmapImageObservations(
            image_id=parsed_image.image_id,
            image_name=parsed_image.image_name,
            rotation=parsed_image.rotation,
            translation=parsed_image.translation,
            camera_xyz=_to_xyz_array(camera_xyz),
            observed_uv=_to_uv_array(observed_uv),
            point_ids=_to_int_array(point_ids),
        )

    return observations_by_image


def load_colmap_correspondences_by_image(
    txt_dir: Path,
) -> dict[str, ColmapImageCorrespondences]:
    _, tracks = _load_points3d(txt_dir)
    images_by_id = _load_colmap_images(txt_dir)
    correspondences_by_image: dict[str, ColmapImageCorrespondences] = {
        parsed.image_name: ColmapImageCorrespondences(
            image_id=parsed.image_id,
            image_name=parsed.image_name,
            source_uv=np.zeros((0, 2), dtype=np.float64),
            target_image_ids=np.zeros((0,), dtype=np.int64),
            target_uv=np.zeros((0, 2), dtype=np.float64),
            point_ids=np.zeros((0,), dtype=np.int64),
        )
        for parsed in images_by_id.values()
    }
    source_uv_lists: dict[str, list[list[float]]] = {
        parsed.image_name: [] for parsed in images_by_id.values()
    }
    target_id_lists: dict[str, list[int]] = {
        parsed.image_name: [] for parsed in images_by_id.values()
    }
    target_uv_lists: dict[str, list[list[float]]] = {
        parsed.image_name: [] for parsed in images_by_id.values()
    }
    point_id_lists: dict[str, list[int]] = {
        parsed.image_name: [] for parsed in images_by_id.values()
    }

    for point_id, track in tracks.items():
        valid_track = [
            (image_id, point2d_idx)
            for image_id, point2d_idx in track
            if image_id in images_by_id
            and 0 <= point2d_idx < images_by_id[image_id].point2d_uv.shape[0]
        ]
        for source_image_id, source_idx in valid_track:
            source_image = images_by_id[source_image_id]
            source_name = source_image.image_name
            source_uv = source_image.point2d_uv[source_idx]
            for target_image_id, target_idx in valid_track:
                if target_image_id == source_image_id:
                    continue
                target_image = images_by_id[target_image_id]
                target_uv = target_image.point2d_uv[target_idx]
                source_uv_lists[source_name].append([float(source_uv[0]), float(source_uv[1])])
                target_id_lists[source_name].append(int(target_image_id))
                target_uv_lists[source_name].append([float(target_uv[0]), float(target_uv[1])])
                point_id_lists[source_name].append(int(point_id))

    for parsed in images_by_id.values():
        correspondences_by_image[parsed.image_name] = ColmapImageCorrespondences(
            image_id=parsed.image_id,
            image_name=parsed.image_name,
            source_uv=_to_uv_array(source_uv_lists[parsed.image_name]),
            target_image_ids=_to_int_array(target_id_lists[parsed.image_name]),
            target_uv=_to_uv_array(target_uv_lists[parsed.image_name]),
            point_ids=_to_int_array(point_id_lists[parsed.image_name]),
        )

    return correspondences_by_image


def load_colmap_tracks(txt_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations_xyz = []
    observations_uv = []
    counts = []
    for image_observations in load_colmap_tracks_by_image(txt_dir).values():
        if image_observations.camera_xyz.size == 0:
            continue
        observations_xyz.append(image_observations.camera_xyz)
        observations_uv.append(image_observations.observed_uv)
        counts.append(int(image_observations.camera_xyz.shape[0]))

    if not observations_xyz:
        empty_xyz = np.zeros((0, 3), dtype=np.float64)
        empty_uv = np.zeros((0, 2), dtype=np.float64)
        return empty_xyz, empty_uv, np.zeros((0,), dtype=np.int64)

    return (
        np.concatenate(observations_xyz, axis=0).astype(np.float64),
        np.concatenate(observations_uv, axis=0).astype(np.float64),
        np.asarray(counts, dtype=np.int64),
    )


class SeparableSharedIntrinsicSolver:
    def fit(
        self,
        *,
        camera_xyz: np.ndarray,
        observed_uv: np.ndarray,
        init_intrinsics: PinholeIntrinsics,
    ) -> SeparableSolveResult:
        if least_squares is None:
            raise RuntimeError("scipy is required for SeparableSharedIntrinsicSolver")
        x0 = np.array(
            [
                init_intrinsics.focal_x,
                init_intrinsics.focal_y,
                init_intrinsics.cx,
                init_intrinsics.cy,
            ],
            dtype=np.float64,
        )
        result = least_squares(
            fun=self._residual,
            x0=x0,
            args=(camera_xyz, observed_uv),
            method="trf",
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=200,
        )
        fx, fy, cx, cy = result.x
        return SeparableSolveResult(
            intrinsics=PinholeIntrinsics(focal_x=float(fx), focal_y=float(fy), cx=float(cx), cy=float(cy)),
            residual_rms=float(math.sqrt(np.mean(result.fun ** 2))),
            num_observations=int(len(observed_uv)),
        )

    @staticmethod
    def _residual(params: np.ndarray, camera_xyz: np.ndarray, observed_uv: np.ndarray) -> np.ndarray:
        fx, fy, cx, cy = params
        z = camera_xyz[:, 2]
        pred_u = fx * camera_xyz[:, 0] / z + cx
        pred_v = fy * camera_xyz[:, 1] / z + cy
        return np.concatenate([pred_u - observed_uv[:, 0], pred_v - observed_uv[:, 1]], axis=0)
