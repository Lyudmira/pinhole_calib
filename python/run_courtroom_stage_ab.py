from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import torch

from pinhole_calib import (
    ColmapImageCorrespondences,
    ColmapImageObservations,
    ColmapSharedIntrinsicEstimator,
    GatingConfig,
    MiscalibrationPriorAdapter,
    PinholeIntrinsics,
    SeparableSharedIntrinsicSolver,
    load_colmap_correspondences_by_image,
    load_colmap_tracks,
    load_colmap_tracks_by_image,
)
from pinhole_calib.image_ops import (
    recenter_image_from_estimated_intrinsics,
    shift_image_principal_point,
)


MOGE_REPO = Path("/data/users/mia/current/MoGe")
COURTROOM = Path("/data/users/mia/current/instantsplat_data/TanksAndTemples_advanced/Courtroom")
COLMAP_BINARY = Path("/data/users/mia/conda_envs/colmap410build/bin/colmap")


@dataclass
class FrameMetrics:
    image_name: str
    correspondence_residual: float
    correspondence_count: int
    shifted_point_rmse: float
    corrected_point_rmse: float
    rerun_point_rmse: float
    shifted_depth_mae: float
    corrected_depth_mae: float
    rerun_depth_mae: float
    shifted_normal_deg: float
    corrected_normal_deg: float
    rerun_normal_deg: float
    gate: float


def import_moge():
    import sys

    repo = str(MOGE_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from moge.model.v2 import MoGeModel

    return MoGeModel


def to_tensor_rgb(image_bgr: np.ndarray, device: str) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)


@torch.inference_mode()
def run_moge(model, image_bgr: np.ndarray, *, device: str, use_fp16: bool) -> dict[str, torch.Tensor]:
    tensor = to_tensor_rgb(image_bgr, device)
    output = model.infer(tensor, use_fp16=use_fp16)
    return {k: v.detach().float().cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()}


def normalized_intrinsics_to_pixels(intrinsics: torch.Tensor, *, width: int, height: int) -> PinholeIntrinsics:
    return PinholeIntrinsics(
        focal_x=float(intrinsics[0, 0] * width),
        focal_y=float(intrinsics[1, 1] * height),
        cx=float(intrinsics[0, 2] * width),
        cy=float(intrinsics[1, 2] * height),
    )


def centered_intrinsics_like(intrinsics: PinholeIntrinsics, *, width: int, height: int) -> PinholeIntrinsics:
    return PinholeIntrinsics(
        focal_x=intrinsics.focal_x,
        focal_y=intrinsics.focal_y,
        cx=width / 2.0,
        cy=height / 2.0,
    )


def masked_point_rmse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    diff = a[mask] - b[mask]
    return float(torch.sqrt(torch.mean(torch.sum(diff * diff, dim=-1))).item())


def masked_depth_mae(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (a[mask] - b[mask]).abs()
    return float(diff.mean().item())


def masked_normal_angle_deg(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    cos = (a[mask] * b[mask]).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.rad2deg(torch.acos(cos))
    return float(ang.mean().item())


def pick_images(image_dir: Path, *, num_images: int, seed: int) -> list[Path]:
    images = sorted(image_dir.glob("*.jpg"))
    rng = random.Random(seed)
    if num_images >= len(images):
        return images
    start = rng.randint(0, len(images) - num_images)
    return images[start : start + num_images]


_LEGACY_RUN_SUBDIRS = ("base_colmap", "base_images", "shifted_images")


def _legacy_dirs_present(work_root: Path) -> bool:
    return any((work_root / name).exists() for name in _LEGACY_RUN_SUBDIRS)


def _max_numeric_subdir(work_root: Path) -> int:
    m = 0
    for p in work_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            m = max(m, int(p.name))
    return m


def migrate_legacy_flat_work_root(work_root: Path) -> None:
    """Move a pre-numbering layout (base_* at work_root) into a numbered folder.

    If ``work_root/1`` already exists, orphan legacy dirs go to ``max_index + 1``.
    """
    if not _legacy_dirs_present(work_root):
        return
    dest = work_root / "1"
    if dest.exists():
        dest = work_root / str(_max_numeric_subdir(work_root) + 1)
    dest.mkdir(parents=True, exist_ok=True)
    for name in _LEGACY_RUN_SUBDIRS:
        src = work_root / name
        if src.exists():
            shutil.move(str(src), str(dest / name))


def allocate_next_run_dir(work_root: Path) -> Path:
    """Return ``work_root / N`` with ``N`` one past the highest existing numeric subdirectory."""
    work_root.mkdir(parents=True, exist_ok=True)
    migrate_legacy_flat_work_root(work_root)
    n = _max_numeric_subdir(work_root) + 1
    run_dir = work_root / str(n)
    run_dir.mkdir(parents=False)
    return run_dir


def allocate_output_path(output_path: Path, *, run_index: int) -> Path:
    """Place outputs under ``output_root / run_index / filename`` to avoid overwrites."""
    output_root = output_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / str(run_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / output_path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Ruicheng/moge-2-vits-normal")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--max-shift-frac", type=float, default=0.06)
    parser.add_argument("--output", type=Path, default=Path("/data/users/mia/current/pinhole_calib/python/output/courtroom_stage_ab.json"))
    parser.add_argument("--work-root", type=Path, default=Path("/data/users/mia/current/pinhole_calib/stage_ab"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    MoGeModel = import_moge()
    model = MoGeModel.from_pretrained(args.model).to(args.device).eval()

    image_paths = pick_images(COURTROOM, num_images=args.num_images, seed=args.seed)
    sample = cv2.imread(str(image_paths[0]))
    height, width = sample.shape[:2]
    delta_cx = random.uniform(-args.max_shift_frac * width, args.max_shift_frac * width)
    delta_cy = random.uniform(-args.max_shift_frac * height, args.max_shift_frac * height)
    min_model_size = max(3, min(8, args.num_images - 1))

    work_root = args.work_root
    run_dir = allocate_next_run_dir(work_root)
    output_path = allocate_output_path(args.output, run_index=int(run_dir.name))
    base_image_dir = run_dir / "base_images"
    shifted_image_dir = run_dir / "shifted_images"
    base_image_dir.mkdir(parents=True, exist_ok=True)
    shifted_image_dir.mkdir(parents=True, exist_ok=True)

    baseline_outputs = []
    shifted_outputs = []
    shifted_images = []
    recentered_images = []

    for path in image_paths:
        image = cv2.imread(str(path))
        cv2.imwrite(str(base_image_dir / path.name), image)
        baseline_outputs.append(run_moge(model, image, device=args.device, use_fp16=args.use_fp16))
        shifted = shift_image_principal_point(image, delta_cx=delta_cx, delta_cy=delta_cy)
        cv2.imwrite(str(shifted_image_dir / path.name), shifted)
        shifted_images.append(shifted)
        shifted_outputs.append(run_moge(model, shifted, device=args.device, use_fp16=args.use_fp16))

    colmap = ColmapSharedIntrinsicEstimator(colmap_binary=COLMAP_BINARY)
    base_estimate = colmap.fit(
        image_dir=base_image_dir,
        work_dir=run_dir / "base_colmap",
        min_model_size=min_model_size,
    )
    base_shared = base_estimate.intrinsics
    colmap_txt_dir = run_dir / "base_colmap" / "txt"
    track_observations = load_colmap_tracks_by_image(colmap_txt_dir)
    image_correspondences = load_colmap_correspondences_by_image(colmap_txt_dir)
    observations_by_id = {obs.image_id: obs for obs in track_observations.values()}
    camera_xyz, observed_uv, _ = load_colmap_tracks(colmap_txt_dir)
    shifted_uv = observed_uv.copy()
    shifted_uv[:, 0] += delta_cx
    shifted_uv[:, 1] += delta_cy
    valid = (
        (shifted_uv[:, 0] >= 0.0)
        & (shifted_uv[:, 0] < width)
        & (shifted_uv[:, 1] >= 0.0)
        & (shifted_uv[:, 1] < height)
    )
    separable = SeparableSharedIntrinsicSolver().fit(
        camera_xyz=camera_xyz[valid],
        observed_uv=shifted_uv[valid],
        init_intrinsics=base_shared,
    )
    estimated = separable.intrinsics
    true_shifted = PinholeIntrinsics(
        focal_x=base_shared.focal_x,
        focal_y=base_shared.focal_y,
        cx=base_shared.cx + delta_cx,
        cy=base_shared.cy + delta_cy,
    )

    adapter = MiscalibrationPriorAdapter(assumed_intrinsics=base_shared)

    rerun_outputs = []
    for shifted in shifted_images:
        recentered = recenter_image_from_estimated_intrinsics(
            shifted,
            estimated_cx=estimated.cx,
            estimated_cy=estimated.cy,
            target_cx=base_shared.cx,
            target_cy=base_shared.cy,
            estimated_focal_x=estimated.focal_x,
            estimated_focal_y=estimated.focal_y,
            target_focal_x=base_shared.focal_x,
            target_focal_y=base_shared.focal_y,
        )
        recentered_images.append(recentered)
        rerun_outputs.append(run_moge(model, recentered, device=args.device, use_fp16=args.use_fp16))

    frame_metrics: list[FrameMetrics] = []
    for path, baseline, shifted_out, rerun_out in zip(image_paths, baseline_outputs, shifted_outputs, rerun_outputs):
        image_tracks = track_observations.get(path.name)
        if image_tracks is None:
            image_tracks = ColmapImageObservations(
                image_id=-1,
                image_name=path.name,
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros((3,), dtype=np.float64),
                camera_xyz=np.zeros((0, 3), dtype=np.float64),
                observed_uv=np.zeros((0, 2), dtype=np.float64),
                point_ids=np.zeros((0,), dtype=np.int64),
            )
        image_corr = image_correspondences.get(path.name)
        if image_corr is None:
            image_corr = ColmapImageCorrespondences(
                image_id=image_tracks.image_id,
                image_name=path.name,
                source_uv=np.zeros((0, 2), dtype=np.float64),
                target_image_ids=np.zeros((0,), dtype=np.int64),
                target_uv=np.zeros((0, 2), dtype=np.float64),
                point_ids=np.zeros((0,), dtype=np.int64),
            )
        corrected = adapter.correct_priors(
            current_intrinsics=estimated,
            depth=shifted_out["depth"],
            normals=shifted_out["normal"],
            pointmap=shifted_out["points"],
        )
        mask = (baseline["mask"] > 0) & (shifted_out["mask"] > 0) & (rerun_out["mask"] > 0)
        shifted_source_uv = image_corr.source_uv.copy()
        if shifted_source_uv.size > 0:
            shifted_source_uv[:, 0] += delta_cx
            shifted_source_uv[:, 1] += delta_cy
        shifted_observed_uv = image_tracks.observed_uv.copy()
        if shifted_observed_uv.size > 0:
            shifted_observed_uv[:, 0] += delta_cx
            shifted_observed_uv[:, 1] += delta_cy
        source_uv = []
        target_rotations = []
        target_translations = []
        target_uv = []
        for observed_source_uv, target_image_id, observed_target_uv in zip(
            shifted_source_uv,
            image_corr.target_image_ids,
            image_corr.target_uv,
        ):
            target_image = observations_by_id.get(int(target_image_id))
            if target_image is None:
                continue
            source_uv.append(observed_source_uv)
            target_rotations.append(target_image.rotation)
            target_translations.append(target_image.translation)
            target_uv.append(observed_target_uv)
        if target_rotations:
            geometry_residual = adapter.pointmap_multiview_correspondence_residual(
                corrected.pointmap,
                source_uv=np.asarray(source_uv, dtype=np.float64),
                target_uv=np.asarray(target_uv, dtype=np.float64),
                source_rotation=image_tracks.rotation,
                source_translation=image_tracks.translation,
                target_rotations=np.asarray(target_rotations, dtype=np.float64),
                target_translations=np.asarray(target_translations, dtype=np.float64),
                target_intrinsics=base_shared,
                mask=shifted_out["mask"] > 0,
            )
            correspondence_count = len(target_rotations)
        else:
            geometry_residual = adapter.pointmap_correspondence_residual(
                corrected.pointmap,
                observed_uv=shifted_observed_uv,
                camera_xyz=image_tracks.camera_xyz,
                mask=shifted_out["mask"] > 0,
            )
            correspondence_count = int(image_tracks.camera_xyz.shape[0])
        signals = adapter.build_default_signals(
            current_intrinsics=estimated,
            geometry_residual=geometry_residual,
        )
        gate = adapter.gate(signals=signals, config=GatingConfig(tau=2.0))
        frame_metrics.append(
            FrameMetrics(
                image_name=path.name,
                correspondence_residual=float(geometry_residual.item()),
                correspondence_count=correspondence_count,
                shifted_point_rmse=masked_point_rmse(shifted_out["points"], baseline["points"], mask),
                corrected_point_rmse=masked_point_rmse(corrected.pointmap, baseline["points"], mask),
                rerun_point_rmse=masked_point_rmse(rerun_out["points"], baseline["points"], mask),
                shifted_depth_mae=masked_depth_mae(shifted_out["depth"], baseline["depth"], mask),
                corrected_depth_mae=masked_depth_mae(corrected.depth, baseline["depth"], mask),
                rerun_depth_mae=masked_depth_mae(rerun_out["depth"], baseline["depth"], mask),
                shifted_normal_deg=masked_normal_angle_deg(shifted_out["normal"], baseline["normal"], mask),
                corrected_normal_deg=masked_normal_angle_deg(corrected.normals, baseline["normal"], mask),
                rerun_normal_deg=masked_normal_angle_deg(rerun_out["normal"], baseline["normal"], mask),
                gate=float(gate.item()),
            )
        )

    summary = {
        "work_root": str(work_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        "run_index": int(run_dir.name),
        "output_path": str(output_path.resolve()),
        "image_paths": [str(p) for p in image_paths],
        "seed": args.seed,
        "delta_cx": delta_cx,
        "delta_cy": delta_cy,
        "base_shared_intrinsics": asdict(base_shared),
        "true_shifted_intrinsics": asdict(true_shifted),
        "estimated_shifted_intrinsics": asdict(estimated),
        "base_colmap_camera": asdict(base_estimate.intrinsics),
        "stage_a_solver": {
            "type": "base_colmap_plus_separable_shared_intrinsics",
            "num_observations": separable.num_observations,
            "residual_rms": separable.residual_rms,
        },
        "estimation_error": {
            "focal_x": estimated.focal_x - true_shifted.focal_x,
            "focal_y": estimated.focal_y - true_shifted.focal_y,
            "cx": estimated.cx - true_shifted.cx,
            "cy": estimated.cy - true_shifted.cy,
            "delta_cx_error_vs_base": (estimated.cx - base_shared.cx) - delta_cx,
            "delta_cy_error_vs_base": (estimated.cy - base_shared.cy) - delta_cy,
        },
        "frame_metrics": [asdict(m) for m in frame_metrics],
        "aggregate_metrics": {
            key: float(np.mean([getattr(m, key) for m in frame_metrics]))
            for key in (
                "shifted_point_rmse",
                "corrected_point_rmse",
                "rerun_point_rmse",
                "correspondence_residual",
                "shifted_depth_mae",
                "corrected_depth_mae",
                "rerun_depth_mae",
                "shifted_normal_deg",
                "corrected_normal_deg",
                "rerun_normal_deg",
                "gate",
            )
        },
    }

    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate_metrics"], indent=2))
    print(json.dumps(summary["estimation_error"], indent=2))
    print(f"run artifacts: {run_dir}")
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
