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
    ColmapSharedIntrinsicEstimator,
    GatingConfig,
    MiscalibrationPriorAdapter,
    PinholeIntrinsics,
    SeparableSharedIntrinsicSolver,
    load_colmap_tracks,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Ruicheng/moge-2-vits-normal")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--max-shift-frac", type=float, default=0.06)
    parser.add_argument("--output", type=Path, default=Path("/data/users/mia/current/pinhole_calib/python/output/courtroom_stage_ab.json"))
    parser.add_argument("--work-root", type=Path, default=Path("/data/users/mia/current/pinhole_calib/python/tmp/stage_ab"))
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
    camera_xyz, observed_uv, _ = load_colmap_tracks((run_dir / "base_colmap" / "txt"))
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
        )
        recentered_images.append(recentered)
        rerun_outputs.append(run_moge(model, recentered, device=args.device, use_fp16=args.use_fp16))

    frame_metrics: list[FrameMetrics] = []
    for path, baseline, shifted_out, rerun_out in zip(image_paths, baseline_outputs, shifted_outputs, rerun_outputs):
        corrected = adapter.correct_priors(
            current_intrinsics=estimated,
            depth=shifted_out["depth"],
            normals=shifted_out["normal"],
            pointmap=shifted_out["points"],
        )
        mask = (baseline["mask"] > 0) & (shifted_out["mask"] > 0) & (rerun_out["mask"] > 0)
        signals = adapter.build_default_signals(
            current_intrinsics=estimated,
            geometry_residual=separable.residual_rms,
            bias_alignment=0.0,
        )
        gate = adapter.gate(signals=signals, config=GatingConfig(tau=2.0))
        frame_metrics.append(
            FrameMetrics(
                image_name=path.name,
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate_metrics"], indent=2))
    print(json.dumps(summary["estimation_error"], indent=2))
    print(f"run artifacts: {run_dir}")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
