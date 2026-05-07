from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

from .intrinsics import PinholeIntrinsics


@dataclass(slots=True)
class ColmapSharedIntrinsicEstimate:
    intrinsics: PinholeIntrinsics
    model_dir: Path
    cameras_txt: Path


class ColmapSharedIntrinsicEstimator:
    def __init__(self, *, colmap_binary: str | Path) -> None:
        self.colmap_binary = str(colmap_binary)

    def fit(
        self,
        *,
        image_dir: Path,
        work_dir: Path,
        use_gpu: bool = True,
        min_model_size: int = 8,
    ) -> ColmapSharedIntrinsicEstimate:
        work_dir.mkdir(parents=True, exist_ok=True)
        db = work_dir / "database.db"
        sparse = work_dir / "sparse"
        ba = work_dir / "ba"
        txt = work_dir / "txt"
        for path in (sparse, ba, txt):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        if db.exists():
            db.unlink()

        self._run(["database_creator", "--database_path", str(db)])
        self._run(
            [
                "feature_extractor",
                "--database_path",
                str(db),
                "--image_path",
                str(image_dir),
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_model",
                "PINHOLE",
                "--FeatureExtraction.use_gpu",
                "1" if use_gpu else "0",
            ]
        )
        self._run(
            [
                "exhaustive_matcher",
                "--database_path",
                str(db),
                "--FeatureMatching.use_gpu",
                "1" if use_gpu else "0",
            ]
        )
        self._run(
            [
                "mapper",
                "--database_path",
                str(db),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(sparse),
                "--Mapper.ba_refine_principal_point",
                "1",
                "--Mapper.multiple_models",
                "0",
                "--Mapper.min_model_size",
                str(min_model_size),
            ]
        )
        input_model = sparse / "0"
        self._run(
            [
                "bundle_adjuster",
                "--input_path",
                str(input_model),
                "--output_path",
                str(ba),
                "--BundleAdjustment.refine_principal_point",
                "1",
            ]
        )
        self._run(
            [
                "model_converter",
                "--input_path",
                str(ba),
                "--output_path",
                str(txt),
                "--output_type",
                "TXT",
            ]
        )
        intrinsics = self._parse_cameras_txt(txt / "cameras.txt")
        return ColmapSharedIntrinsicEstimate(
            intrinsics=intrinsics,
            model_dir=ba,
            cameras_txt=txt / "cameras.txt",
        )

    def _run(self, args: list[str]) -> None:
        cmd = [self.colmap_binary, *args]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    @staticmethod
    def _parse_cameras_txt(path: Path) -> PinholeIntrinsics:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[1] != "PINHOLE":
                raise ValueError(f"Expected PINHOLE camera, got: {parts[1]}")
            fx, fy, cx, cy = map(float, parts[4:8])
            return PinholeIntrinsics(focal_x=fx, focal_y=fy, cx=cx, cy=cy)
        raise ValueError(f"No camera entries found in {path}")
