from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _scalar_like(value: float | Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


@dataclass(slots=True)
class PinholeIntrinsics:
    focal_x: float | Tensor
    focal_y: float | Tensor
    cx: float | Tensor
    cy: float | Tensor

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> "PinholeIntrinsics":
        assert matrix.shape == (3, 3), matrix.shape
        return cls(
            focal_x=matrix[0, 0],
            focal_y=matrix[1, 1],
            cx=matrix[0, 2],
            cy=matrix[1, 2],
        )

    def to(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "PinholeIntrinsics":
        matrix = self.matrix(device=device, dtype=dtype)
        return self.from_matrix(matrix)

    def matrix(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        device = device or self.device
        dtype = dtype or self.dtype
        matrix = torch.eye(3, device=device, dtype=dtype)
        matrix[0, 0] = _scalar_like(self.focal_x, device=device, dtype=dtype)
        matrix[1, 1] = _scalar_like(self.focal_y, device=device, dtype=dtype)
        matrix[0, 2] = _scalar_like(self.cx, device=device, dtype=dtype)
        matrix[1, 2] = _scalar_like(self.cy, device=device, dtype=dtype)
        return matrix

    @property
    def device(self) -> torch.device:
        for value in (self.focal_x, self.focal_y, self.cx, self.cy):
            if isinstance(value, Tensor):
                return value.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for value in (self.focal_x, self.focal_y, self.cx, self.cy):
            if isinstance(value, Tensor):
                return value.dtype
        return torch.float32

    def delta_pixels(self, assumed: "PinholeIntrinsics") -> Tensor:
        current = self.matrix()
        assumed_matrix = assumed.matrix(device=current.device, dtype=current.dtype)
        return torch.stack(
            [
                current[0, 2] - assumed_matrix[0, 2],
                current[1, 2] - assumed_matrix[1, 2],
            ]
        )

    def delta_theta(self, assumed: "PinholeIntrinsics") -> Tensor:
        current = self.matrix()
        assumed_matrix = assumed.matrix(device=current.device, dtype=current.dtype)
        return torch.stack(
            [
                current[0, 0] - assumed_matrix[0, 0],
                current[1, 1] - assumed_matrix[1, 1],
                current[0, 2] - assumed_matrix[0, 2],
                current[1, 2] - assumed_matrix[1, 2],
            ]
        )

    def normalized_principal_point_shift(self, assumed: "PinholeIntrinsics") -> Tensor:
        current = self.matrix()
        assumed_matrix = assumed.matrix(device=current.device, dtype=current.dtype)
        return torch.stack(
            [
                (current[0, 2] - assumed_matrix[0, 2]) / assumed_matrix[0, 0],
                (current[1, 2] - assumed_matrix[1, 2]) / assumed_matrix[1, 1],
            ]
        )

    def mismatch_matrix(self, assumed: "PinholeIntrinsics") -> Tensor:
        current = self.matrix()
        assumed_matrix = assumed.matrix(device=current.device, dtype=current.dtype)
        return torch.linalg.inv(assumed_matrix) @ current
