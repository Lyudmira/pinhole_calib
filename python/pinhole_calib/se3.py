from __future__ import annotations

import torch
from torch import Tensor


def _skew(vector: Tensor) -> Tensor:
    assert vector.shape[-1] == 3, vector.shape
    zeros = torch.zeros_like(vector[..., 0])
    x, y, z = vector.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(omega: Tensor) -> Tensor:
    assert omega.shape[-1] == 3, omega.shape
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta_sq = theta * theta
    skew = _skew(omega)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(skew.shape)

    sin_over_theta = torch.where(
        theta > 1e-8,
        torch.sin(theta) / theta,
        1 - theta_sq / 6,
    )
    one_minus_cos_over_theta_sq = torch.where(
        theta > 1e-8,
        (1 - torch.cos(theta)) / theta_sq,
        0.5 - theta_sq / 24,
    )
    return eye + sin_over_theta[..., None] * skew + one_minus_cos_over_theta_sq[
        ..., None
    ] * (skew @ skew)


def se3_exp(xi: Tensor) -> Tensor:
    assert xi.shape[-1] == 6, xi.shape
    translation = xi[..., :3]
    omega = xi[..., 3:]
    rotation = so3_exp(omega)
    transform = torch.eye(4, device=xi.device, dtype=xi.dtype).expand(
        xi.shape[:-1] + (4, 4)
    ).clone()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    return transform
