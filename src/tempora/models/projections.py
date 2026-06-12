"""Projection utilities for sufficient contraction constraints."""

from __future__ import annotations

from typing import cast

import torch


def spectral_norm(matrix: torch.Tensor) -> torch.Tensor:
    """Return the matrix spectral norm.

    The spectral norm is the largest singular value and appears in TEMPORA's MVP
    sufficient contraction condition:

    `min(D) > L_sigma * ||W||_2 + margin`.
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be rank-2.")
    return cast(torch.Tensor, torch.linalg.matrix_norm(matrix, ord=2))


def contraction_margin(
    damping: torch.Tensor,
    recurrent_weights: torch.Tensor,
    *,
    lipschitz: float = 1.0,
) -> torch.Tensor:
    """Compute `min(D) - L_sigma * ||W||_2`.

    Positive values certify TEMPORA's sufficient contraction condition for the
    latent dynamics under the documented assumptions. This is not a necessary
    condition and should not be read as a broad semantic guarantee.
    """

    if damping.ndim != 1:
        raise ValueError("damping must be rank-1.")
    if recurrent_weights.ndim != 2:
        raise ValueError("recurrent_weights must be rank-2.")
    if recurrent_weights.shape[0] != recurrent_weights.shape[1]:
        raise ValueError("recurrent_weights must be square.")
    if damping.shape[0] != recurrent_weights.shape[0]:
        raise ValueError("damping and recurrent_weights dimensions must match.")
    if not torch.isfinite(damping).all():
        raise ValueError("damping must be finite.")
    if not torch.isfinite(recurrent_weights).all():
        raise ValueError("recurrent_weights must be finite.")
    if lipschitz <= 0.0:
        raise ValueError("lipschitz must be positive.")

    return torch.min(damping) - float(lipschitz) * spectral_norm(recurrent_weights)


def project_recurrent_weights(
    recurrent_weights: torch.Tensor,
    damping: torch.Tensor,
    *,
    margin: float,
    lipschitz: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project `W` into the MVP sufficient contraction set.

    If `L_sigma * ||W||_2 >= min(D) - margin`, this scales `W` so the spectral
    norm sits just inside the allowed radius. The operation is radial and keeps
    the matrix direction unchanged; it is a conservative numerical safeguard
    rather than an optimal projection in every possible norm.
    """

    if margin <= 0.0:
        raise ValueError("margin must be positive.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    if torch.any(damping <= 0.0):
        raise ValueError("damping must be strictly positive.")
    allowed_radius = torch.min(damping) - float(margin)
    if bool((allowed_radius <= 0.0).item()):
        raise ValueError("margin must be smaller than min(damping).")

    current_norm = spectral_norm(recurrent_weights)
    current_radius = float(lipschitz) * current_norm
    if bool((current_radius < allowed_radius).item()):
        return recurrent_weights.clone()

    scale = allowed_radius / (current_radius + eps)
    return recurrent_weights * scale
