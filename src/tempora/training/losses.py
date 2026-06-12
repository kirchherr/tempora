"""Loss functions for small TEMPORA training loops."""

from __future__ import annotations

import torch


def prediction_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error with finite-value and shape checks."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes.")
    if not torch.isfinite(predictions).all():
        raise ValueError("predictions must be finite.")
    if not torch.isfinite(targets).all():
        raise ValueError("targets must be finite.")
    return torch.mean((predictions - targets).square())


def next_step_prediction_loss(
    latent_states: torch.Tensor,
    observations: torch.Tensor,
) -> torch.Tensor:
    """Predict `observations[:, 1:]` from `latent_states[:, 1:]`.

    This Phase 3 helper intentionally supports the smoke setting where
    `latent_dim == observation_dim`. A learned decoder can be added later when
    the benchmark layer is introduced.
    """

    if latent_states.ndim != 3:
        raise ValueError("latent_states must have shape (batch, steps, latent_dim).")
    if observations.ndim != 3:
        raise ValueError(
            "observations must have shape (batch, steps, observation_dim)."
        )
    if latent_states.shape[0] != observations.shape[0]:
        raise ValueError(
            "latent_states and observations must have matching batch size."
        )
    if latent_states.shape[1] != observations.shape[1]:
        raise ValueError(
            "latent_states and observations must have matching step count."
        )
    if latent_states.shape[2] != observations.shape[2]:
        raise ValueError("Phase 3 smoke loss requires latent_dim == observation_dim.")
    return prediction_mse(latent_states[:, 1:, :], observations[:, 1:, :])
