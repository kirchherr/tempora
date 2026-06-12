"""Projected local plasticity updates for contractive recurrent weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from tempora.models.projections import contraction_margin, project_recurrent_weights


class SupportsProjectedPlasticity(Protocol):
    """Small protocol needed for projected recurrent-weight updates."""

    @property
    def damping(self) -> torch.Tensor: ...

    @property
    def recurrent_weight(self) -> torch.Tensor: ...

    margin: float
    lipschitz: float


@dataclass(frozen=True)
class PlasticityLog:
    """Serializable diagnostics for one projected plasticity update."""

    margin_before: float
    margin_after: float
    weight_norm_before: float
    weight_norm_after: float
    update_norm: float

    def to_metrics(self) -> dict[str, float]:
        return {
            "margin_before": self.margin_before,
            "margin_after": self.margin_after,
            "weight_norm_before": self.weight_norm_before,
            "weight_norm_after": self.weight_norm_after,
            "update_norm": self.update_norm,
        }


def oja_delta(
    recurrent_weights: torch.Tensor,
    activations: torch.Tensor,
    *,
    learning_rate: float,
    stabilization: float = 1.0,
    homeostatic_decay: float = 0.0,
) -> torch.Tensor:
    """Compute an Oja/Hebb-style recurrent update.

    The update is:

    `Delta W = eta * (mean(y y^T) - alpha W) - lambda W`

    It is intentionally local and simple. Stability is not inferred from this
    rule itself; callers must project the result back into the contractive set.
    """

    if (
        recurrent_weights.ndim != 2
        or recurrent_weights.shape[0] != recurrent_weights.shape[1]
    ):
        raise ValueError("recurrent_weights must be a square matrix.")
    if activations.ndim not in {2, 3}:
        raise ValueError(
            "activations must have shape (samples, dim) or (batch, steps, dim)."
        )
    if learning_rate < 0.0:
        raise ValueError("learning_rate must be non-negative.")
    if stabilization < 0.0:
        raise ValueError("stabilization must be non-negative.")
    if homeostatic_decay < 0.0:
        raise ValueError("homeostatic_decay must be non-negative.")
    if not torch.isfinite(recurrent_weights).all():
        raise ValueError("recurrent_weights must be finite.")
    if not torch.isfinite(activations).all():
        raise ValueError("activations must be finite.")

    flattened = activations.reshape(-1, activations.shape[-1])
    if flattened.shape[-1] != recurrent_weights.shape[0]:
        raise ValueError("activation dimension must match recurrent_weights.")
    covariance = flattened.transpose(0, 1) @ flattened / float(flattened.shape[0])
    return (
        float(learning_rate) * (covariance - float(stabilization) * recurrent_weights)
        - float(homeostatic_decay) * recurrent_weights
    )


def apply_projected_oja_update_(
    model: SupportsProjectedPlasticity,
    activations: torch.Tensor,
    *,
    learning_rate: float,
    stabilization: float = 1.0,
    homeostatic_decay: float = 0.0,
) -> PlasticityLog:
    """Apply one Oja-style update and project `W` back into the contractive set."""

    with torch.no_grad():
        before_weights = model.recurrent_weight.detach().clone()
        margin_before = contraction_margin(
            model.damping,
            before_weights,
            lipschitz=model.lipschitz,
        )
        delta = oja_delta(
            before_weights,
            activations.detach(),
            learning_rate=learning_rate,
            stabilization=stabilization,
            homeostatic_decay=homeostatic_decay,
        )
        updated = before_weights + delta
        projected = project_recurrent_weights(
            updated,
            model.damping,
            margin=model.margin,
            lipschitz=model.lipschitz,
        )
        model.recurrent_weight.copy_(projected)
        margin_after = contraction_margin(
            model.damping,
            model.recurrent_weight,
            lipschitz=model.lipschitz,
        )
        return PlasticityLog(
            margin_before=float(margin_before.item()),
            margin_after=float(margin_after.item()),
            weight_norm_before=float(torch.linalg.matrix_norm(before_weights).item()),
            weight_norm_after=float(
                torch.linalg.matrix_norm(model.recurrent_weight).item()
            ),
            update_norm=float(torch.linalg.matrix_norm(delta).item()),
        )
