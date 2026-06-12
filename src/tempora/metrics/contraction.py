"""Contraction diagnostics for TEMPORA models."""

from __future__ import annotations

from typing import Protocol

import torch

from tempora.models.projections import contraction_margin


class SupportsContraction(Protocol):
    @property
    def damping(self) -> torch.Tensor: ...

    @property
    def recurrent_weight(self) -> torch.Tensor: ...

    lipschitz: float


def model_contraction_margin(model: SupportsContraction) -> torch.Tensor:
    """Return a model's sufficient contraction margin.

    The model must expose `damping`, `recurrent_weight`, and `lipschitz`
    attributes. This small protocol keeps the metric reusable without forcing a
    broad inheritance tree this early in the project.
    """

    return contraction_margin(
        damping=model.damping,
        recurrent_weights=model.recurrent_weight,
        lipschitz=model.lipschitz,
    )
