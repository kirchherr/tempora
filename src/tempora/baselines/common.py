"""Common utilities for small deterministic baselines."""

from __future__ import annotations

import random

import numpy as np
import torch

from tempora.baselines.protocol import BaselineMetrics
from tempora.data import TemporalDataset
from tempora.data.types import FloatArray
from tempora.metrics import reconstruction_mse


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for small baseline runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def observations_tensor(dataset: TemporalDataset) -> torch.Tensor:
    """Return observations as a `(1, steps, dim)` float32 tensor."""

    return torch.as_tensor(dataset.observations, dtype=torch.float32).unsqueeze(0)


def ensure_fitted(value: object | None, *, name: str) -> object:
    if value is None:
        raise RuntimeError(f"{name} must be fit before calling encode or predict.")
    return value


def validate_prediction(dataset: TemporalDataset, prediction: FloatArray) -> None:
    if prediction.shape != dataset.observations.shape:
        raise ValueError("prediction shape must match dataset observations.")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction must contain only finite values.")


def baseline_metrics(
    dataset: TemporalDataset,
    prediction: FloatArray,
    *,
    fit_epochs: int,
) -> BaselineMetrics:
    """Compute common Phase 6 metric fields for a baseline prediction."""

    validate_prediction(dataset, prediction)
    if dataset.n_steps < 2:
        raise ValueError("dataset must have at least two time steps.")
    return {
        "prediction_mse": reconstruction_mse(
            dataset.observations[1:],
            prediction[1:],
        ),
        "reconstruction_mse": reconstruction_mse(
            dataset.observations,
            prediction,
        ),
        "fit_epochs": float(fit_epochs),
    }
