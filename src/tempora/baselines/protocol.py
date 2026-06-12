"""Shared baseline interface for fair TEMPORA comparisons."""

from __future__ import annotations

from typing import Protocol, TypedDict

from tempora.data import TemporalDataset
from tempora.data.types import FloatArray


class BaselineMetrics(TypedDict):
    """Common metric fields returned by every Phase 6 baseline."""

    prediction_mse: float
    reconstruction_mse: float
    fit_epochs: float


class TemporalModelProtocol(Protocol):
    """Minimal interface shared by TEMPORA baselines."""

    name: str

    def fit(self, dataset: TemporalDataset, seed: int) -> BaselineMetrics:
        """Fit model parameters on a finite temporal dataset."""
        ...

    def encode(self, dataset: TemporalDataset) -> FloatArray:
        """Return a finite representation for every time step."""
        ...

    def predict(self, dataset: TemporalDataset) -> FloatArray:
        """Return finite predictions with the same shape as observations."""
        ...
