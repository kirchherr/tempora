"""Fair small-model comparison helpers for TEMPORA baselines."""

from __future__ import annotations

from typing import Any, cast

from tempora.baselines import (
    GRUBaseline,
    ReservoirBaseline,
    TemporalModelProtocol,
    VanillaNeuralODEBaseline,
)
from tempora.baselines.common import set_reproducible_seed
from tempora.data import TemporalDataset
from tempora.models import ContractiveCTRNN
from tempora.training import train_circle_next_step

COMMON_METRIC_FIELDS = ("prediction_mse", "reconstruction_mse", "fit_epochs")


def default_baselines() -> list[TemporalModelProtocol]:
    """Return small deterministic baseline instances for Phase 6 tests."""

    return [
        GRUBaseline(hidden_dim=8, epochs=20, learning_rate=0.03),
        VanillaNeuralODEBaseline(epochs=20, learning_rate=0.03),
        ReservoirBaseline(reservoir_dim=12),
    ]


def compare_baselines(
    dataset: TemporalDataset,
    *,
    seed: int = 42,
    baselines: list[TemporalModelProtocol] | None = None,
    include_tempora_smoke: bool = True,
) -> dict[str, Any]:
    """Fit TEMPORA smoke model and baselines with shared metric fields."""

    resolved_baselines = default_baselines() if baselines is None else baselines
    results: dict[str, dict[str, float | str]] = {}

    if include_tempora_smoke:
        results["tempora_contractivectrnn"] = _fit_tempora_smoke(dataset, seed=seed)

    for offset, baseline in enumerate(resolved_baselines):
        metrics = baseline.fit(dataset, seed + offset + 1)
        results[baseline.name] = {
            "model": baseline.name,
            "prediction_mse": metrics["prediction_mse"],
            "reconstruction_mse": metrics["reconstruction_mse"],
            "fit_epochs": metrics["fit_epochs"],
        }

    return {
        "seed": seed,
        "dataset_system": str(dataset.metadata.get("system", "unknown")),
        "metric_fields": list(COMMON_METRIC_FIELDS),
        "models": results,
    }


def _fit_tempora_smoke(
    dataset: TemporalDataset, *, seed: int
) -> dict[str, float | str]:
    set_reproducible_seed(seed)
    model = ContractiveCTRNN(
        input_dim=dataset.observation_dim,
        latent_dim=dataset.observation_dim,
        damping_init=1.4,
        margin=0.1,
        recurrent_scale=0.02,
    )
    result = train_circle_next_step(
        model,
        dataset,
        epochs=20,
        learning_rate=0.03,
        plasticity_learning_rate=5e-4,
    )
    final_loss = float(cast(float, result.metrics["final_loss"]))
    return {
        "model": "tempora_contractivectrnn",
        "prediction_mse": final_loss,
        "reconstruction_mse": final_loss,
        "fit_epochs": 20.0,
    }
