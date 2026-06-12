import json

import numpy as np

from tempora.baselines import (
    GRUBaseline,
    ReservoirBaseline,
    TemporalModelProtocol,
    VanillaNeuralODEBaseline,
)
from tempora.data import generate_circle
from tempora.experiments import compare_baselines


def test_all_baselines_share_fit_encode_predict_interface() -> None:
    dataset = generate_circle(n_steps=28, duration=1.0, phase=0.0, seed=701)
    baselines: list[TemporalModelProtocol] = [
        GRUBaseline(hidden_dim=5, epochs=4, learning_rate=0.02),
        VanillaNeuralODEBaseline(epochs=4, learning_rate=0.02),
        ReservoirBaseline(reservoir_dim=6),
    ]

    for baseline in baselines:
        metrics = baseline.fit(dataset, seed=702)
        encoded = baseline.encode(dataset)
        prediction = baseline.predict(dataset)

        assert set(metrics) == {"prediction_mse", "reconstruction_mse", "fit_epochs"}
        assert encoded.shape[0] == dataset.n_steps
        assert prediction.shape == dataset.observations.shape
        assert np.all(np.isfinite(encoded))
        assert np.all(np.isfinite(prediction))
        assert np.isfinite(metrics["prediction_mse"])
        assert np.isfinite(metrics["reconstruction_mse"])


def test_baseline_comparison_is_reproducible_for_same_seed() -> None:
    dataset = generate_circle(n_steps=24, duration=1.0, phase=0.0, seed=711)

    first = compare_baselines(dataset, seed=712)
    second = compare_baselines(dataset, seed=712)

    assert first == second
    assert first["metric_fields"] == [
        "prediction_mse",
        "reconstruction_mse",
        "fit_epochs",
    ]
    assert set(first["models"]) == {
        "tempora_contractivectrnn",
        "gru",
        "vanilla_neural_ode",
        "reservoir",
    }
    json.dumps(first)


def test_reservoir_baseline_is_seed_dependent_but_reproducible() -> None:
    dataset = generate_circle(n_steps=30, duration=1.0, phase=0.0, seed=721)
    first = ReservoirBaseline(reservoir_dim=8)
    second = ReservoirBaseline(reservoir_dim=8)
    other = ReservoirBaseline(reservoir_dim=8)

    first_metrics = first.fit(dataset, seed=722)
    second_metrics = second.fit(dataset, seed=722)
    other_metrics = other.fit(dataset, seed=723)

    assert first_metrics == second_metrics
    np.testing.assert_allclose(first.predict(dataset), second.predict(dataset))
    assert first_metrics != other_metrics
