import json

import numpy as np
import pytest

from tempora.data import add_gaussian_noise, generate_circle
from tempora.experiments import evaluate_topology_pair
from tempora.metrics import (
    OptionalDependencyError,
    compare_point_clouds,
    compute_persistence_diagrams,
    persistence_distance,
    tda,
)
from tempora.viz import plot_persistence_diagram


def test_circle_has_one_dominant_h1_loop() -> None:
    dataset = generate_circle(n_steps=96, phase=0.0, seed=401)

    result = compute_persistence_diagrams(dataset.clean_states, maxdim=1)
    h1 = result.diagram(1)

    assert result.metadata["backend"] == "ripser"
    assert h1.shape[1] == 2
    assert result.dominant_lifetime(1) > 1.0
    assert result.dominant_lifetime(1) > 5.0 * result.dominant_lifetime(0)


def test_noise_changes_persistence_distance() -> None:
    dataset = generate_circle(n_steps=96, phase=0.0, seed=402)
    noisy = add_gaussian_noise(dataset, noise_std=0.12, seed=403)

    clean_result = compute_persistence_diagrams(dataset.observations, maxdim=1)
    noisy_result = compute_persistence_diagrams(noisy.observations, maxdim=1)
    distance = persistence_distance(clean_result, noisy_result, homology_dim=1)

    assert distance > 0.01


def test_input_and_latent_point_clouds_can_be_compared() -> None:
    dataset = generate_circle(n_steps=80, phase=0.0, seed=404)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    latent_points = dataset.clean_states @ rotation

    metrics = evaluate_topology_pair(dataset.clean_states, latent_points, maxdim=1)

    assert metrics["bottleneck_h0"] < 1e-6
    assert metrics["bottleneck_h1"] < 1e-6
    assert metrics["input_h1_dominant_lifetime"] > 1.0
    json.dumps(metrics)


def test_compare_point_clouds_supports_wasserstein() -> None:
    dataset = generate_circle(n_steps=48, phase=0.0, seed=405)

    metrics = compare_point_clouds(
        dataset.clean_states,
        dataset.clean_states.copy(),
        maxdim=1,
        metric="wasserstein",
    )

    assert metrics["wasserstein_h0"] == pytest.approx(0.0)
    assert metrics["wasserstein_h1"] == pytest.approx(0.0)


def test_persistence_plot_helper_returns_figure() -> None:
    dataset = generate_circle(n_steps=48, phase=0.0, seed=406)
    result = compute_persistence_diagrams(dataset.clean_states, maxdim=1)

    figure = plot_persistence_diagram(result, homology_dim=1)

    assert figure is not None


def test_missing_ripser_dependency_has_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(name: str) -> object:
        if name == "ripser":
            raise ImportError("missing ripser")
        raise AssertionError(name)

    monkeypatch.setattr(tda, "import_module", fail_import)

    with pytest.raises(OptionalDependencyError, match="ripser"):
        compute_persistence_diagrams([[0.0], [1.0]], maxdim=1)


def test_missing_persim_dependency_has_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = (np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64))
    second = (np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64))

    def fail_import(name: str) -> object:
        if name == "persim":
            raise ImportError("missing persim")
        raise AssertionError(name)

    monkeypatch.setattr(tda, "import_module", fail_import)

    with pytest.raises(OptionalDependencyError, match="persim"):
        persistence_distance(first, second, homology_dim=1)
