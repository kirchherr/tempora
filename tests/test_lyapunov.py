import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from tempora.data import generate_circle
from tempora.experiments.evaluate_stability import (
    DatasetName,
    evaluate_dataset_stability,
    run_stability_evaluation,
)
from tempora.metrics import (
    estimate_largest_lyapunov,
    missing_segment_robustness_score,
    noise_robustness_score,
    reconstruction_mse,
    representation_distance,
    time_warp_invariance_score,
)


def test_largest_lyapunov_estimate_is_finite_for_circle() -> None:
    dataset = generate_circle(n_steps=80, phase=0.0, seed=501)

    estimate = estimate_largest_lyapunov(
        dataset.clean_states,
        times=dataset.times,
        min_separation=3,
        max_horizon=5,
    )

    assert np.isfinite(estimate.largest_exponent)
    assert estimate.pair_count > 0
    assert len(estimate.horizons) == 5
    json.dumps(estimate.to_metrics())


def test_invariance_and_robustness_scores_are_finite_and_measurable() -> None:
    dataset = generate_circle(n_steps=64, phase=0.0, seed=502)

    warp_score = time_warp_invariance_score(dataset)
    noise_score = noise_robustness_score(dataset, noise_std=0.1, seed=503)
    mask_score = missing_segment_robustness_score(dataset, fraction=0.2, seed=504)

    assert warp_score == pytest.approx(1.0)
    assert 0.0 < noise_score < 1.0
    assert 0.0 < mask_score < 1.0
    assert representation_distance(dataset.observations, dataset.observations) == 0.0


def test_reconstruction_mse_checks_shape_and_finite_values() -> None:
    target = np.zeros((4, 2), dtype=np.float64)
    reconstruction = np.ones((4, 2), dtype=np.float64)

    assert reconstruction_mse(target, reconstruction) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="identical shapes"):
        reconstruction_mse(target, np.ones((4, 3), dtype=np.float64))


def test_dataset_stability_metrics_are_json_serializable() -> None:
    dataset = generate_circle(n_steps=72, phase=0.0, seed=505)

    metrics = evaluate_dataset_stability(dataset, dataset_name="circle", seed=505)

    assert metrics["dataset"] == "circle"
    assert np.isfinite(metrics["largest_lyapunov_estimate"])
    assert metrics["time_warp_invariance_score"] == pytest.approx(1.0)
    assert 0.0 < metrics["noise_robustness_score"] <= 1.0
    json.dumps(metrics)


def test_stability_evaluation_writes_reproducible_json_and_figures(
    tmp_path: Path,
) -> None:
    datasets = cast(tuple[DatasetName, ...], ("circle", "torus"))

    first = run_stability_evaluation(
        run_id="first",
        output_root=tmp_path / "a",
        seed=600,
        dataset_names=datasets,
        n_steps=48,
    )
    second = run_stability_evaluation(
        run_id="second",
        output_root=tmp_path / "b",
        seed=600,
        dataset_names=datasets,
        n_steps=48,
    )

    assert first.metrics_path.exists()
    assert second.metrics_path.exists()
    assert (first.output_dir / "figures" / "circle_trajectory.png").exists()
    assert (first.output_dir / "figures" / "torus_trajectory.png").exists()

    first_payload = json.loads(first.metrics_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.metrics_path.read_text(encoding="utf-8"))
    first_payload["run_id"] = "same"
    second_payload["run_id"] = "same"
    first_payload["figures"] = {}
    second_payload["figures"] = {}

    assert first_payload == second_payload
