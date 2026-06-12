"""Stability and invariance evaluation for synthetic TEMPORA trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tempora.data import (
    TemporalDataset,
    generate_circle,
    generate_lorenz,
    generate_rossler,
    generate_torus,
)
from tempora.metrics import (
    estimate_largest_lyapunov,
    missing_segment_robustness_score,
    noise_robustness_score,
    reconstruction_mse,
    time_warp_invariance_score,
)

DatasetName = Literal["circle", "torus", "lorenz", "rossler"]
DEFAULT_DATASETS: tuple[DatasetName, ...] = ("circle", "torus", "lorenz", "rossler")


@dataclass(frozen=True)
class StabilityRunResult:
    """Locations and metrics for one stability evaluation run."""

    output_dir: Path
    metrics_path: Path
    metrics: dict[str, Any]


def run_stability_evaluation(
    *,
    run_id: str,
    output_root: Path,
    seed: int = 42,
    dataset_names: tuple[DatasetName, ...] = DEFAULT_DATASETS,
    n_steps: int = 96,
) -> StabilityRunResult:
    """Run a deterministic stability evaluation and write JSON/figure artifacts."""

    if not run_id:
        raise ValueError("run_id must not be empty.")
    if n_steps < 16:
        raise ValueError("n_steps must be at least 16.")

    output_dir = output_root / run_id
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_metrics: dict[str, Any] = {}
    figure_paths: dict[str, str] = {}
    for offset, dataset_name in enumerate(dataset_names):
        dataset = make_dataset(dataset_name, n_steps=n_steps, seed=seed + offset)
        dataset_metrics[dataset_name] = evaluate_dataset_stability(
            dataset,
            dataset_name=dataset_name,
            seed=seed + offset,
        )
        figure_path = figures_dir / f"{dataset_name}_trajectory.png"
        save_trajectory_figure(dataset, figure_path)
        figure_paths[dataset_name] = str(figure_path)

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "datasets": dataset_metrics,
        "figures": figure_paths,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return StabilityRunResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        metrics=metrics,
    )


def evaluate_dataset_stability(
    dataset: TemporalDataset,
    *,
    dataset_name: str,
    seed: int,
) -> dict[str, Any]:
    """Compute finite stability and perturbation metrics for one dataset."""

    lyapunov = estimate_largest_lyapunov(
        dataset.observations,
        times=dataset.times,
        min_separation=3,
        max_horizon=min(6, max(1, dataset.n_steps // 8)),
    )
    metrics: dict[str, Any] = {
        "dataset": dataset_name,
        "seed": seed,
        "n_steps": dataset.n_steps,
        "observation_dim": dataset.observation_dim,
        "reconstruction_mse": reconstruction_mse(
            dataset.clean_states,
            dataset.observations,
        ),
        "time_warp_invariance_score": time_warp_invariance_score(dataset),
        "noise_robustness_score": noise_robustness_score(
            dataset,
            noise_std=0.05,
            seed=seed + 10_000,
        ),
        "missing_segment_robustness_score": missing_segment_robustness_score(
            dataset,
            fraction=0.1,
            seed=seed + 20_000,
        ),
    }
    metrics.update(lyapunov.to_metrics())
    _validate_metrics_are_finite(metrics)
    return metrics


def make_dataset(
    dataset_name: DatasetName, *, n_steps: int, seed: int
) -> TemporalDataset:
    """Create one synthetic dataset with deterministic Phase-5 defaults."""

    if dataset_name == "circle":
        return generate_circle(
            n_steps=n_steps, duration=2.0 * np.pi, phase=0.0, seed=seed
        )
    if dataset_name == "torus":
        return generate_torus(n_steps=n_steps, duration=2.0 * np.pi, seed=seed)
    if dataset_name == "lorenz":
        return generate_lorenz(n_steps=n_steps, duration=4.0, seed=seed)
    if dataset_name == "rossler":
        return generate_rossler(n_steps=n_steps, duration=8.0, seed=seed)
    raise ValueError(f"unsupported dataset: {dataset_name}")


def save_trajectory_figure(dataset: TemporalDataset, path: Path) -> None:
    """Save a small trajectory figure without returning generated artifacts."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(5, 4))
    if dataset.observation_dim >= 3:
        axis: Any = figure.add_subplot(111, projection="3d")
        axis.plot(
            dataset.observations[:, 0],
            dataset.observations[:, 1],
            dataset.observations[:, 2],
            linewidth=1,
        )
        axis.set_zlabel("z")
    else:
        axis = figure.add_subplot(111)
        axis.plot(dataset.observations[:, 0], dataset.observations[:, 1], linewidth=1)
    axis.set_title(str(dataset.metadata.get("system", "trajectory")))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _validate_metrics_are_finite(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, int | str | list):
            continue
        if isinstance(value, float) and np.isfinite(value):
            continue
        raise RuntimeError(f"metric {key!r} is not finite: {value!r}")
