"""End-to-end synthetic smoke benchmark for TEMPORA."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from tempora.baselines import (
    GRUBaseline,
    ReservoirBaseline,
    TemporalModelProtocol,
    VanillaNeuralODEBaseline,
)
from tempora.baselines.common import set_reproducible_seed
from tempora.data import TemporalDataset
from tempora.data.types import FloatArray
from tempora.experiments.compare_baselines import compare_baselines
from tempora.experiments.evaluate_stability import (
    DatasetName,
    make_dataset,
    save_trajectory_figure,
)
from tempora.experiments.evaluate_topology import evaluate_topology_pair
from tempora.metrics import (
    compute_persistence_diagrams,
    estimate_largest_lyapunov,
    missing_segment_robustness_score,
    noise_robustness_score,
    reconstruction_mse,
    time_warp_invariance_score,
)
from tempora.models import ContractiveCTRNN
from tempora.training import train_circle_next_step
from tempora.viz import plot_persistence_diagram


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    """Configuration for TEMPORA's CI-safe synthetic benchmark."""

    run_id: str
    seed: int
    output_root: Path
    datasets: tuple[DatasetName, ...]
    n_steps: int
    epochs: int
    learning_rate: float
    plasticity_learning_rate: float
    baseline_epochs: int

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["datasets"] = list(self.datasets)
        return payload


@dataclass(frozen=True)
class SyntheticBenchmarkResult:
    """Locations and metrics for one synthetic benchmark run."""

    output_dir: Path
    config_path: Path
    metrics_path: Path
    report_path: Path
    checkpoint_paths: tuple[Path, ...]
    metrics: dict[str, Any]


def load_benchmark_config(path: Path) -> SyntheticBenchmarkConfig:
    """Load a YAML benchmark config from disk."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark config must be a mapping.")
    datasets = tuple(cast(DatasetName, item) for item in raw.get("datasets", []))
    if not datasets:
        raise ValueError("benchmark config must list at least one dataset.")
    return SyntheticBenchmarkConfig(
        run_id=str(raw.get("run_id", "benchmark_smoke")),
        seed=int(raw.get("seed", 42)),
        output_root=Path(str(raw.get("output_root", "outputs"))),
        datasets=datasets,
        n_steps=int(raw.get("n_steps", 32)),
        epochs=int(raw.get("epochs", 6)),
        learning_rate=float(raw.get("learning_rate", 0.03)),
        plasticity_learning_rate=float(raw.get("plasticity_learning_rate", 5e-4)),
        baseline_epochs=int(raw.get("baseline_epochs", 4)),
    )


def run_synthetic_benchmark(
    config: SyntheticBenchmarkConfig,
) -> SyntheticBenchmarkResult:
    """Run TEMPORA's synthetic smoke benchmark and write artifacts."""

    if not config.run_id:
        raise ValueError("run_id must not be empty.")
    if config.n_steps < 16:
        raise ValueError("n_steps must be at least 16.")
    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    if config.baseline_epochs < 1:
        raise ValueError("baseline_epochs must be positive.")

    started_at = time.perf_counter()
    output_dir = config.output_root / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    write_benchmark_config(config, config_path)

    dataset_results: dict[str, Any] = {}
    for offset, dataset_name in enumerate(config.datasets):
        dataset_seed = config.seed + offset
        dataset = make_dataset(dataset_name, n_steps=config.n_steps, seed=dataset_seed)
        dataset_results[dataset_name] = run_dataset_benchmark(
            dataset,
            dataset_name=dataset_name,
            seed=dataset_seed,
            config=config,
            figures_dir=figures_dir,
            checkpoints_dir=checkpoints_dir,
        )
    checkpoint_paths = tuple(
        Path(cast(str, dataset_result["checkpoint"]))
        for dataset_result in dataset_results.values()
    )

    metrics: dict[str, Any] = {
        "run_id": config.run_id,
        "seed": config.seed,
        "config": config.to_jsonable(),
        "artifacts": {
            "config": str(config_path),
            "metrics": str(metrics_path),
            "report": str(report_path),
            "checkpoints": [str(path) for path in checkpoint_paths],
        },
        "git_commit": current_git_commit(),
        "dependency_versions": dependency_versions(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": time.perf_counter() - started_at,
        },
        "datasets": dataset_results,
    }
    validate_json_metrics(metrics)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path.write_text(render_benchmark_report(metrics), encoding="utf-8")
    return SyntheticBenchmarkResult(
        output_dir=output_dir,
        config_path=config_path,
        metrics_path=metrics_path,
        report_path=report_path,
        checkpoint_paths=checkpoint_paths,
        metrics=metrics,
    )


def run_dataset_benchmark(
    dataset: TemporalDataset,
    *,
    dataset_name: DatasetName,
    seed: int,
    config: SyntheticBenchmarkConfig,
    figures_dir: Path,
    checkpoints_dir: Path,
) -> dict[str, Any]:
    """Train TEMPORA smoke model and baselines on one synthetic dataset."""

    set_reproducible_seed(seed)
    model = ContractiveCTRNN(
        input_dim=dataset.observation_dim,
        latent_dim=dataset.observation_dim,
        damping_init=1.4,
        margin=0.1,
        recurrent_scale=0.02,
    )
    training = train_circle_next_step(
        model,
        dataset,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        plasticity_learning_rate=config.plasticity_learning_rate,
    )
    latent = encode_dataset(model, dataset)
    topology = evaluate_topology_pair(dataset.observations, latent, maxdim=1)
    lyapunov = estimate_largest_lyapunov(
        latent,
        times=dataset.times,
        min_separation=3,
        max_horizon=min(5, max(1, dataset.n_steps // 8)),
    )

    input_figure = figures_dir / f"{dataset_name}_input_trajectory.png"
    latent_figure = figures_dir / f"{dataset_name}_latent_trajectory.png"
    input_persistence_figure = figures_dir / f"{dataset_name}_persistence_input.png"
    latent_persistence_figure = figures_dir / f"{dataset_name}_persistence_latent.png"
    save_trajectory_figure(dataset, input_figure)
    save_point_figure(latent, latent_figure, title=f"{dataset_name} latent")
    save_persistence_figure(
        dataset.observations,
        input_persistence_figure,
        title=f"{dataset_name} input H1 persistence",
    )
    save_persistence_figure(
        latent,
        latent_persistence_figure,
        title=f"{dataset_name} latent H1 persistence",
    )

    baselines: list[TemporalModelProtocol] = [
        GRUBaseline(hidden_dim=8, epochs=config.baseline_epochs, learning_rate=0.03),
        VanillaNeuralODEBaseline(epochs=config.baseline_epochs, learning_rate=0.03),
        ReservoirBaseline(reservoir_dim=12),
    ]
    baseline_report = compare_baselines(
        dataset,
        seed=seed + 1000,
        baselines=baselines,
        include_tempora_smoke=False,
    )

    prediction_mse = reconstruction_mse(dataset.observations[1:], latent[1:])
    full_reconstruction_mse = reconstruction_mse(dataset.observations, latent)
    contraction_margin_min = float(
        cast(float, training.metrics["contraction_margin_min"])
    )
    contraction_margin_final = float(
        cast(float, training.metrics["contraction_margin_final"])
    )
    checkpoint_path = checkpoints_dir / f"{dataset_name}_model.pt"
    save_model_checkpoint(
        model,
        checkpoint_path,
        dataset_name=dataset_name,
        seed=seed,
        config=config,
        training_metrics=training.metrics,
    )
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "seed": seed,
        "model": "tempora_contractivectrnn",
        "checkpoint": str(checkpoint_path),
        "prediction_mse": prediction_mse,
        "reconstruction_mse": full_reconstruction_mse,
        "contraction_margin_min": contraction_margin_min,
        "contraction_margin_final": contraction_margin_final,
        "largest_lyapunov_estimate": lyapunov.largest_exponent,
        "tda_bottleneck_h0": topology["bottleneck_h0"],
        "tda_bottleneck_h1": topology["bottleneck_h1"],
        "time_warp_invariance_score": time_warp_invariance_score(dataset),
        "noise_robustness_score": noise_robustness_score(
            dataset,
            noise_std=0.05,
            seed=seed + 2000,
        ),
        "missing_segment_robustness_score": missing_segment_robustness_score(
            dataset,
            fraction=0.1,
            seed=seed + 3000,
        ),
        "training": training.metrics,
        "topology": topology,
        "lyapunov": lyapunov.to_metrics(),
        "baselines": baseline_report["models"],
        "figures": {
            "input_trajectory": str(input_figure),
            "latent_trajectory": str(latent_figure),
            "persistence_input": str(input_persistence_figure),
            "persistence_latent": str(latent_persistence_figure),
        },
    }
    validate_json_metrics(result)
    return result


def write_benchmark_config(config: SyntheticBenchmarkConfig, path: Path) -> None:
    """Write the resolved benchmark configuration next to generated artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.to_jsonable(), sort_keys=True),
        encoding="utf-8",
    )


def save_model_checkpoint(
    model: ContractiveCTRNN,
    path: Path,
    *,
    dataset_name: DatasetName,
    seed: int,
    config: SyntheticBenchmarkConfig,
    training_metrics: dict[str, object],
) -> None:
    """Save a reconstructable smoke-model checkpoint for one dataset run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": "ContractiveCTRNN",
            "model_kwargs": {
                "input_dim": model.input_dim,
                "latent_dim": model.latent_dim,
                "margin": model.margin,
                "lipschitz": model.lipschitz,
                "use_torchdiffeq": model.use_torchdiffeq,
            },
            "dataset": dataset_name,
            "seed": seed,
            "config": config.to_jsonable(),
            "training_metrics": training_metrics,
            "state_dict": model.state_dict(),
        },
        path,
    )


def encode_dataset(model: ContractiveCTRNN, dataset: TemporalDataset) -> FloatArray:
    """Encode a dataset through a trained contractive CTRNN."""

    with torch.no_grad():
        observations = torch.as_tensor(
            dataset.observations, dtype=torch.float32
        ).unsqueeze(0)
        times = torch.as_tensor(dataset.times, dtype=torch.float32)
        initial_state = observations[:, 0, :]
        states = model(observations, times=times, initial_state=initial_state)
    return cast(FloatArray, np.asarray(states.squeeze(0).numpy(), dtype=np.float64))


def render_benchmark_report(metrics: dict[str, Any]) -> str:
    """Render a concise Markdown benchmark report from saved metrics."""

    lines = [
        f"# TEMPORA Synthetic Benchmark: {metrics['run_id']}",
        "",
        "## Claims",
        "",
        "- This smoke benchmark records finite numerical evidence under explicit "
        "synthetic settings.",
        "- It does not prove general temporal semantics, homeomorphism, or "
        "real-world robustness.",
        "",
        "## Evidence",
        "",
    ]
    datasets = cast(dict[str, Any], metrics["datasets"])
    for dataset_name, dataset_metrics in datasets.items():
        lines.extend(
            [
                f"### {dataset_name}",
                "",
                f"- prediction_mse: {dataset_metrics['prediction_mse']:.6g}",
                "- contraction_margin_final: "
                f"{dataset_metrics['contraction_margin_final']:.6g}",
                "- largest_lyapunov_estimate: "
                f"{dataset_metrics['largest_lyapunov_estimate']:.6g}",
                f"- tda_bottleneck_h1: {dataset_metrics['tda_bottleneck_h1']:.6g}",
                "- time_warp_invariance_score: "
                f"{dataset_metrics['time_warp_invariance_score']:.6g}",
                "",
            ]
        )
        baselines = cast(dict[str, Any], dataset_metrics["baselines"])
        lines.append("| Model | prediction_mse | reconstruction_mse | fit_epochs |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            "| TEMPORA Contractive CTRNN | "
            f"{dataset_metrics['prediction_mse']:.6g} | "
            f"{dataset_metrics['reconstruction_mse']:.6g} | "
            f"{metrics['config']['epochs']} |"
        )
        for model_name, baseline_metrics in baselines.items():
            lines.append(
                f"| {model_name} | "
                f"{baseline_metrics['prediction_mse']:.6g} | "
                f"{baseline_metrics['reconstruction_mse']:.6g} | "
                f"{baseline_metrics['fit_epochs']:.0f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Open Points",
            "",
            "- Defaults are intentionally CI-small and not tuned.",
            "- Results are smoke artifacts until reviewed and rerun with "
            "benchmark settings.",
            "- Future phases should expand reporting and release documentation.",
            "",
        ]
    )
    return "\n".join(lines)


def save_point_figure(points: FloatArray, path: Path, *, title: str) -> None:
    """Save a 2D or 3D trajectory-like point plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(5, 4))
    if points.shape[1] >= 3:
        axis: Any = figure.add_subplot(111, projection="3d")
        axis.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=1)
        axis.set_zlabel("z")
    else:
        axis = figure.add_subplot(111)
        axis.plot(points[:, 0], points[:, 1], linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_persistence_figure(points: FloatArray, path: Path, *, title: str) -> None:
    """Save an H1 persistence diagram for a finite point cloud."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    result = compute_persistence_diagrams(points, maxdim=1)
    figure = cast(Any, plot_persistence_diagram(result, homology_dim=1))
    if figure.axes:
        figure.axes[0].set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def current_git_commit() -> str | None:
    """Return current git commit hash if available."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def dependency_versions() -> dict[str, str | None]:
    """Return versions for key runtime dependencies."""

    packages = ("numpy", "scipy", "torch", "ripser", "persim", "matplotlib")
    return {package: _package_version(package) for package in packages}


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def validate_json_metrics(value: Any) -> None:
    """Ensure nested benchmark metrics can be serialized and contain finite floats."""

    if isinstance(value, dict):
        for nested in value.values():
            validate_json_metrics(nested)
    elif isinstance(value, list):
        for nested in value:
            validate_json_metrics(nested)
    elif isinstance(value, float):
        if not np.isfinite(value):
            raise RuntimeError("benchmark metric contains a non-finite float.")
    else:
        json.dumps(value)
