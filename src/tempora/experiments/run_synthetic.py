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
from tempora.proof import (
    certify_model_contraction,
    certify_projected_update_stability,
    certify_topology_comparison,
)
from tempora.training import save_contractive_ctrnn_checkpoint, train_circle_next_step
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
    topology_max_distance: float
    required_certificates: tuple[str, ...] = ("contraction",)

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["datasets"] = list(self.datasets)
        payload["required_certificates"] = list(self.required_certificates)
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
    required_certificate_items = raw.get("required_certificates", ["contraction"])
    required_certificates: tuple[str, ...]
    if isinstance(required_certificate_items, str):
        required_certificates = (required_certificate_items,)
    else:
        required_certificates = tuple(str(item) for item in required_certificate_items)
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
        topology_max_distance=float(raw.get("topology_max_distance", 1.0)),
        required_certificates=required_certificates,
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
    if config.topology_max_distance < 0.0:
        raise ValueError("topology_max_distance must be non-negative.")

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

    certificate_summary = summarize_benchmark_certificates(dataset_results)
    certificate_gate = evaluate_certificate_gate(
        dataset_results,
        certificate_summary,
        required_certificates=config.required_certificates,
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
        "certificate_summary": certificate_summary,
        "certificate_gate": certificate_gate,
    }
    validate_benchmark_metrics(metrics)
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
    save_contractive_ctrnn_checkpoint(
        model,
        checkpoint_path,
        dataset_name=dataset_name,
        seed=seed,
        config=config.to_jsonable(),
        training_metrics=training.metrics,
    )
    contraction_certificate = certify_model_contraction(model)
    certificates: dict[str, Any] = {
        "contraction": contraction_certificate.to_jsonable(),
    }
    if training.last_plasticity_log is not None:
        learning_certificate = certify_projected_update_stability(
            training.last_plasticity_log,
            required_margin=model.margin,
        )
        certificates["learning_stability"] = learning_certificate.to_jsonable()
    topology_certificate = certify_topology_comparison(
        topology,
        homology_dim=1,
        max_distance=config.topology_max_distance,
    )
    certificates["topology_comparison"] = topology_certificate.to_jsonable()
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
        "certificates": certificates,
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
    lines.extend(_render_run_metadata(metrics))
    lines.extend(_render_artifacts(metrics))
    lines.extend(_render_dependency_table(metrics))
    lines.extend(_render_certificate_summary(metrics))
    lines.extend(_render_certificate_gate(metrics))
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
        dataset_figures = cast(dict[str, Any], dataset_metrics.get("figures", {}))
        if dataset_figures:
            lines.append("Figures:")
            for figure_name, figure_path in sorted(dataset_figures.items()):
                lines.append(f"- {figure_name}: `{figure_path}`")
            lines.append("")
        if checkpoint_path := dataset_metrics.get("checkpoint"):
            lines.extend(["Checkpoint:", f"- `{checkpoint_path}`", ""])
        dataset_certificates = cast(
            dict[str, Any],
            dataset_metrics.get("certificates", {}),
        )
        if dataset_certificates:
            lines.append("Certificates:")
            contraction_certificate = cast(
                dict[str, Any],
                dataset_certificates.get("contraction", {}),
            )
            if contraction_certificate:
                lines.extend(
                    [
                        "- contraction: "
                        f"`certified={contraction_certificate['is_certified']}`, "
                        "margin="
                        f"`{contraction_certificate['contraction_margin']:.6g}`, "
                        "required="
                        f"`{contraction_certificate['required_margin']:.6g}`",
                    ]
                )
            learning_certificate = cast(
                dict[str, Any],
                dataset_certificates.get("learning_stability", {}),
            )
            if learning_certificate:
                lines.extend(
                    [
                        "- learning_stability: "
                        f"`certified={learning_certificate['is_certified']}`, "
                        "margin_after="
                        f"`{learning_certificate['margin_after']:.6g}`, "
                        "required="
                        f"`{learning_certificate['required_margin']:.6g}`",
                    ]
                )
            topology_certificate = cast(
                dict[str, Any],
                dataset_certificates.get("topology_comparison", {}),
            )
            if topology_certificate:
                lines.extend(
                    [
                        "- topology_comparison: "
                        f"`certified={topology_certificate['is_certified']}`, "
                        f"h{topology_certificate['homology_dim']} "
                        f"{topology_certificate['metric']}="
                        f"`{topology_certificate['distance']:.6g}`, "
                        "max="
                        f"`{topology_certificate['max_distance']:.6g}`",
                    ]
                )
            lines.append("")
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


def summarize_benchmark_certificates(datasets: dict[str, Any]) -> dict[str, Any]:
    """Summarize per-dataset certificate status for benchmark review artifacts."""

    by_certificate: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    for dataset_name, dataset_metrics in sorted(datasets.items()):
        certificates = cast(dict[str, Any], dataset_metrics.get("certificates", {}))
        for certificate_name, certificate_payload in sorted(certificates.items()):
            certificate = cast(dict[str, Any], certificate_payload)
            summary = by_certificate.setdefault(
                certificate_name,
                {"total": 0, "certified": 0, "failed": 0},
            )
            summary["total"] += 1
            if certificate.get("is_certified") is True:
                summary["certified"] += 1
            else:
                summary["failed"] += 1
                failures.append(
                    _certificate_failure_summary(
                        dataset_name,
                        certificate_name,
                        certificate,
                    )
                )
    return {
        "all_certified": not failures,
        "by_certificate": by_certificate,
        "failures": failures,
    }


def evaluate_certificate_gate(
    datasets: dict[str, Any],
    certificate_summary: dict[str, Any],
    *,
    required_certificates: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate a run-level gate over selected certificate types."""

    dataset_count = len(datasets)
    by_certificate = cast(dict[str, Any], certificate_summary.get("by_certificate", {}))
    summary_failures = cast(
        list[dict[str, Any]], certificate_summary.get("failures", [])
    )
    required = tuple(dict.fromkeys(required_certificates))
    failures: list[dict[str, Any]] = []
    for certificate_name in required:
        counts = by_certificate.get(certificate_name)
        if not isinstance(counts, dict):
            failures.append(
                {
                    "certificate": certificate_name,
                    "reason": "missing",
                    "expected_datasets": dataset_count,
                    "observed_datasets": 0,
                }
            )
            continue
        observed_total = int(counts.get("total", 0))
        if observed_total < dataset_count:
            failures.append(
                {
                    "certificate": certificate_name,
                    "reason": "missing_some_datasets",
                    "expected_datasets": dataset_count,
                    "observed_datasets": observed_total,
                }
            )
        if int(counts.get("failed", 0)) > 0:
            failures.extend(
                failure
                for failure in summary_failures
                if failure.get("certificate") == certificate_name
            )
    return {
        "passed": not failures,
        "required_certificates": list(required),
        "failures": failures,
    }


def validate_benchmark_metrics(metrics: dict[str, Any]) -> None:
    """Validate the benchmark `metrics.json` payload structure.

    The validator checks the stable schema consumed by report rendering,
    release review, and certificate-gate tooling. It intentionally validates
    structure and finite JSON values; it does not certify scientific claims.
    """

    validate_json_metrics(metrics)
    _require_keys(
        metrics,
        (
            "run_id",
            "seed",
            "config",
            "artifacts",
            "git_commit",
            "dependency_versions",
            "runtime",
            "datasets",
            "certificate_summary",
            "certificate_gate",
        ),
        "metrics",
    )
    _require_non_empty_string(metrics["run_id"], "metrics.run_id")
    _require_int(metrics["seed"], "metrics.seed")
    _validate_metrics_config(_require_mapping(metrics["config"], "metrics.config"))
    _validate_metrics_artifacts(
        _require_mapping(metrics["artifacts"], "metrics.artifacts")
    )
    if metrics["git_commit"] is not None:
        _require_non_empty_string(metrics["git_commit"], "metrics.git_commit")
    _require_mapping(
        metrics["dependency_versions"],
        "metrics.dependency_versions",
    )
    _validate_runtime(_require_mapping(metrics["runtime"], "metrics.runtime"))
    datasets = _require_mapping(metrics["datasets"], "metrics.datasets")
    if not datasets:
        raise ValueError("metrics.datasets must contain at least one dataset.")
    for dataset_name, dataset_payload in datasets.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("metrics.datasets keys must be non-empty strings.")
        _validate_dataset_metrics(
            _require_mapping(dataset_payload, f"metrics.datasets.{dataset_name}"),
            f"metrics.datasets.{dataset_name}",
        )
    _validate_certificate_summary(
        _require_mapping(
            metrics["certificate_summary"],
            "metrics.certificate_summary",
        )
    )
    _validate_certificate_gate(
        _require_mapping(metrics["certificate_gate"], "metrics.certificate_gate")
    )


def _validate_metrics_config(config: dict[str, Any]) -> None:
    _require_keys(
        config,
        (
            "run_id",
            "seed",
            "output_root",
            "datasets",
            "n_steps",
            "epochs",
            "learning_rate",
            "plasticity_learning_rate",
            "baseline_epochs",
            "topology_max_distance",
            "required_certificates",
        ),
        "metrics.config",
    )
    _require_non_empty_string(config["run_id"], "metrics.config.run_id")
    _require_int(config["seed"], "metrics.config.seed")
    _require_non_empty_string(config["output_root"], "metrics.config.output_root")
    _require_string_list(config["datasets"], "metrics.config.datasets")
    _require_int(config["n_steps"], "metrics.config.n_steps")
    _require_int(config["epochs"], "metrics.config.epochs")
    _require_number(config["learning_rate"], "metrics.config.learning_rate")
    _require_number(
        config["plasticity_learning_rate"],
        "metrics.config.plasticity_learning_rate",
    )
    _require_int(config["baseline_epochs"], "metrics.config.baseline_epochs")
    _require_number(
        config["topology_max_distance"],
        "metrics.config.topology_max_distance",
    )
    _require_string_list(
        config["required_certificates"],
        "metrics.config.required_certificates",
    )


def _validate_metrics_artifacts(artifacts: dict[str, Any]) -> None:
    _require_keys(
        artifacts,
        ("config", "metrics", "report", "checkpoints"),
        "metrics.artifacts",
    )
    _require_non_empty_string(artifacts["config"], "metrics.artifacts.config")
    _require_non_empty_string(artifacts["metrics"], "metrics.artifacts.metrics")
    _require_non_empty_string(artifacts["report"], "metrics.artifacts.report")
    _require_string_list(artifacts["checkpoints"], "metrics.artifacts.checkpoints")


def _validate_runtime(runtime: dict[str, Any]) -> None:
    _require_keys(
        runtime,
        ("python", "platform", "elapsed_seconds"),
        "metrics.runtime",
    )
    _require_non_empty_string(runtime["python"], "metrics.runtime.python")
    _require_non_empty_string(runtime["platform"], "metrics.runtime.platform")
    _require_number(runtime["elapsed_seconds"], "metrics.runtime.elapsed_seconds")


def _validate_dataset_metrics(dataset: dict[str, Any], location: str) -> None:
    _require_keys(
        dataset,
        (
            "dataset",
            "seed",
            "model",
            "checkpoint",
            "prediction_mse",
            "reconstruction_mse",
            "contraction_margin_min",
            "contraction_margin_final",
            "largest_lyapunov_estimate",
            "tda_bottleneck_h0",
            "tda_bottleneck_h1",
            "time_warp_invariance_score",
            "noise_robustness_score",
            "missing_segment_robustness_score",
            "training",
            "topology",
            "lyapunov",
            "certificates",
            "baselines",
            "figures",
        ),
        location,
    )
    _require_non_empty_string(dataset["dataset"], f"{location}.dataset")
    _require_int(dataset["seed"], f"{location}.seed")
    _require_non_empty_string(dataset["model"], f"{location}.model")
    _require_non_empty_string(dataset["checkpoint"], f"{location}.checkpoint")
    for metric_name in (
        "prediction_mse",
        "reconstruction_mse",
        "contraction_margin_min",
        "contraction_margin_final",
        "largest_lyapunov_estimate",
        "tda_bottleneck_h0",
        "tda_bottleneck_h1",
        "time_warp_invariance_score",
        "noise_robustness_score",
        "missing_segment_robustness_score",
    ):
        _require_number(dataset[metric_name], f"{location}.{metric_name}")
    _require_mapping(dataset["training"], f"{location}.training")
    _require_mapping(dataset["topology"], f"{location}.topology")
    _require_mapping(dataset["lyapunov"], f"{location}.lyapunov")
    certificates = _require_mapping(dataset["certificates"], f"{location}.certificates")
    if not certificates:
        raise ValueError(f"{location}.certificates must not be empty.")
    for certificate_name, certificate_payload in certificates.items():
        if not isinstance(certificate_name, str) or not certificate_name:
            raise ValueError(f"{location}.certificates keys must be non-empty strings.")
        _validate_certificate_payload(
            _require_mapping(
                certificate_payload,
                f"{location}.certificates.{certificate_name}",
            ),
            f"{location}.certificates.{certificate_name}",
        )
    _require_mapping(dataset["baselines"], f"{location}.baselines")
    _require_mapping(dataset["figures"], f"{location}.figures")


def _validate_certificate_payload(certificate: dict[str, Any], location: str) -> None:
    _require_keys(
        certificate,
        ("theorem", "is_certified", "assumptions", "limitation"),
        location,
    )
    _require_non_empty_string(certificate["theorem"], f"{location}.theorem")
    _require_bool(certificate["is_certified"], f"{location}.is_certified")
    _require_string_list(certificate["assumptions"], f"{location}.assumptions")
    _require_non_empty_string(certificate["limitation"], f"{location}.limitation")


def _validate_certificate_summary(summary: dict[str, Any]) -> None:
    _require_keys(
        summary,
        ("all_certified", "by_certificate", "failures"),
        "metrics.certificate_summary",
    )
    _require_bool(
        summary["all_certified"],
        "metrics.certificate_summary.all_certified",
    )
    by_certificate = _require_mapping(
        summary["by_certificate"],
        "metrics.certificate_summary.by_certificate",
    )
    for certificate_name, counts_payload in by_certificate.items():
        if not isinstance(certificate_name, str) or not certificate_name:
            raise ValueError(
                "metrics.certificate_summary.by_certificate keys must be "
                "non-empty strings."
            )
        counts = _require_mapping(
            counts_payload,
            f"metrics.certificate_summary.by_certificate.{certificate_name}",
        )
        _require_keys(
            counts,
            ("total", "certified", "failed"),
            f"metrics.certificate_summary.by_certificate.{certificate_name}",
        )
        _require_int(
            counts["total"],
            f"metrics.certificate_summary.by_certificate.{certificate_name}.total",
        )
        _require_int(
            counts["certified"],
            f"metrics.certificate_summary.by_certificate.{certificate_name}.certified",
        )
        _require_int(
            counts["failed"],
            f"metrics.certificate_summary.by_certificate.{certificate_name}.failed",
        )
    _require_list(summary["failures"], "metrics.certificate_summary.failures")


def _validate_certificate_gate(gate: dict[str, Any]) -> None:
    _require_keys(
        gate,
        ("passed", "required_certificates", "failures"),
        "metrics.certificate_gate",
    )
    _require_bool(gate["passed"], "metrics.certificate_gate.passed")
    _require_string_list(
        gate["required_certificates"],
        "metrics.certificate_gate.required_certificates",
    )
    _require_list(gate["failures"], "metrics.certificate_gate.failures")


def _require_keys(
    payload: dict[str, Any],
    required_keys: tuple[str, ...],
    location: str,
) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{location} missing required field(s): {missing_text}.")


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object.")
    return cast(dict[str, Any], value)


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a JSON list.")
    return value


def _require_string_list(value: Any, location: str) -> list[str]:
    items = _require_list(value, location)
    if not all(isinstance(item, str) and item for item in items):
        raise ValueError(f"{location} must contain only non-empty strings.")
    return cast(list[str], items)


def _require_non_empty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string.")
    return value


def _require_int(value: Any, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{location} must be an integer.")
    return value


def _require_number(value: Any, location: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{location} must be numeric.")
    return float(value)


def _require_bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be boolean.")
    return value


def _certificate_failure_summary(
    dataset_name: str,
    certificate_name: str,
    certificate: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": dataset_name,
        "certificate": certificate_name,
        "theorem": certificate.get("theorem", "unknown"),
    }
    if certificate_name == "topology_comparison":
        payload.update(
            {
                "metric": certificate.get("metric", "unknown"),
                "homology_dim": certificate.get("homology_dim"),
                "distance": certificate.get("distance"),
                "max_distance": certificate.get("max_distance"),
            }
        )
    elif "required_margin" in certificate:
        payload["margin"] = certificate.get(
            "contraction_margin",
            certificate.get("margin_after"),
        )
        payload["required_margin"] = certificate.get("required_margin")
    return payload


def _render_certificate_summary(metrics: dict[str, Any]) -> list[str]:
    datasets = cast(dict[str, Any], metrics.get("datasets", {}))
    summary = cast(
        dict[str, Any],
        metrics.get("certificate_summary")
        or summarize_benchmark_certificates(datasets),
    )
    by_certificate = cast(dict[str, Any], summary.get("by_certificate", {}))
    if not by_certificate:
        return [
            "## Certificate Summary",
            "",
            "- No certificate payloads were recorded.",
            "",
        ]

    lines = [
        "## Certificate Summary",
        "",
        "| Certificate | certified | failed | total |",
        "|---|---:|---:|---:|",
    ]
    for certificate_name, certificate_counts in sorted(by_certificate.items()):
        counts = cast(dict[str, int], certificate_counts)
        lines.append(
            f"| {certificate_name} | {counts['certified']} | "
            f"{counts['failed']} | {counts['total']} |"
        )

    failures = cast(list[dict[str, Any]], summary.get("failures", []))
    if failures:
        lines.extend(["", "Certificate failures:"])
        for failure in failures:
            lines.append(f"- {_format_certificate_failure(failure)}")
    lines.append("")
    return lines


def _render_certificate_gate(metrics: dict[str, Any]) -> list[str]:
    config = cast(dict[str, Any], metrics.get("config", {}))
    datasets = cast(dict[str, Any], metrics.get("datasets", {}))
    summary = cast(
        dict[str, Any],
        metrics.get("certificate_summary")
        or summarize_benchmark_certificates(datasets),
    )
    required = tuple(str(item) for item in config.get("required_certificates", []))
    gate = cast(
        dict[str, Any],
        metrics.get("certificate_gate")
        or evaluate_certificate_gate(
            datasets,
            summary,
            required_certificates=required,
        ),
    )
    required_certificates = cast(list[str], gate.get("required_certificates", []))
    required_text = (
        ", ".join(f"`{certificate}`" for certificate in required_certificates)
        if required_certificates
        else "`none`"
    )
    lines = [
        "## Certificate Gate",
        "",
        f"- passed: `{gate.get('passed', False)}`",
        f"- required_certificates: {required_text}",
    ]
    failures = cast(list[dict[str, Any]], gate.get("failures", []))
    if failures:
        lines.extend(["", "Gate failures:"])
        for failure in failures:
            lines.append(f"- {_format_gate_failure(failure)}")
    lines.append("")
    return lines


def _format_gate_failure(failure: dict[str, Any]) -> str:
    if "reason" in failure:
        certificate = failure.get("certificate", "unknown")
        reason = failure.get("reason", "unknown")
        expected = failure.get("expected_datasets", "unknown")
        observed = failure.get("observed_datasets", "unknown")
        return (
            f"{certificate}: `{reason}`, "
            f"observed=`{observed}`, expected=`{expected}`"
        )
    return _format_certificate_failure(failure)


def _format_certificate_failure(failure: dict[str, Any]) -> str:
    dataset = failure.get("dataset", "unknown")
    certificate = failure.get("certificate", "unknown")
    theorem = failure.get("theorem", "unknown")
    if "distance" in failure:
        return (
            f"{dataset}/{certificate}: `{theorem}`, "
            f"distance=`{_format_report_value(failure.get('distance'))}`, "
            f"max=`{_format_report_value(failure.get('max_distance'))}`"
        )
    if "margin" in failure:
        return (
            f"{dataset}/{certificate}: `{theorem}`, "
            f"margin=`{_format_report_value(failure.get('margin'))}`, "
            f"required=`{_format_report_value(failure.get('required_margin'))}`"
        )
    return f"{dataset}/{certificate}: `{theorem}`"


def _format_report_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _render_run_metadata(metrics: dict[str, Any]) -> list[str]:
    runtime = cast(dict[str, Any], metrics.get("runtime", {}))
    elapsed = runtime.get("elapsed_seconds")
    elapsed_text = f"{elapsed:.3f}" if isinstance(elapsed, int | float) else "unknown"
    git_commit = metrics.get("git_commit") or "unavailable"
    return [
        "## Run Metadata",
        "",
        f"- seed: `{metrics.get('seed', 'unknown')}`",
        f"- git_commit: `{git_commit}`",
        f"- python: `{runtime.get('python', 'unknown')}`",
        f"- platform: `{runtime.get('platform', 'unknown')}`",
        f"- elapsed_seconds: `{elapsed_text}`",
        "",
    ]


def _render_artifacts(metrics: dict[str, Any]) -> list[str]:
    artifacts = cast(dict[str, Any], metrics.get("artifacts", {}))
    if not artifacts:
        return [
            "## Artifacts",
            "",
            "- Artifact paths were not recorded in this metrics payload.",
            "",
        ]
    lines = ["## Artifacts", ""]
    for name in ("config", "metrics", "report"):
        if artifact_path := artifacts.get(name):
            lines.append(f"- {name}: `{artifact_path}`")
    checkpoints = artifacts.get("checkpoints", [])
    if isinstance(checkpoints, list) and checkpoints:
        lines.append("- checkpoints:")
        for checkpoint_path in checkpoints:
            lines.append(f"  - `{checkpoint_path}`")
    lines.append("")
    return lines


def _render_dependency_table(metrics: dict[str, Any]) -> list[str]:
    dependencies = cast(dict[str, Any], metrics.get("dependency_versions", {}))
    if not dependencies:
        return [
            "## Dependency Versions",
            "",
            "- Dependency versions were not recorded in this metrics payload.",
            "",
        ]
    lines = [
        "## Dependency Versions",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    for package, package_version in sorted(dependencies.items()):
        version_text = package_version if package_version is not None else "unavailable"
        lines.append(f"| {package} | `{version_text}` |")
    lines.append("")
    return lines


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

    safe_directory = str(Path.cwd().resolve())
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={safe_directory}", "rev-parse", "HEAD"],
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
