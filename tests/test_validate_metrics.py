import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tempora.experiments.run_synthetic import validate_benchmark_metrics

SCRIPT = Path("scripts/validate_metrics.py")


def test_validate_benchmark_metrics_accepts_schema_payload() -> None:
    validate_benchmark_metrics(_valid_metrics())


def test_validate_benchmark_metrics_rejects_missing_top_level_field() -> None:
    metrics = _valid_metrics()
    del metrics["datasets"]

    with pytest.raises(ValueError, match="datasets"):
        validate_benchmark_metrics(metrics)


def test_validate_benchmark_metrics_rejects_non_finite_values() -> None:
    metrics = _valid_metrics()
    datasets = metrics["datasets"]
    assert isinstance(datasets, dict)
    circle = datasets["circle"]
    assert isinstance(circle, dict)
    circle["prediction_mse"] = float("nan")

    with pytest.raises(RuntimeError, match="non-finite"):
        validate_benchmark_metrics(metrics)


def test_validate_metrics_cli_accepts_schema_payload(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_valid_metrics()), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(metrics_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "metrics schema valid" in completed.stdout


def test_validate_metrics_cli_rejects_schema_payload(tmp_path: Path) -> None:
    metrics = _valid_metrics()
    del metrics["certificate_gate"]
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(metrics_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "certificate_gate" in completed.stderr


def _valid_metrics() -> dict[str, Any]:
    return copy.deepcopy(
        {
            "run_id": "benchmark_smoke",
            "seed": 42,
            "config": {
                "run_id": "benchmark_smoke",
                "seed": 42,
                "output_root": "outputs",
                "datasets": ["circle"],
                "n_steps": 20,
                "epochs": 2,
                "learning_rate": 0.03,
                "plasticity_learning_rate": 0.0005,
                "baseline_epochs": 2,
                "topology_max_distance": 1.0,
                "required_certificates": ["contraction"],
            },
            "artifacts": {
                "config": "outputs/benchmark_smoke/config.yaml",
                "metrics": "outputs/benchmark_smoke/metrics.json",
                "report": "outputs/benchmark_smoke/report.md",
                "checkpoints": ["outputs/benchmark_smoke/checkpoints/circle.pt"],
            },
            "git_commit": None,
            "dependency_versions": {"numpy": "1.26.0", "torch": "2.3.0"},
            "runtime": {
                "python": "3.11.0",
                "platform": "test-platform",
                "elapsed_seconds": 1.0,
            },
            "datasets": {
                "circle": {
                    "dataset": "circle",
                    "seed": 42,
                    "model": "tempora_contractivectrnn",
                    "checkpoint": "outputs/benchmark_smoke/checkpoints/circle.pt",
                    "prediction_mse": 0.1,
                    "reconstruction_mse": 0.1,
                    "contraction_margin_min": 0.2,
                    "contraction_margin_final": 0.3,
                    "largest_lyapunov_estimate": -0.1,
                    "tda_bottleneck_h0": 0.0,
                    "tda_bottleneck_h1": 0.05,
                    "time_warp_invariance_score": 0.1,
                    "noise_robustness_score": 0.1,
                    "missing_segment_robustness_score": 0.1,
                    "training": {"loss_final": 0.1},
                    "topology": {"bottleneck_h1": 0.05},
                    "lyapunov": {"largest_exponent": -0.1},
                    "certificates": {
                        "contraction": {
                            "theorem": "theorem_01_sufficient_contraction",
                            "damping_min": 1.0,
                            "recurrent_spectral_norm": 0.4,
                            "lipschitz": 1.0,
                            "contraction_margin": 0.6,
                            "required_margin": 0.05,
                            "is_certified": True,
                            "assumptions": ["bounded tanh nonlinearity"],
                            "limitation": "Sufficient contraction certificate only.",
                        }
                    },
                    "baselines": {
                        "gru": {
                            "prediction_mse": 0.2,
                            "reconstruction_mse": 0.3,
                            "fit_epochs": 2,
                        }
                    },
                    "figures": {
                        "input_trajectory": (
                            "outputs/benchmark_smoke/figures/circle_input.png"
                        )
                    },
                }
            },
            "certificate_summary": {
                "all_certified": True,
                "by_certificate": {
                    "contraction": {"total": 1, "certified": 1, "failed": 0}
                },
                "failures": [],
            },
            "certificate_gate": {
                "passed": True,
                "required_certificates": ["contraction"],
                "failures": [],
            },
        }
    )
