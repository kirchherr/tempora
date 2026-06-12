import json
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import yaml

from tempora.experiments import load_benchmark_config, run_synthetic_benchmark
from tempora.experiments.run_synthetic import (
    current_git_commit,
    render_benchmark_report,
)
from tempora.training import load_contractive_ctrnn_checkpoint


def test_synthetic_smoke_benchmark_writes_metrics_figures_and_report(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "ci_smoke",
                "seed": 801,
                "output_root": str(tmp_path / "outputs"),
                "datasets": ["circle", "torus", "lorenz", "rossler"],
                "n_steps": 20,
                "epochs": 2,
                "learning_rate": 0.03,
                "plasticity_learning_rate": 0.0005,
                "baseline_epochs": 2,
            }
        ),
        encoding="utf-8",
    )
    config = load_benchmark_config(config_path)

    result = run_synthetic_benchmark(config)

    assert result.config_path.exists()
    assert result.metrics_path.exists()
    assert result.report_path.exists()
    assert len(result.checkpoint_paths) == 4
    resolved_config = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert resolved_config["run_id"] == "ci_smoke"
    assert resolved_config["datasets"] == ["circle", "torus", "lorenz", "rossler"]

    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "ci_smoke"
    assert set(payload["datasets"]) == {"circle", "torus", "lorenz", "rossler"}
    assert payload["config"]["epochs"] == 2
    assert payload["artifacts"]["config"] == str(result.config_path)
    assert set(payload["artifacts"]["checkpoints"]) == {
        str(path) for path in result.checkpoint_paths
    }
    assert payload["dependency_versions"]["torch"] is not None
    for dataset_name, dataset_metrics in payload["datasets"].items():
        assert dataset_metrics["prediction_mse"] >= 0.0
        assert dataset_metrics["contraction_margin_final"] > 0.0
        assert "gru" in dataset_metrics["baselines"]
        assert "vanilla_neural_ode" in dataset_metrics["baselines"]
        assert Path(dataset_metrics["figures"]["input_trajectory"]).exists()
        assert Path(dataset_metrics["figures"]["latent_trajectory"]).exists()
        assert Path(dataset_metrics["figures"]["persistence_input"]).exists()
        assert Path(dataset_metrics["figures"]["persistence_latent"]).exists()
        checkpoint_path = Path(dataset_metrics["checkpoint"])
        assert checkpoint_path.exists()
        checkpoint = load_contractive_ctrnn_checkpoint(checkpoint_path)
        assert checkpoint.metadata["model_class"] == "ContractiveCTRNN"
        assert checkpoint.metadata["dataset"] == dataset_name
        assert checkpoint.metadata["seed"] == dataset_metrics["seed"]
        assert checkpoint.metadata["config"]["run_id"] == "ci_smoke"
        assert checkpoint.model.input_dim == checkpoint.model.latent_dim
    assert "## Claims" in result.report_path.read_text(encoding="utf-8")
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "## Run Metadata" in report_text
    assert "## Artifacts" in report_text
    assert "## Dependency Versions" in report_text


def test_render_benchmark_report_separates_claims_evidence_and_open_points() -> None:
    metrics = {
        "run_id": "example",
        "seed": 11,
        "config": {"epochs": 1},
        "artifacts": {
            "config": "outputs/example/config.yaml",
            "metrics": "outputs/example/metrics.json",
            "report": "outputs/example/report.md",
            "checkpoints": ["outputs/example/checkpoints/circle_model.pt"],
        },
        "dependency_versions": {"numpy": "1.0.0", "torch": "2.0.0"},
        "git_commit": "abc123",
        "runtime": {
            "python": "3.11.0",
            "platform": "test-platform",
            "elapsed_seconds": 1.25,
        },
        "datasets": {
            "circle": {
                "prediction_mse": 0.1,
                "reconstruction_mse": 0.2,
                "contraction_margin_final": 0.3,
                "largest_lyapunov_estimate": -0.1,
                "tda_bottleneck_h1": 0.05,
                "time_warp_invariance_score": 1.0,
                "baselines": {
                    "gru": {
                        "prediction_mse": 0.4,
                        "reconstruction_mse": 0.5,
                        "fit_epochs": 1.0,
                    }
                },
                "figures": {
                    "input_trajectory": "outputs/example/figures/circle.png",
                    "persistence_input": "outputs/example/figures/circle_h1.png",
                },
                "checkpoint": "outputs/example/checkpoints/circle_model.pt",
            }
        },
    }

    report = render_benchmark_report(metrics)

    assert "## Claims" in report
    assert "## Evidence" in report
    assert "## Run Metadata" in report
    assert "## Artifacts" in report
    assert "## Dependency Versions" in report
    assert "## Open Points" in report
    assert "git_commit: `abc123`" in report
    assert "outputs/example/config.yaml" in report
    assert "| torch | `2.0.0` |" in report
    assert "persistence_input" in report
    assert "outputs/example/checkpoints/circle_model.pt" in report
    assert "TEMPORA Contractive CTRNN" in report


def test_current_git_commit_marks_workspace_safe(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> CompletedProcess[str]:
        commands.append(command)
        return CompletedProcess(command, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr("tempora.experiments.run_synthetic.subprocess.run", fake_run)

    assert current_git_commit() == "abc123"
    assert commands
    assert commands[0][1] == "-c"
    assert commands[0][2].startswith("safe.directory=")
