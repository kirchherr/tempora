from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.release_smoke import main, run_release_smoke

from tempora.experiments.run_synthetic import SyntheticBenchmarkResult


def test_run_release_smoke_renders_report_and_checks_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path)
    result = _result(tmp_path, passed=True)
    calls: list[Path] = []

    monkeypatch.setattr(
        "scripts.release_smoke.run_synthetic_benchmark",
        lambda config: result,
    )
    monkeypatch.setattr(
        "scripts.release_smoke.validate_artifact_paths",
        lambda metrics, *, base_dir: calls.append(base_dir),
    )

    completed = run_release_smoke(config_path)

    assert completed is result
    assert calls == [Path.cwd()]
    assert "TEMPORA Synthetic Benchmark" in result.report_path.read_text(
        encoding="utf-8"
    )
    assert "passed=True" in capsys.readouterr().out


def test_run_release_smoke_rejects_failing_certificate_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    result = _result(tmp_path, passed=False)

    monkeypatch.setattr(
        "scripts.release_smoke.run_synthetic_benchmark",
        lambda config: result,
    )
    monkeypatch.setattr(
        "scripts.release_smoke.validate_artifact_paths",
        lambda metrics, *, base_dir: None,
    )

    with pytest.raises(RuntimeError, match="certificate gate failed"):
        run_release_smoke(config_path)


def test_release_smoke_main_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path)

    def fail_release_smoke(config_path: Path) -> SyntheticBenchmarkResult:
        raise RuntimeError(f"failed config: {config_path}")

    monkeypatch.setattr("scripts.release_smoke.run_release_smoke", fail_release_smoke)

    assert main(["--config", str(config_path)]) == 1
    assert "release smoke failed" in capsys.readouterr().err


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "benchmark_smoke",
                "seed": 42,
                "output_root": str(tmp_path / "outputs"),
                "datasets": ["circle"],
                "n_steps": 20,
                "epochs": 2,
                "learning_rate": 0.03,
                "plasticity_learning_rate": 0.0005,
                "baseline_epochs": 2,
                "topology_max_distance": 1.0,
                "required_certificates": ["contraction"],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _result(tmp_path: Path, *, passed: bool) -> SyntheticBenchmarkResult:
    output_dir = tmp_path / "outputs" / "benchmark_smoke"
    config_path = output_dir / "config.yaml"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    checkpoint_path = output_dir / "checkpoints" / "circle.pt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = _valid_metrics(
        config_path=config_path,
        metrics_path=metrics_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        figure_path=output_dir / "figures" / "circle_input.png",
        passed=passed,
    )
    return SyntheticBenchmarkResult(
        output_dir=output_dir,
        config_path=config_path,
        metrics_path=metrics_path,
        report_path=report_path,
        checkpoint_paths=(checkpoint_path,),
        metrics=metrics,
    )


def _valid_metrics(
    *,
    config_path: Path,
    metrics_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    figure_path: Path,
    passed: bool,
) -> dict[str, Any]:
    return {
        "run_id": "benchmark_smoke",
        "seed": 42,
        "config": {
            "run_id": "benchmark_smoke",
            "seed": 42,
            "output_root": str(config_path.parents[1]),
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
            "config": str(config_path),
            "metrics": str(metrics_path),
            "report": str(report_path),
            "checkpoints": [str(checkpoint_path)],
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
                "checkpoint": str(checkpoint_path),
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
                        "contraction_margin": 0.6 if passed else 0.01,
                        "required_margin": 0.05,
                        "is_certified": passed,
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
                "figures": {"input_trajectory": str(figure_path)},
            }
        },
        "certificate_summary": {
            "all_certified": passed,
            "by_certificate": {
                "contraction": {
                    "total": 1,
                    "certified": 1 if passed else 0,
                    "failed": 0 if passed else 1,
                }
            },
            "failures": []
            if passed
            else [
                {
                    "dataset": "circle",
                    "certificate": "contraction",
                    "theorem": "theorem_01_sufficient_contraction",
                    "margin": 0.01,
                    "required_margin": 0.05,
                }
            ],
        },
        "certificate_gate": {
            "passed": passed,
            "required_certificates": ["contraction"],
            "failures": []
            if passed
            else [
                {
                    "dataset": "circle",
                    "certificate": "contraction",
                    "theorem": "theorem_01_sufficient_contraction",
                    "margin": 0.01,
                    "required_margin": 0.05,
                }
            ],
        },
    }
