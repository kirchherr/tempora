import json
from pathlib import Path
from typing import Any

import pytest
from scripts.release_evidence import ReleaseEvidenceResult, main, run_release_evidence

from tempora.experiments.run_synthetic import SyntheticBenchmarkResult


def test_run_release_evidence_builds_and_validates_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("configs/benchmark_smoke.yaml")
    config_path.parent.mkdir(parents=True)
    config_path.write_text("run_id: benchmark_smoke\n", encoding="utf-8")
    smoke_result = _write_smoke_result()
    seen_configs: list[Path] = []

    def fake_release_smoke(path: Path) -> SyntheticBenchmarkResult:
        seen_configs.append(path)
        return smoke_result

    monkeypatch.setattr(
        "scripts.release_evidence.run_release_smoke",
        fake_release_smoke,
    )

    result = run_release_evidence(
        config_path,
        evidence_index_path=Path("outputs/evidence_index.json"),
        evidence_report_path=Path("outputs/evidence_report.md"),
        evidence_bundle_path=Path("outputs/evidence_bundle.json"),
    )

    assert seen_configs == [config_path]
    assert result.metrics_path == Path("outputs/benchmark_smoke/metrics.json")
    assert result.artifact_manifest_path == Path(
        "outputs/benchmark_smoke/artifact_manifest.json"
    )
    assert result.evidence_index_path.exists()
    assert result.evidence_report_path.exists()
    assert result.evidence_bundle_path.exists()

    bundle = json.loads(result.evidence_bundle_path.read_text(encoding="utf-8"))
    assert bundle["schema"] == "tempora.evidence_bundle.v1"
    assert bundle["run_count"] == 1
    assert bundle["git_commits"] == ["abc123"]


def test_release_evidence_main_reports_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = ReleaseEvidenceResult(
        metrics_path=tmp_path / "metrics.json",
        benchmark_report_path=tmp_path / "report.md",
        artifact_manifest_path=tmp_path / "artifact_manifest.json",
        evidence_index_path=tmp_path / "evidence_index.json",
        evidence_report_path=tmp_path / "evidence_report.md",
        evidence_bundle_path=tmp_path / "evidence_bundle.json",
    )

    def fake_release_evidence(
        config_path: Path,
        *,
        evidence_index_path: Path,
        evidence_report_path: Path,
        evidence_bundle_path: Path,
    ) -> ReleaseEvidenceResult:
        assert config_path == Path("config.yaml")
        assert evidence_index_path == Path("index.json")
        assert evidence_report_path == Path("report.md")
        assert evidence_bundle_path == Path("bundle.json")
        return expected

    monkeypatch.setattr(
        "scripts.release_evidence.run_release_evidence",
        fake_release_evidence,
    )

    completed = main(
        [
            "--config",
            "config.yaml",
            "--index-output",
            "index.json",
            "--report-output",
            "report.md",
            "--bundle-output",
            "bundle.json",
        ]
    )

    output = capsys.readouterr().out
    assert completed == 0
    assert f"release evidence metrics: {expected.metrics_path}" in output
    assert f"release evidence bundle: {expected.evidence_bundle_path}" in output


def test_release_evidence_main_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_release_evidence(
        config_path: Path,
        *,
        evidence_index_path: Path,
        evidence_report_path: Path,
        evidence_bundle_path: Path,
    ) -> ReleaseEvidenceResult:
        raise RuntimeError(f"failed config: {config_path}")

    monkeypatch.setattr(
        "scripts.release_evidence.run_release_evidence",
        fail_release_evidence,
    )

    assert main(["--config", "broken.yaml"]) == 1
    assert "release evidence failed" in capsys.readouterr().err


def _write_smoke_result() -> SyntheticBenchmarkResult:
    output_dir = Path("outputs/benchmark_smoke")
    config_path = output_dir / "config.yaml"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    checkpoint_path = output_dir / "checkpoints/circle.pt"
    figure_path = output_dir / "figures/circle_input.png"
    for path in (config_path, metrics_path, report_path, checkpoint_path, figure_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    metrics = _valid_metrics(
        config_path=config_path,
        metrics_path=metrics_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        figure_path=figure_path,
    )
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps({"run_id": "benchmark_smoke", "artifact_count": 5}),
        encoding="utf-8",
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
) -> dict[str, Any]:
    return {
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
            "config": str(config_path),
            "metrics": str(metrics_path),
            "report": str(report_path),
            "checkpoints": [str(checkpoint_path)],
        },
        "git_commit": "abc123",
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
                        "is_certified": True,
                        "assumptions": ["bounded tanh nonlinearity"],
                        "limitation": "Sufficient contraction certificate only.",
                    }
                },
                "baselines": {},
                "figures": {"input_trajectory": str(figure_path)},
            }
        },
        "certificate_summary": {
            "all_certified": True,
            "by_certificate": {
                "contraction": {
                    "total": 1,
                    "certified": 1,
                    "failed": 0,
                }
            },
            "failures": [],
        },
        "certificate_gate": {
            "passed": True,
            "required_certificates": ["contraction"],
            "failures": [],
        },
    }
