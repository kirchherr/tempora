import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tempora.experiments.evidence_index import (
    EvidenceRunRecord,
    build_evidence_index,
    build_evidence_run_summary,
    load_artifact_manifest_for_metrics,
    validate_evidence_index,
    write_evidence_index,
)

SCRIPT = Path("scripts/build_evidence_index.py")


def test_build_evidence_index_summarizes_run_and_artifact_manifest() -> None:
    record = EvidenceRunRecord(
        metrics_path=Path("outputs/benchmark_smoke/metrics.json"),
        metrics=_valid_metrics(),
        artifact_manifest_path=Path("outputs/benchmark_smoke/artifact_manifest.json"),
        artifact_manifest=_artifact_manifest(),
    )

    index = build_evidence_index((record,))

    assert index["schema"] == "tempora.evidence_index.v1"
    assert index["run_count"] == 1
    assert index["dataset_count"] == 1
    assert index["git_commits"] == ["abc123"]
    assert index["all_certificate_gates_passed"] is True
    run = index["runs"][0]
    assert run["run_id"] == "benchmark_smoke"
    assert run["datasets"] == ["circle"]
    assert run["artifact_count"] == 3
    assert run["certificate_types"] == ["contraction"]
    assert run["certificate_gate"]["required_certificates"] == ["contraction"]


def test_build_evidence_run_summary_accepts_missing_manifest() -> None:
    summary = build_evidence_run_summary(
        EvidenceRunRecord(
            metrics_path=Path("outputs/benchmark_smoke/metrics.json"),
            metrics=_valid_metrics(),
        )
    )

    assert summary["artifact_manifest"] is None
    assert summary["artifact_count"] is None


def test_build_evidence_index_records_failed_certificate_gate() -> None:
    metrics = _valid_metrics(gate_passed=False)
    record = EvidenceRunRecord(
        metrics_path=Path("outputs/benchmark_smoke/metrics.json"),
        metrics=metrics,
    )

    index = build_evidence_index((record,))

    assert index["all_certificate_gates_passed"] is False
    assert index["runs"][0]["certificate_gate"]["failure_count"] == 1


def test_build_evidence_index_rejects_manifest_run_id_mismatch() -> None:
    manifest = _artifact_manifest()
    manifest["run_id"] = "other"
    record = EvidenceRunRecord(
        metrics_path=Path("outputs/benchmark_smoke/metrics.json"),
        metrics=_valid_metrics(),
        artifact_manifest_path=Path("outputs/benchmark_smoke/artifact_manifest.json"),
        artifact_manifest=manifest,
    )

    with pytest.raises(ValueError, match="does not match metrics run_id"):
        build_evidence_index((record,))


def test_validate_evidence_index_rejects_mismatched_counts() -> None:
    index = build_evidence_index(
        (
            EvidenceRunRecord(
                metrics_path=Path("outputs/benchmark_smoke/metrics.json"),
                metrics=_valid_metrics(),
            ),
        )
    )
    index["dataset_count"] = 99

    with pytest.raises(ValueError, match="dataset_count"):
        validate_evidence_index(index)


def test_load_artifact_manifest_for_metrics_respects_requirement(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_valid_metrics()), encoding="utf-8")

    assert load_artifact_manifest_for_metrics(metrics_path) == (None, None)
    with pytest.raises(FileNotFoundError, match="artifact manifest"):
        load_artifact_manifest_for_metrics(metrics_path, require_manifest=True)


def test_write_evidence_index_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "evidence_index.json"
    index = build_evidence_index(
        (
            EvidenceRunRecord(
                metrics_path=Path("outputs/benchmark_smoke/metrics.json"),
                metrics=_valid_metrics(),
            ),
        )
    )

    completed = write_evidence_index(index, output_path)

    assert completed == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "tempora.evidence_index.v1"


def test_build_evidence_index_cli_writes_index_with_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark_smoke"
    output_dir.mkdir()
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "artifact_manifest.json"
    index_path = tmp_path / "evidence_index.json"
    metrics_path.write_text(json.dumps(_valid_metrics()), encoding="utf-8")
    manifest_path.write_text(json.dumps(_artifact_manifest()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(metrics_path),
            "--require-manifest",
            "--output",
            str(index_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "evidence index written" in completed.stdout
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["run_count"] == 1


def test_build_evidence_index_cli_rejects_missing_manifest(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_valid_metrics()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(metrics_path),
            "--require-manifest",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "artifact manifest" in completed.stderr


def _artifact_manifest() -> dict[str, Any]:
    return {
        "run_id": "benchmark_smoke",
        "artifact_count": 3,
        "artifacts": [
            {"path": "outputs/benchmark_smoke/config.yaml"},
            {"path": "outputs/benchmark_smoke/metrics.json"},
            {"path": "outputs/benchmark_smoke/report.md"},
        ],
    }


def _valid_metrics(*, gate_passed: bool = True) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    if not gate_passed:
        failures.append(
            {
                "dataset": "circle",
                "certificate": "contraction",
                "theorem": "theorem_01_sufficient_contraction",
            }
        )
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
                            "contraction_margin": 0.6 if gate_passed else 0.01,
                            "required_margin": 0.05,
                            "is_certified": gate_passed,
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
                "all_certified": gate_passed,
                "by_certificate": {
                    "contraction": {
                        "total": 1,
                        "certified": 1 if gate_passed else 0,
                        "failed": 0 if gate_passed else 1,
                    }
                },
                "failures": failures,
            },
            "certificate_gate": {
                "passed": gate_passed,
                "required_certificates": ["contraction"],
                "failures": failures,
            },
        }
    )
