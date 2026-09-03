import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.validate_evidence_report import validate_evidence_report_file

from tempora.experiments.evidence_report import expected_evidence_report_text

SCRIPT = Path("scripts/validate_evidence_report.py")


def test_validate_evidence_report_file_accepts_current_report(tmp_path: Path) -> None:
    index = _valid_index()
    index_path = tmp_path / "evidence_index.json"
    report_path = tmp_path / "evidence_report.md"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    report_path.write_text(expected_evidence_report_text(index), encoding="utf-8")

    validate_evidence_report_file(index_path, report_path)


def test_validate_evidence_report_file_rejects_stale_report(tmp_path: Path) -> None:
    index = _valid_index()
    index_path = tmp_path / "evidence_index.json"
    report_path = tmp_path / "evidence_report.md"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    report_path.write_text("# stale\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        validate_evidence_report_file(index_path, report_path)


def test_validate_evidence_report_cli_accepts_current_report(tmp_path: Path) -> None:
    index = _valid_index()
    index_path = tmp_path / "evidence_index.json"
    report_path = tmp_path / "evidence_report.md"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    report_path.write_text(expected_evidence_report_text(index), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(index_path), str(report_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "evidence report valid" in completed.stdout


def test_validate_evidence_report_cli_rejects_stale_report(tmp_path: Path) -> None:
    index_path = tmp_path / "evidence_index.json"
    report_path = tmp_path / "evidence_report.md"
    index_path.write_text(json.dumps(_valid_index()), encoding="utf-8")
    report_path.write_text("# stale\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(index_path), str(report_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "does not match" in completed.stderr


def _valid_index() -> dict[str, Any]:
    return {
        "schema": "tempora.evidence_index.v1",
        "run_count": 1,
        "dataset_count": 1,
        "git_commits": ["abc123"],
        "all_certificate_gates_passed": True,
        "runs": [
            {
                "run_id": "benchmark_smoke",
                "seed": 42,
                "metrics_path": "outputs/benchmark_smoke/metrics.json",
                "git_commit": "abc123",
                "datasets": ["circle"],
                "dataset_count": 1,
                "artifact_manifest": "outputs/benchmark_smoke/artifact_manifest.json",
                "artifact_count": 3,
                "certificate_types": ["contraction"],
                "certificate_summary": {
                    "all_certified": True,
                    "failure_count": 0,
                    "failures": [],
                },
                "certificate_gate": {
                    "passed": True,
                    "required_certificates": ["contraction"],
                    "failure_count": 0,
                    "failures": [],
                },
            }
        ],
        "limitations": [
            "This index summarizes generated benchmark artifacts for review.",
            "It does not prove general temporal semantic preservation.",
        ],
    }
