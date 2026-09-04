import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.validate_evidence_bundle import validate_evidence_bundle_file

from tempora.experiments.evidence_bundle import (
    build_evidence_bundle_manifest,
    write_evidence_bundle_manifest,
)
from tempora.experiments.evidence_report import expected_evidence_report_text

SCRIPT = Path("scripts/validate_evidence_bundle.py").resolve()


def test_validate_evidence_bundle_file_accepts_current_bundle(
    tmp_path: Path,
) -> None:
    bundle_path = _write_bundle(tmp_path)

    manifest = validate_evidence_bundle_file(bundle_path, base_dir=tmp_path)

    assert manifest["schema"] == "tempora.evidence_bundle.v1"


def test_validate_evidence_bundle_file_rejects_stale_bundle(tmp_path: Path) -> None:
    bundle_path = _write_bundle(tmp_path)
    report_path = tmp_path / "outputs" / "evidence_report.md"
    report_path.write_text("# stale\n", encoding="utf-8")

    with pytest.raises(ValueError, match="byte count|sha256"):
        validate_evidence_bundle_file(bundle_path, base_dir=tmp_path)


def test_validate_evidence_bundle_cli_accepts_current_bundle(tmp_path: Path) -> None:
    bundle_path = _write_bundle(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(bundle_path)],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert completed.returncode == 0
    assert "evidence bundle valid" in completed.stdout


def test_validate_evidence_bundle_cli_rejects_stale_bundle(tmp_path: Path) -> None:
    bundle_path = _write_bundle(tmp_path)
    report_path = tmp_path / "outputs" / "evidence_report.md"
    report_path.write_text("# stale\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(bundle_path)],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )

    assert completed.returncode == 1
    assert "evidence bundle invalid" in completed.stderr


def _write_bundle(tmp_path: Path) -> Path:
    index = _valid_index()
    index_path = tmp_path / "outputs" / "evidence_index.json"
    report_path = tmp_path / "outputs" / "evidence_report.md"
    bundle_path = tmp_path / "outputs" / "evidence_bundle.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(expected_evidence_report_text(index), encoding="utf-8")
    manifest = build_evidence_bundle_manifest(
        Path("outputs/evidence_index.json"),
        Path("outputs/evidence_report.md"),
        base_dir=tmp_path,
    )
    write_evidence_bundle_manifest(manifest, bundle_path)
    return bundle_path


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
