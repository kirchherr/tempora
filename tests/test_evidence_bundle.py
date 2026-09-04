import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tempora.experiments.evidence_bundle import (
    build_evidence_bundle_manifest,
    validate_evidence_bundle_artifacts,
    validate_evidence_bundle_files,
    validate_evidence_bundle_manifest,
    write_evidence_bundle_manifest,
)
from tempora.experiments.evidence_report import expected_evidence_report_text

SCRIPT = Path("scripts/build_evidence_bundle.py")


def test_build_evidence_bundle_manifest_records_report_hashes(tmp_path: Path) -> None:
    index_path, report_path = _write_evidence_pair(tmp_path)

    manifest = build_evidence_bundle_manifest(
        index_path.relative_to(tmp_path),
        report_path.relative_to(tmp_path),
        base_dir=tmp_path,
    )

    assert manifest["schema"] == "tempora.evidence_bundle.v1"
    assert manifest["run_count"] == 1
    assert manifest["dataset_count"] == 1
    assert manifest["run_ids"] == ["benchmark_smoke"]
    assert manifest["git_commits"] == ["abc123"]
    assert manifest["all_certificate_gates_passed"] is True
    assert manifest["evidence_index"]["bytes"] > 0
    assert len(manifest["evidence_index"]["sha256"]) == 64
    assert len(manifest["evidence_report"]["sha256"]) == 64
    validate_evidence_bundle_manifest(manifest)
    validate_evidence_bundle_files(manifest, base_dir=tmp_path)
    validate_evidence_bundle_artifacts(manifest, base_dir=tmp_path)


def test_build_evidence_bundle_manifest_rejects_stale_report(tmp_path: Path) -> None:
    index_path, report_path = _write_evidence_pair(tmp_path)
    report_path.write_text("# stale\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        build_evidence_bundle_manifest(
            index_path.relative_to(tmp_path),
            report_path.relative_to(tmp_path),
            base_dir=tmp_path,
        )


def test_validate_evidence_bundle_files_rejects_hash_mismatch(tmp_path: Path) -> None:
    index_path, report_path = _write_evidence_pair(tmp_path)
    manifest = build_evidence_bundle_manifest(
        index_path.relative_to(tmp_path),
        report_path.relative_to(tmp_path),
        base_dir=tmp_path,
    )
    report_path.write_text(
        expected_evidence_report_text(_valid_index("changed")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="byte count|sha256"):
        validate_evidence_bundle_files(manifest, base_dir=tmp_path)


def test_validate_evidence_bundle_artifacts_rejects_summary_mismatch(
    tmp_path: Path,
) -> None:
    index_path, report_path = _write_evidence_pair(tmp_path)
    manifest = build_evidence_bundle_manifest(
        index_path.relative_to(tmp_path),
        report_path.relative_to(tmp_path),
        base_dir=tmp_path,
    )
    manifest["run_ids"] = ["other"]

    with pytest.raises(ValueError, match="run_ids"):
        validate_evidence_bundle_artifacts(manifest, base_dir=tmp_path)


def test_write_evidence_bundle_manifest_writes_sorted_json(tmp_path: Path) -> None:
    index_path, report_path = _write_evidence_pair(tmp_path)
    manifest = build_evidence_bundle_manifest(
        index_path.relative_to(tmp_path),
        report_path.relative_to(tmp_path),
        base_dir=tmp_path,
    )
    output_path = tmp_path / "outputs" / "evidence_bundle.json"

    completed = write_evidence_bundle_manifest(manifest, output_path)

    assert completed == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "tempora.evidence_bundle.v1"


def test_build_evidence_bundle_cli_writes_manifest(tmp_path: Path) -> None:
    index_path, report_path = _write_evidence_pair(tmp_path)
    output_path = tmp_path / "outputs" / "evidence_bundle.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--index",
            str(index_path),
            "--report",
            str(report_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "evidence bundle written" in completed.stdout
    assert output_path.exists()


def test_build_evidence_bundle_cli_rejects_stale_report(tmp_path: Path) -> None:
    index_path, report_path = _write_evidence_pair(tmp_path)
    report_path.write_text("# stale\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--index",
            str(index_path),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "does not match" in completed.stderr


def _write_evidence_pair(tmp_path: Path) -> tuple[Path, Path]:
    index = _valid_index()
    index_path = tmp_path / "outputs" / "evidence_index.json"
    report_path = tmp_path / "outputs" / "evidence_report.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(expected_evidence_report_text(index), encoding="utf-8")
    return index_path, report_path


def _valid_index(git_commit: str = "abc123") -> dict[str, Any]:
    index = {
        "schema": "tempora.evidence_index.v1",
        "run_count": 1,
        "dataset_count": 1,
        "git_commits": [git_commit],
        "all_certificate_gates_passed": True,
        "runs": [
            {
                "run_id": "benchmark_smoke",
                "seed": 42,
                "metrics_path": "outputs/benchmark_smoke/metrics.json",
                "git_commit": git_commit,
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
    return copy.deepcopy(index)
