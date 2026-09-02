import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.validate_evidence_index import validate_evidence_index_file

SCRIPT = Path("scripts/validate_evidence_index.py")


def test_validate_evidence_index_file_accepts_release_review_index(
    tmp_path: Path,
) -> None:
    index_path = _write_index_with_files(tmp_path, _valid_index())

    index = validate_evidence_index_file(
        index_path,
        require_gates_passed=True,
        require_manifests=True,
        require_git_commits=True,
        check_files=True,
        base_dir=tmp_path,
    )

    assert index["schema"] == "tempora.evidence_index.v1"


def test_validate_evidence_index_file_rejects_failed_gate(tmp_path: Path) -> None:
    index = _valid_index()
    index["all_certificate_gates_passed"] = False
    index["runs"][0]["certificate_gate"]["passed"] = False
    index["runs"][0]["certificate_gate"]["failure_count"] = 1
    index_path = _write_index_with_files(tmp_path, index)

    with pytest.raises(ValueError, match="not all certificate gates passed"):
        validate_evidence_index_file(
            index_path,
            require_gates_passed=True,
            base_dir=tmp_path,
        )


def test_validate_evidence_index_file_rejects_missing_manifest_reference(
    tmp_path: Path,
) -> None:
    index = _valid_index()
    index["runs"][0]["artifact_manifest"] = None
    index["runs"][0]["artifact_count"] = None
    index_path = _write_index_with_files(tmp_path, index, write_manifest=False)

    with pytest.raises(ValueError, match="missing artifact_manifest"):
        validate_evidence_index_file(
            index_path,
            require_manifests=True,
            base_dir=tmp_path,
        )


def test_validate_evidence_index_file_rejects_missing_git_commit(
    tmp_path: Path,
) -> None:
    index = _valid_index()
    index["git_commits"] = []
    index["runs"][0]["git_commit"] = None
    index_path = _write_index_with_files(tmp_path, index)

    with pytest.raises(ValueError, match="missing git_commit"):
        validate_evidence_index_file(
            index_path,
            require_git_commits=True,
            base_dir=tmp_path,
        )


def test_validate_evidence_index_file_rejects_missing_referenced_file(
    tmp_path: Path,
) -> None:
    index_path = _write_index_with_files(tmp_path, _valid_index(), write_manifest=False)

    with pytest.raises(ValueError, match="artifact_manifest missing"):
        validate_evidence_index_file(index_path, check_files=True, base_dir=tmp_path)


def test_validate_evidence_index_cli_accepts_release_review_index(
    tmp_path: Path,
) -> None:
    index_path = _write_index_with_files(tmp_path, _valid_index())

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(index_path),
            "--require-gates-passed",
            "--require-manifests",
            "--require-git-commits",
            "--check-files",
        ],
        check=False,
        capture_output=True,
        cwd=Path.cwd(),
        text=True,
    )

    assert completed.returncode == 0
    assert "evidence index valid" in completed.stdout


def test_validate_evidence_index_cli_rejects_release_review_index(
    tmp_path: Path,
) -> None:
    index = _valid_index()
    index["runs"][0]["git_commit"] = None
    index_path = _write_index_with_files(tmp_path, index)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(index_path),
            "--require-git-commits",
        ],
        check=False,
        capture_output=True,
        cwd=Path.cwd(),
        text=True,
    )

    assert completed.returncode == 1
    assert "missing git_commit" in completed.stderr


def _write_index_with_files(
    tmp_path: Path,
    index: dict[str, Any],
    *,
    write_manifest: bool = True,
) -> Path:
    metrics_path = tmp_path / "outputs" / "benchmark_smoke" / "metrics.json"
    manifest_path = tmp_path / "outputs" / "benchmark_smoke" / "artifact_manifest.json"
    index_path = tmp_path / "outputs" / "evidence_index.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")
    if write_manifest:
        manifest_path.write_text("{}", encoding="utf-8")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(index)
    payload["runs"][0]["metrics_path"] = str(metrics_path)
    if payload["runs"][0]["artifact_manifest"] is not None:
        payload["runs"][0]["artifact_manifest"] = str(manifest_path)
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    return index_path


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
                },
                "certificate_gate": {
                    "passed": True,
                    "required_certificates": ["contraction"],
                    "failure_count": 0,
                },
            }
        ],
        "limitations": [
            "This index summarizes generated benchmark artifacts for review.",
            "It does not prove general temporal semantic preservation.",
        ],
    }
