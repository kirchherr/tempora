import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tempora.experiments.evidence_report import (
    expected_evidence_report_text,
    render_evidence_report,
    validate_evidence_report,
    write_evidence_report,
)

SCRIPT = Path("scripts/render_evidence_report.py")


def test_render_evidence_report_summarizes_runs_and_failures() -> None:
    report = render_evidence_report(_valid_index(optional_failure=True))

    assert "# TEMPORA Evidence Index Report" in report
    assert "It does not prove general temporal semantic preservation." in report
    assert "- runs: 1" in report
    assert "- all certificate gates passed: yes" in report
    assert "### benchmark_smoke" in report
    assert "- datasets: `circle`, `torus`" in report
    assert "- certificate summary all certified: no" in report
    assert "#### Certificate Gate Failures" in report
    assert "No failure records." in report
    assert "#### Certificate Summary Failures" in report
    assert "dataset=`torus`" in report
    assert "distance=1.2" in report


def test_render_evidence_report_handles_failed_certificate_gate() -> None:
    report = render_evidence_report(_valid_index(gate_passed=False))

    assert "- all certificate gates passed: no" in report
    assert "- certificate gate passed: no" in report
    assert "- certificate gate failure count: 1" in report
    assert "certificate=`contraction`" in report


def test_render_evidence_report_rejects_invalid_index() -> None:
    index = _valid_index()
    index["schema"] = "wrong"

    with pytest.raises(ValueError, match="schema"):
        render_evidence_report(index)


def test_write_evidence_report_writes_markdown(tmp_path: Path) -> None:
    output_path = tmp_path / "evidence_report.md"

    completed = write_evidence_report("report", output_path)

    assert completed == output_path
    assert output_path.read_text(encoding="utf-8") == "report\n"


def test_validate_evidence_report_accepts_expected_text() -> None:
    index = _valid_index()

    validate_evidence_report(expected_evidence_report_text(index), index)


def test_validate_evidence_report_rejects_stale_text() -> None:
    index = _valid_index()
    report = expected_evidence_report_text(index).replace("- runs: 1", "- runs: 2")

    with pytest.raises(ValueError, match="does not match"):
        validate_evidence_report(report, index)


def test_render_evidence_report_cli_writes_report(tmp_path: Path) -> None:
    index_path = tmp_path / "evidence_index.json"
    report_path = tmp_path / "evidence_report.md"
    index_path.write_text(json.dumps(_valid_index()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(index_path),
            "--output",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "evidence report written" in completed.stdout
    assert report_path.exists()
    assert "TEMPORA Evidence Index Report" in report_path.read_text(encoding="utf-8")


def _valid_index(
    *,
    gate_passed: bool = True,
    optional_failure: bool = False,
) -> dict[str, Any]:
    gate_failures: list[dict[str, Any]] = []
    if not gate_passed:
        gate_failures.append(
            {
                "dataset": "circle",
                "certificate": "contraction",
                "theorem": "theorem_01_sufficient_contraction",
            }
        )

    summary_failures = copy.deepcopy(gate_failures)
    if optional_failure:
        summary_failures.append(
            {
                "dataset": "torus",
                "certificate": "topology_comparison",
                "theorem": "theorem_03_empirical_persistence_diagram_comparison",
                "metric": "bottleneck",
                "homology_dim": 1,
                "distance": 1.2,
                "max_distance": 1.0,
            }
        )

    return {
        "schema": "tempora.evidence_index.v1",
        "run_count": 1,
        "dataset_count": 2,
        "git_commits": ["abc123"],
        "all_certificate_gates_passed": gate_passed,
        "runs": [
            {
                "run_id": "benchmark_smoke",
                "seed": 42,
                "metrics_path": "outputs/benchmark_smoke/metrics.json",
                "git_commit": "abc123",
                "datasets": ["circle", "torus"],
                "dataset_count": 2,
                "artifact_manifest": "outputs/benchmark_smoke/artifact_manifest.json",
                "artifact_count": 3,
                "certificate_types": ["contraction", "topology_comparison"],
                "certificate_summary": {
                    "all_certified": not summary_failures,
                    "failure_count": len(summary_failures),
                    "failures": summary_failures,
                },
                "certificate_gate": {
                    "passed": gate_passed,
                    "required_certificates": ["contraction"],
                    "failure_count": len(gate_failures),
                    "failures": gate_failures,
                },
            }
        ],
        "limitations": [
            "This index summarizes generated benchmark artifacts for review.",
            "It does not prove general temporal semantic preservation.",
        ],
    }
