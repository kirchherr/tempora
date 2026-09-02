import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from scripts.release_provenance import (
    build_release_provenance,
    collect_git_state,
    main,
    validate_release_provenance,
    write_release_provenance,
)


def test_build_release_provenance_links_tag_to_metrics_commit() -> None:
    provenance = _provenance()

    assert provenance["schema"] == "tempora.release_provenance.v1"
    assert provenance["release_tag"] == "v0.1.0-alpha"
    assert provenance["tag_matches_metrics_commit"] is True
    assert provenance["head_matches_tag"] is False
    assert provenance["artifacts"]["artifact_count"] == 3
    assert provenance["certificate_gate"]["passed"] is True


def test_validate_release_provenance_accepts_matching_release_record() -> None:
    validate_release_provenance(_provenance())


def test_validate_release_provenance_rejects_commit_mismatch() -> None:
    provenance = _provenance(tag_commit="different")

    with pytest.raises(ValueError, match="does not match metrics git_commit"):
        validate_release_provenance(provenance)


def test_validate_release_provenance_rejects_failed_certificate_gate() -> None:
    provenance = _provenance(gate_passed=False)

    with pytest.raises(ValueError, match="certificate gate did not pass"):
        validate_release_provenance(provenance)


def test_write_release_provenance_writes_sorted_json(tmp_path: Path) -> None:
    output_path = tmp_path / "release_provenance.json"

    completed = write_release_provenance(_provenance(), output_path)

    assert completed == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["release_tag"] == "v0.1.0-alpha"


def test_release_provenance_cli_writes_valid_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics_path = tmp_path / "metrics.json"
    manifest_path = tmp_path / "artifact_manifest.json"
    output_path = tmp_path / "release_provenance.json"
    metrics_path.write_text(json.dumps(_metrics()), encoding="utf-8")
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.release_provenance.collect_git_state",
        lambda tag, *, cwd: {
            "tag_commit": "abc123",
            "head_commit": "abc123",
            "branch": "main",
            "clean_worktree": True,
        },
    )

    exit_code = main(
        [
            "--tag",
            "v0.1.0-alpha",
            "--metrics",
            str(metrics_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert "release provenance valid" in capsys.readouterr().out


def test_collect_git_state_marks_workspace_as_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        cwd: Path,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert check is False
        assert capture_output is True
        assert cwd == tmp_path
        assert text is True
        stdout = ""
        if "rev-list" in cmd:
            stdout = "tag-commit\n"
        elif "rev-parse" in cmd:
            stdout = "head-commit\n"
        elif "branch" in cmd:
            stdout = "main\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("scripts.release_provenance.subprocess.run", fake_run)

    state = collect_git_state("v0.1.0-alpha", cwd=tmp_path)

    assert state == {
        "tag_commit": "tag-commit",
        "head_commit": "head-commit",
        "branch": "main",
        "clean_worktree": True,
    }
    assert calls
    assert all(
        call[:3] == ["git", "-c", f"safe.directory={tmp_path}"] for call in calls
    )


def _provenance(
    *,
    tag_commit: str | None = "abc123",
    gate_passed: bool = True,
) -> dict[str, Any]:
    return build_release_provenance(
        release_tag="v0.1.0-alpha",
        metrics=_metrics(gate_passed=gate_passed),
        artifact_manifest=_manifest(),
        metrics_path=Path("outputs/benchmark_smoke/metrics.json"),
        artifact_manifest_path=Path("outputs/benchmark_smoke/artifact_manifest.json"),
        tag_commit=tag_commit,
        head_commit="post-release",
        branch="kirchherr/phase-33-release-provenance",
        clean_worktree=True,
    )


def _metrics(*, gate_passed: bool = True) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    if not gate_passed:
        failures.append({"dataset": "circle", "certificate": "contraction"})
    return {
        "run_id": "benchmark_smoke",
        "git_commit": "abc123",
        "certificate_gate": {
            "passed": gate_passed,
            "required_certificates": ["contraction", "learning_stability"],
            "failures": failures,
        },
    }


def _manifest() -> dict[str, Any]:
    return {
        "run_id": "benchmark_smoke",
        "artifact_count": 3,
        "artifacts": [
            {"path": "outputs/benchmark_smoke/metrics.json"},
            {"path": "outputs/benchmark_smoke/report.md"},
            {"path": "outputs/benchmark_smoke/artifact_manifest.json"},
        ],
    }
