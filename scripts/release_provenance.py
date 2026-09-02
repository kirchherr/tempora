from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file that must contain a top-level object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def build_release_provenance(
    *,
    release_tag: str,
    metrics: dict[str, Any],
    artifact_manifest: dict[str, Any],
    metrics_path: Path,
    artifact_manifest_path: Path,
    tag_commit: str | None,
    head_commit: str | None,
    branch: str | None,
    clean_worktree: bool | None,
) -> dict[str, Any]:
    """Build a conservative provenance record for one release smoke run."""

    run_id = _required_string(metrics, "run_id")
    metrics_git_commit = metrics.get("git_commit")
    if metrics_git_commit is not None and not isinstance(metrics_git_commit, str):
        raise ValueError("metrics.git_commit must be a string or null")

    manifest_run_id = _required_string(artifact_manifest, "run_id")
    if manifest_run_id != run_id:
        raise ValueError(
            "artifact manifest run_id does not match metrics run_id: "
            f"{manifest_run_id!r} != {run_id!r}"
        )

    artifact_count = artifact_manifest.get("artifact_count")
    if not isinstance(artifact_count, int) or artifact_count < 0:
        raise ValueError("artifact_manifest.artifact_count must be a non-negative int")

    certificate_gate = _required_mapping(metrics, "certificate_gate")
    gate_failures = certificate_gate.get("failures", [])
    if not isinstance(gate_failures, list):
        raise ValueError("certificate_gate.failures must be a list")
    required_certificates = certificate_gate.get("required_certificates", [])
    if not _is_string_list(required_certificates):
        raise ValueError("certificate_gate.required_certificates must be a string list")

    return {
        "schema": "tempora.release_provenance.v1",
        "release_tag": release_tag,
        "tag_commit": tag_commit,
        "head_commit": head_commit,
        "metrics_git_commit": metrics_git_commit,
        "tag_matches_metrics_commit": (
            tag_commit is not None
            and metrics_git_commit is not None
            and tag_commit == metrics_git_commit
        ),
        "head_matches_tag": (
            head_commit is not None
            and tag_commit is not None
            and head_commit == tag_commit
        ),
        "source_state": {
            "branch": branch,
            "clean_worktree": clean_worktree,
        },
        "artifacts": {
            "run_id": run_id,
            "metrics": str(metrics_path),
            "artifact_manifest": str(artifact_manifest_path),
            "artifact_count": artifact_count,
        },
        "certificate_gate": {
            "passed": certificate_gate.get("passed") is True,
            "required_certificates": required_certificates,
            "failure_count": len(gate_failures),
        },
        "limitations": [
            "This record links one generated release smoke run to one Git tag.",
            "It does not prove general temporal semantic preservation.",
            "It does not validate artifacts outside the referenced manifest.",
        ],
    }


def validate_release_provenance(provenance: dict[str, Any]) -> None:
    """Validate release provenance fields that should block release audit."""

    errors: list[str] = []
    if provenance.get("tag_commit") is None:
        errors.append("release tag could not be resolved")
    if provenance.get("metrics_git_commit") is None:
        errors.append("metrics.json does not record git_commit")
    if provenance.get("tag_matches_metrics_commit") is not True:
        errors.append("release tag commit does not match metrics git_commit")

    artifacts = _required_mapping(provenance, "artifacts")
    artifact_count = artifacts.get("artifact_count")
    if not isinstance(artifact_count, int) or artifact_count <= 0:
        errors.append("artifact manifest must contain at least one artifact")

    gate = _required_mapping(provenance, "certificate_gate")
    if gate.get("passed") is not True:
        errors.append("certificate gate did not pass")
    if gate.get("failure_count") != 0:
        errors.append("certificate gate records failures")

    if errors:
        raise ValueError("; ".join(errors))


def collect_git_state(tag: str, *, cwd: Path) -> dict[str, str | bool | None]:
    """Collect local Git state for release provenance context."""

    return {
        "tag_commit": _git_stdout(["rev-list", "-n", "1", tag], cwd=cwd) or None,
        "head_commit": _git_stdout(["rev-parse", "HEAD"], cwd=cwd) or None,
        "branch": _git_stdout(["branch", "--show-current"], cwd=cwd) or None,
        "clean_worktree": _git_stdout(["status", "--porcelain"], cwd=cwd) == "",
    }


def write_release_provenance(provenance: dict[str, Any], path: Path) -> Path:
    """Write a release provenance JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _git_stdout(args: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={cwd}", *args],
        check=False,
        capture_output=True,
        cwd=cwd,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return cast(dict[str, Any], value)


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write and validate TEMPORA release provenance."
    )
    parser.add_argument("--tag", required=True, help="Git release tag to audit.")
    parser.add_argument(
        "--metrics",
        default="outputs/benchmark_smoke/metrics.json",
        help="Path to release smoke metrics.json.",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/benchmark_smoke/artifact_manifest.json",
        help="Path to release smoke artifact_manifest.json.",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark_smoke/release_provenance.json",
        help="Path to write release provenance JSON.",
    )
    args = parser.parse_args(argv)

    try:
        metrics_path = Path(args.metrics)
        manifest_path = Path(args.manifest)
        git_state = collect_git_state(args.tag, cwd=Path.cwd())
        provenance = build_release_provenance(
            release_tag=args.tag,
            metrics=load_json_object(metrics_path),
            artifact_manifest=load_json_object(manifest_path),
            metrics_path=metrics_path,
            artifact_manifest_path=manifest_path,
            tag_commit=cast(str | None, git_state["tag_commit"]),
            head_commit=cast(str | None, git_state["head_commit"]),
            branch=cast(str | None, git_state["branch"]),
            clean_worktree=cast(bool | None, git_state["clean_worktree"]),
        )
        validate_release_provenance(provenance)
        write_release_provenance(provenance, Path(args.output))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"release provenance failed: {exc}", file=sys.stderr)
        return 1

    print(f"release provenance valid: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
