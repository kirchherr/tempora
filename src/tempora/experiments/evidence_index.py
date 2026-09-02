"""Build compact evidence indexes from benchmark metrics artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tempora.experiments.run_synthetic import validate_benchmark_metrics


@dataclass(frozen=True)
class EvidenceRunRecord:
    """Input artifacts for one benchmark run to include in an evidence index."""

    metrics_path: Path
    metrics: dict[str, Any]
    artifact_manifest_path: Path | None = None
    artifact_manifest: dict[str, Any] | None = None


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON artifact that must contain a top-level object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return cast(dict[str, Any], payload)


def load_artifact_manifest_for_metrics(
    metrics_path: Path,
    *,
    manifest_name: str = "artifact_manifest.json",
    require_manifest: bool = False,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Load a release-smoke artifact manifest stored next to metrics if present."""

    manifest_path = metrics_path.parent / manifest_name
    if not manifest_path.exists():
        if require_manifest:
            raise FileNotFoundError(f"missing artifact manifest: {manifest_path}")
        return None, None
    return manifest_path, load_json_object(manifest_path)


def build_evidence_index(records: Sequence[EvidenceRunRecord]) -> dict[str, Any]:
    """Build a compact, review-oriented index from benchmark metrics artifacts."""

    if not records:
        raise ValueError("at least one evidence run is required.")

    runs = [build_evidence_run_summary(record) for record in records]
    dataset_count = sum(_required_int(run, "dataset_count") for run in runs)
    gate_failures = [
        run
        for run in runs
        if _required_mapping(run, "certificate_gate").get("passed") is not True
    ]
    git_commits = sorted(
        {
            git_commit
            for run in runs
            if isinstance(git_commit := run.get("git_commit"), str) and git_commit
        }
    )
    index = {
        "schema": "tempora.evidence_index.v1",
        "run_count": len(runs),
        "dataset_count": dataset_count,
        "git_commits": git_commits,
        "all_certificate_gates_passed": not gate_failures,
        "runs": runs,
        "limitations": [
            "This index summarizes generated benchmark artifacts for review.",
            "It does not prove general temporal semantic preservation.",
            "It does not compare against unrecorded or unpublished runs.",
        ],
    }
    validate_evidence_index(index)
    return index


def build_evidence_run_summary(record: EvidenceRunRecord) -> dict[str, Any]:
    """Summarize one benchmark metrics payload for evidence review."""

    validate_benchmark_metrics(record.metrics)

    run_id = _required_string(record.metrics, "run_id")
    datasets = _required_mapping(record.metrics, "datasets")
    dataset_names = sorted(_non_empty_string_keys(datasets, "metrics.datasets"))
    gate = _required_mapping(record.metrics, "certificate_gate")
    certificate_summary = _required_mapping(record.metrics, "certificate_summary")
    by_certificate = _required_mapping(certificate_summary, "by_certificate")
    certificate_types = sorted(
        _non_empty_string_keys(by_certificate, "certificate_summary.by_certificate")
    )

    artifact_count: int | None = None
    if record.artifact_manifest is not None:
        manifest_run_id = _required_string(record.artifact_manifest, "run_id")
        if manifest_run_id != run_id:
            raise ValueError(
                "artifact manifest run_id does not match metrics run_id: "
                f"{manifest_run_id!r} != {run_id!r}"
            )
        artifact_count = _required_non_negative_int(
            record.artifact_manifest,
            "artifact_count",
        )

    return {
        "run_id": run_id,
        "seed": _required_int(record.metrics, "seed"),
        "metrics_path": str(record.metrics_path),
        "git_commit": record.metrics.get("git_commit"),
        "datasets": dataset_names,
        "dataset_count": len(dataset_names),
        "artifact_manifest": (
            str(record.artifact_manifest_path)
            if record.artifact_manifest_path is not None
            else None
        ),
        "artifact_count": artifact_count,
        "certificate_types": certificate_types,
        "certificate_summary": {
            "all_certified": certificate_summary.get("all_certified") is True,
            "failure_count": len(_required_list(certificate_summary, "failures")),
        },
        "certificate_gate": {
            "passed": gate.get("passed") is True,
            "required_certificates": _required_string_list(
                gate,
                "required_certificates",
            ),
            "failure_count": len(_required_list(gate, "failures")),
        },
    }


def validate_evidence_index(index: dict[str, Any]) -> None:
    """Validate internal consistency of an evidence index payload."""

    if index.get("schema") != "tempora.evidence_index.v1":
        raise ValueError("evidence index schema must be tempora.evidence_index.v1.")
    runs = _required_list(index, "runs")
    run_count = _required_int(index, "run_count")
    if run_count != len(runs):
        raise ValueError("evidence index run_count does not match runs length.")
    dataset_count = _required_int(index, "dataset_count")
    observed_dataset_count = 0
    observed_gate_statuses: list[bool] = []
    for run in runs:
        run_payload = _ensure_mapping(run, "runs[]")
        observed_dataset_count += _required_int(run_payload, "dataset_count")
        gate = _required_mapping(run_payload, "certificate_gate")
        observed_gate_statuses.append(gate.get("passed") is True)
        artifact_count = run_payload.get("artifact_count")
        if artifact_count is not None and (
            not isinstance(artifact_count, int) or artifact_count < 0
        ):
            raise ValueError("run artifact_count must be null or a non-negative int.")
    if dataset_count != observed_dataset_count:
        raise ValueError("evidence index dataset_count does not match run summaries.")
    expected_gate_status = all(observed_gate_statuses)
    if index.get("all_certificate_gates_passed") is not expected_gate_status:
        raise ValueError("all_certificate_gates_passed does not match run summaries.")


def write_evidence_index(index: dict[str, Any], path: Path) -> Path:
    """Write an evidence index JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer.")
    return value


def _required_non_negative_int(payload: dict[str, Any], key: str) -> int:
    value = _required_int(payload, key)
    if value < 0:
        raise ValueError(f"{key} must be non-negative.")
    return value


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return _ensure_mapping(payload.get(key), key)


def _ensure_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object.")
    return cast(dict[str, Any], value)


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON list.")
    return value


def _required_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = _required_list(payload, key)
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{key} must contain only non-empty strings.")
    return cast(list[str], value)


def _non_empty_string_keys(payload: dict[str, Any], location: str) -> list[str]:
    keys = list(payload)
    if not all(isinstance(key, str) and key for key in keys):
        raise ValueError(f"{location} keys must be non-empty strings.")
    return keys
