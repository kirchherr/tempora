from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from tempora.experiments.run_synthetic import validate_benchmark_metrics


def collect_artifact_paths(metrics: dict[str, Any]) -> tuple[Path, ...]:
    """Collect file paths referenced by a benchmark metrics payload."""

    paths: list[Path] = []
    artifacts = metrics.get("artifacts", {})
    if isinstance(artifacts, dict):
        for key in ("config", "metrics", "report"):
            value = artifacts.get(key)
            if isinstance(value, str):
                paths.append(Path(value))
        checkpoints = artifacts.get("checkpoints", [])
        if isinstance(checkpoints, list):
            paths.extend(Path(value) for value in checkpoints if isinstance(value, str))

    datasets = metrics.get("datasets", {})
    if isinstance(datasets, dict):
        for dataset_payload in datasets.values():
            if not isinstance(dataset_payload, dict):
                continue
            checkpoint = dataset_payload.get("checkpoint")
            if isinstance(checkpoint, str):
                paths.append(Path(checkpoint))
            figures = dataset_payload.get("figures", {})
            if isinstance(figures, dict):
                paths.extend(
                    Path(value) for value in figures.values() if isinstance(value, str)
                )

    return tuple(dict.fromkeys(paths))


def validate_artifact_paths(
    metrics: dict[str, Any],
    *,
    base_dir: Path,
) -> None:
    """Validate that artifact files referenced by metrics exist."""

    missing: list[str] = []
    for artifact_path in collect_artifact_paths(metrics):
        candidate = (
            artifact_path if artifact_path.is_absolute() else base_dir / artifact_path
        )
        if not candidate.exists():
            missing.append(str(artifact_path))
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"metrics artifact path(s) missing: {missing_text}")


def load_metrics(path: Path) -> dict[str, Any]:
    """Load a benchmark metrics payload from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metrics payload must be a JSON object.")
    return cast(dict[str, Any], payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate TEMPORA metrics.json.")
    parser.add_argument("metrics", help="Path to benchmark metrics.json.")
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Require every artifact path referenced by metrics to exist.",
    )
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics)
    try:
        metrics = load_metrics(metrics_path)
        validate_benchmark_metrics(metrics)
        if args.check_files:
            validate_artifact_paths(metrics, base_dir=Path.cwd())
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"metrics schema invalid: {exc}", file=sys.stderr)
        return 1
    print(f"metrics schema valid: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
