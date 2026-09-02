from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tempora.experiments.evidence_index import (
    EvidenceRunRecord,
    build_evidence_index,
    load_artifact_manifest_for_metrics,
    load_json_object,
    write_evidence_index,
)

try:
    from scripts.validate_metrics import validate_artifact_paths
except ModuleNotFoundError:  # pragma: no cover - used when invoked as a script path.
    from validate_metrics import validate_artifact_paths


def build_records(
    metrics_paths: Sequence[Path],
    *,
    require_manifest: bool,
    check_files: bool,
) -> tuple[EvidenceRunRecord, ...]:
    """Load benchmark metrics and optional release-smoke manifests."""

    records: list[EvidenceRunRecord] = []
    for metrics_path in metrics_paths:
        metrics = load_json_object(metrics_path)
        if check_files:
            validate_artifact_paths(metrics, base_dir=Path.cwd())
        manifest_path, manifest = load_artifact_manifest_for_metrics(
            metrics_path,
            require_manifest=require_manifest,
        )
        records.append(
            EvidenceRunRecord(
                metrics_path=metrics_path,
                metrics=metrics,
                artifact_manifest_path=manifest_path,
                artifact_manifest=manifest,
            )
        )
    return tuple(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a TEMPORA evidence index from benchmark metrics."
    )
    parser.add_argument(
        "metrics",
        nargs="+",
        help="One or more benchmark metrics.json files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/evidence_index.json",
        help="Path to write evidence index JSON.",
    )
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="Require artifact_manifest.json next to every metrics file.",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Require every artifact path referenced by metrics to exist.",
    )
    args = parser.parse_args(argv)

    try:
        records = build_records(
            tuple(Path(path) for path in args.metrics),
            require_manifest=args.require_manifest,
            check_files=args.check_files,
        )
        index = build_evidence_index(records)
        write_evidence_index(index, Path(args.output))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"evidence index failed: {exc}", file=sys.stderr)
        return 1

    print(f"evidence index written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
