from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from tempora.experiments.evidence_index import load_json_object, validate_evidence_index


def validate_evidence_index_file(
    path: Path,
    *,
    require_gates_passed: bool = False,
    require_manifests: bool = False,
    require_git_commits: bool = False,
    check_files: bool = False,
    base_dir: Path,
) -> dict[str, Any]:
    """Load and validate an evidence index with optional release-review gates."""

    index = load_json_object(path)
    validate_evidence_index(index)

    errors: list[str] = []
    if require_gates_passed and index.get("all_certificate_gates_passed") is not True:
        errors.append("not all certificate gates passed")

    runs = cast(list[Any], index["runs"])
    for position, run_value in enumerate(runs):
        if not isinstance(run_value, dict):
            errors.append(f"runs[{position}] must be an object")
            continue
        run = cast(dict[str, Any], run_value)
        run_id = str(run.get("run_id", f"runs[{position}]"))
        _check_run(
            run,
            run_id=run_id,
            require_manifests=require_manifests,
            require_git_commits=require_git_commits,
            check_files=check_files,
            base_dir=base_dir,
            errors=errors,
        )

    if errors:
        raise ValueError("; ".join(errors))
    return index


def _check_run(
    run: dict[str, Any],
    *,
    run_id: str,
    require_manifests: bool,
    require_git_commits: bool,
    check_files: bool,
    base_dir: Path,
    errors: list[str],
) -> None:
    if require_git_commits and not _non_empty_string(run.get("git_commit")):
        errors.append(f"{run_id}: missing git_commit")
    artifact_manifest = run.get("artifact_manifest")
    if require_manifests and not _non_empty_string(artifact_manifest):
        errors.append(f"{run_id}: missing artifact_manifest")
    if check_files:
        _require_existing_file(
            run.get("metrics_path"),
            run_id=run_id,
            field="metrics_path",
            base_dir=base_dir,
            errors=errors,
        )
        if artifact_manifest is not None:
            _require_existing_file(
                artifact_manifest,
                run_id=run_id,
                field="artifact_manifest",
                base_dir=base_dir,
                errors=errors,
            )


def _require_existing_file(
    value: object,
    *,
    run_id: str,
    field: str,
    base_dir: Path,
    errors: list[str],
) -> None:
    if not _non_empty_string(value):
        errors.append(f"{run_id}: {field} must be a non-empty path")
        return
    path = Path(value)
    candidate = path if path.is_absolute() else base_dir / path
    if not candidate.exists():
        errors.append(f"{run_id}: {field} missing: {path}")


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a TEMPORA evidence index.")
    parser.add_argument("index", help="Path to evidence_index.json.")
    parser.add_argument(
        "--require-gates-passed",
        action="store_true",
        help="Require all indexed certificate gates to have passed.",
    )
    parser.add_argument(
        "--require-manifests",
        action="store_true",
        help="Require every indexed run to reference an artifact manifest.",
    )
    parser.add_argument(
        "--require-git-commits",
        action="store_true",
        help="Require every indexed run to record a git commit.",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Require indexed metrics and manifest files to exist.",
    )
    args = parser.parse_args(argv)

    try:
        validate_evidence_index_file(
            Path(args.index),
            require_gates_passed=args.require_gates_passed,
            require_manifests=args.require_manifests,
            require_git_commits=args.require_git_commits,
            check_files=args.check_files,
            base_dir=Path.cwd(),
        )
    except (OSError, ValueError) as exc:
        print(f"evidence index invalid: {exc}", file=sys.stderr)
        return 1

    print(f"evidence index valid: {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
