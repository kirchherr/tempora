from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tempora.experiments.evidence_bundle import validate_evidence_bundle_artifacts
from tempora.experiments.evidence_index import load_json_object


def validate_evidence_bundle_file(path: Path, *, base_dir: Path) -> dict[str, object]:
    """Load and validate an evidence bundle manifest against local artifacts."""

    manifest = load_json_object(path)
    validate_evidence_bundle_artifacts(manifest, base_dir=base_dir)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a TEMPORA evidence bundle manifest."
    )
    parser.add_argument(
        "bundle",
        nargs="?",
        default="outputs/evidence_bundle.json",
        help="Path to evidence_bundle.json.",
    )
    args = parser.parse_args(argv)

    try:
        validate_evidence_bundle_file(Path(args.bundle), base_dir=Path.cwd())
    except (OSError, ValueError) as exc:
        print(f"evidence bundle invalid: {exc}", file=sys.stderr)
        return 1

    print(f"evidence bundle valid: {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
