from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tempora.experiments.evidence_bundle import (
    build_evidence_bundle_manifest,
    write_evidence_bundle_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a checksum manifest for TEMPORA evidence artifacts."
    )
    parser.add_argument(
        "--index",
        default="outputs/evidence_index.json",
        help="Path to evidence_index.json.",
    )
    parser.add_argument(
        "--report",
        default="outputs/evidence_report.md",
        help="Path to evidence_report.md.",
    )
    parser.add_argument(
        "--output",
        default="outputs/evidence_bundle.json",
        help="Path to write evidence bundle JSON.",
    )
    args = parser.parse_args(argv)

    try:
        manifest = build_evidence_bundle_manifest(
            Path(args.index),
            Path(args.report),
            base_dir=Path.cwd(),
        )
        write_evidence_bundle_manifest(manifest, Path(args.output))
    except (OSError, ValueError) as exc:
        print(f"evidence bundle failed: {exc}", file=sys.stderr)
        return 1

    print(f"evidence bundle written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
