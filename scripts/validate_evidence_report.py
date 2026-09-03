from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tempora.experiments.evidence_index import load_json_object
from tempora.experiments.evidence_report import validate_evidence_report


def validate_evidence_report_file(index_path: Path, report_path: Path) -> None:
    """Validate that a rendered Markdown report matches an evidence index file."""

    index = load_json_object(index_path)
    validate_evidence_report(report_path.read_text(encoding="utf-8"), index)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a rendered TEMPORA evidence report against its index."
    )
    parser.add_argument("index", help="Path to evidence_index.json.")
    parser.add_argument("report", help="Path to evidence_report.md.")
    args = parser.parse_args(argv)

    try:
        validate_evidence_report_file(Path(args.index), Path(args.report))
    except (OSError, ValueError) as exc:
        print(f"evidence report invalid: {exc}", file=sys.stderr)
        return 1

    print(f"evidence report valid: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
