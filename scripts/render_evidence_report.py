from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tempora.experiments.evidence_index import load_json_object
from tempora.experiments.evidence_report import (
    render_evidence_report,
    write_evidence_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a Markdown TEMPORA evidence report from an evidence index."
    )
    parser.add_argument("index", help="Path to evidence_index.json.")
    parser.add_argument(
        "--output",
        default="outputs/evidence_report.md",
        help="Path to write evidence report Markdown.",
    )
    args = parser.parse_args(argv)

    try:
        report = render_evidence_report(load_json_object(Path(args.index)))
        write_evidence_report(report, Path(args.output))
    except (OSError, ValueError) as exc:
        print(f"evidence report failed: {exc}", file=sys.stderr)
        return 1

    print(f"evidence report written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
