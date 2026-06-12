from __future__ import annotations

import argparse
from pathlib import Path

from tempora.experiments.evaluate_stability import (
    DEFAULT_DATASETS,
    run_stability_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TEMPORA synthetic stability evaluation."
    )
    parser.add_argument(
        "--run-id", default="phase5_smoke", help="Output run identifier."
    )
    parser.add_argument(
        "--output-root", default="outputs", help="Output root directory."
    )
    parser.add_argument("--seed", type=int, default=42, help="Base deterministic seed.")
    parser.add_argument("--n-steps", type=int, default=96, help="Samples per dataset.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
        help="Datasets to evaluate.",
    )
    args = parser.parse_args()
    result = run_stability_evaluation(
        run_id=args.run_id,
        output_root=Path(args.output_root),
        seed=args.seed,
        dataset_names=tuple(args.datasets),
        n_steps=args.n_steps,
    )
    print(result.metrics_path)


if __name__ == "__main__":
    main()
