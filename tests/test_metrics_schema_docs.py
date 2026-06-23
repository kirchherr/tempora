from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_schema_doc_mentions_required_fields() -> None:
    text = (ROOT / "docs" / "experiments" / "metrics_schema.md").read_text(
        encoding="utf-8"
    )
    required_terms = (
        "metrics.json",
        "run_id",
        "seed",
        "config",
        "artifacts",
        "git_commit",
        "dependency_versions",
        "runtime",
        "datasets",
        "certificate_summary",
        "certificate_gate",
        "required_certificates",
        "topology_max_distance",
        "contraction",
        "learning_stability",
        "topology_comparison",
    )

    for term in required_terms:
        assert term in text


def test_readme_links_metrics_schema_doc() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/experiments/metrics_schema.md" in text


def test_benchmark_spec_links_metrics_schema_doc() -> None:
    text = (ROOT / "docs" / "experiments" / "benchmark_spec.md").read_text(
        encoding="utf-8"
    )

    assert "metrics_schema.md" in text
