from pathlib import Path

ROOT = Path(".")


def test_release_files_exist() -> None:
    for path in (
        "CHANGELOG.md",
        "CITATION.cff",
        "LICENSE.md",
        "docs/release_v0_1.md",
    ):
        assert (ROOT / path).exists(), f"missing release file: {path}"


def test_release_version_metadata_is_aligned() -> None:
    assert 'version = "0.1.0a0"' in (ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '__version__ = "0.1.0a0"' in (ROOT / "src/tempora/__init__.py").read_text(
        encoding="utf-8"
    )
    assert 'version: "0.1.0-alpha"' in (ROOT / "CITATION.cff").read_text(
        encoding="utf-8"
    )


def test_readme_contains_release_setup_and_smoke_commands() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    required = (
        "python -m pip install -r requirements/docker-cpu.txt",
        "docker compose run --rm tempora",
        "python scripts/train_synth.py --config configs/benchmark_smoke.yaml",
        "docs/release_v0_1.md",
        "LICENSE.md",
        "No invented benchmark results",
    )
    for needle in required:
        assert needle in text


def test_outputs_are_ignored_except_gitkeep() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "outputs/*" in text
    assert "!outputs/.gitkeep" in text


def test_release_docs_do_not_embed_benchmark_results() -> None:
    checked_paths = (
        ROOT / "CHANGELOG.md",
        ROOT / "docs/release_v0_1.md",
        ROOT / "docs/experiments/results_template.md",
    )
    forbidden_phrases = (
        "we achieved",
        "state of the art",
        "outperforms",
        "proves general",
        "benchmark result:",
    )
    for path in checked_paths:
        lowered = path.read_text(encoding="utf-8").lower()
        for phrase in forbidden_phrases:
            assert phrase not in lowered, f"{path} contains release overclaim: {phrase}"
