from pathlib import Path

THEORY_DIR = Path("docs/theory")
THEOREM_FILES = (
    THEORY_DIR / "theorem_01_contraction.md",
    THEORY_DIR / "theorem_02_learning_stability.md",
    THEORY_DIR / "theorem_03_tda_preservation.md",
)
REQUIRED_SECTIONS = (
    "## Statement",
    "## Assumptions",
    "## Proof Sketch",
    "## Implementation Correspondence",
    "## Empirical Checks",
    "## Limitations",
    "## Related Tests",
)
FORBIDDEN_POSITIVE_CLAIMS = (
    "proves general temporal semantics",
    "proves semantic preservation",
    "proves homeomorphism",
    "guarantees semantic equivalence",
    "guarantees real-world understanding",
)


def test_theory_documents_have_required_structure() -> None:
    for path in THEOREM_FILES:
        text = path.read_text(encoding="utf-8")
        for section in REQUIRED_SECTIONS:
            assert section in text, f"{path} is missing {section}"
        assert "tests/" in text, f"{path} must reference related tests"


def test_theory_documents_avoid_positive_overclaims() -> None:
    for path in THEOREM_FILES:
        lowered = path.read_text(encoding="utf-8").lower()
        for claim in FORBIDDEN_POSITIVE_CLAIMS:
            assert claim not in lowered, f"{path} contains overclaim: {claim}"


def test_assumptions_index_lists_all_theorems() -> None:
    text = (THEORY_DIR / "assumptions.md").read_text(encoding="utf-8")
    for path in THEOREM_FILES:
        assert path.name in text
