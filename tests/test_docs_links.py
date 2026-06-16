from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(".")
DOC_ROOTS = (
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "LICENSE.md",
    ROOT / "AGENTS.md",
    ROOT / "docs",
)
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def test_local_markdown_links_resolve() -> None:
    failures: list[str] = []

    for markdown_path in _markdown_files():
        text = markdown_path.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = _clean_target(raw_target)
            if _is_external_or_non_file_target(target):
                continue
            linked_path, anchor = _split_anchor(target)
            resolved = (markdown_path.parent / unquote(linked_path)).resolve()
            if not resolved.exists():
                failures.append(f"{markdown_path}: missing link target {target}")
                continue
            if anchor and resolved.suffix.lower() == ".md":
                anchors = _markdown_anchors(resolved)
                if anchor not in anchors:
                    failures.append(f"{markdown_path}: missing anchor {target}")

    assert not failures, "\n".join(failures)


def _markdown_files() -> list[Path]:
    files: list[Path] = []
    for root in DOC_ROOTS:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.md")))
    return sorted(files)


def _clean_target(raw_target: str) -> str:
    target = raw_target.strip()
    if " " in target:
        target = target.split()[0]
    return target.strip("<>")


def _is_external_or_non_file_target(target: str) -> bool:
    parsed = urlparse(target)
    return bool(parsed.scheme) or target.startswith("#") or target.startswith("mailto:")


def _split_anchor(target: str) -> tuple[str, str | None]:
    linked_path, separator, anchor = target.partition("#")
    return linked_path, anchor if separator else None


def _markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := HEADING_PATTERN.match(line):
            anchors.add(_github_heading_slug(match.group(1)))
    return anchors


def _github_heading_slug(heading: str) -> str:
    slug = heading.strip().lower()
    slug = re.sub(r"`([^`]*)`", r"\1", slug)
    slug = re.sub(r"[^\w\s-]", "", slug, flags=re.UNICODE)
    slug = re.sub(r"\s+", "-", slug)
    return slug.strip("-")
