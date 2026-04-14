#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path


CATEGORY_ORDER = ("changed", "updated", "added", "improved")
CATEGORY_TITLES = {
    "changed": "What's Changed",
    "updated": "What's Updated",
    "added": "What's Added",
    "improved": "What's Improved",
}
TAG_VERSION_PATTERN = re.compile(r"^v?(\d+(?:\.\d+)*)$")
CONVENTIONAL_PREFIX_PATTERN = re.compile(r"^[a-z]+(?:\([^)]+\))?!?:\s*", re.IGNORECASE)
CONVENTIONAL_TYPE_PATTERN = re.compile(
    r"^(?P<type>[a-z]+)(?:\([^)]+\))?!?:\s*", re.IGNORECASE
)
VERSION_ONLY_COMMIT_PATTERNS = (
    re.compile(r"\b(?:bump|update) version\b", re.IGNORECASE),
    re.compile(r"\brelease v?\d", re.IGNORECASE),
)


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"git {' '.join(args)} failed")
    return result


def _decode(output: bytes) -> str:
    return output.decode("utf-8", errors="replace").strip()


def _read_version_from_pyproject() -> str:
    pyproject_path = Path("pyproject.toml")
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def _parse_version(value: str) -> tuple[int, ...]:
    match = TAG_VERSION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Unsupported version/tag format: {value}")
    return tuple(int(part) for part in match.group(1).split("."))


def _find_previous_tag(current_version: str) -> str | None:
    current_tuple = _parse_version(current_version)
    result = _run_git("tag", "--merged", "HEAD")
    candidates: list[tuple[tuple[int, ...], str]] = []
    for raw_tag in _decode(result.stdout).splitlines():
        tag = raw_tag.strip()
        if not tag:
            continue
        try:
            parsed = _parse_version(tag)
        except ValueError:
            continue
        if parsed < current_tuple:
            candidates.append((parsed, tag))

    if not candidates:
        return None

    candidates.sort()
    return candidates[-1][1]


def _commit_range(previous_tag: str | None) -> str:
    if previous_tag is None:
        return "HEAD"
    return f"{previous_tag}..HEAD"


def _list_commit_subjects(previous_tag: str | None) -> list[str]:
    git_range = _commit_range(previous_tag)
    result = _run_git("log", "--format=%s", "--reverse", "--no-merges", git_range)
    return [line for line in _decode(result.stdout).splitlines() if line]


def _list_changed_files(previous_tag: str | None) -> list[str]:
    git_range = _commit_range(previous_tag)
    result = _run_git("diff", "--name-only", git_range)
    return [line for line in _decode(result.stdout).splitlines() if line]


def _is_version_only_commit(subject: str) -> bool:
    lowered = subject.lower()
    if "pyproject.toml" not in lowered and "version" not in lowered:
        return False
    return any(pattern.search(subject) for pattern in VERSION_ONLY_COMMIT_PATTERNS)


def _normalize_subject(subject: str) -> str:
    cleaned = CONVENTIONAL_PREFIX_PATTERN.sub("", subject).strip()
    if not cleaned:
        cleaned = subject.strip()
    if cleaned and cleaned[-1] == ".":
        cleaned = cleaned[:-1]
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _classify_subject(subject: str) -> str:
    lowered = subject.lower()
    conventional_match = CONVENTIONAL_TYPE_PATTERN.match(subject.strip())
    if conventional_match is not None:
        commit_type = conventional_match.group("type").lower()
        if commit_type == "feat":
            return "added"
        if commit_type in {"refactor", "perf"}:
            return "improved"
        if commit_type in {"docs", "chore", "build", "ci", "style"}:
            return "updated"
        if commit_type == "fix":
            return "changed"

    if re.search(r"\b(feat|feature|add|added|new|introduce|introduced|support|supported)\b", lowered):
        return "added"
    if re.search(r"\b(improve|improved|enhance|enhanced|optimi[sz]e|optimized|refactor|cleanup|clean up|readability|performance)\b", lowered):
        return "improved"
    if re.search(r"\b(update|updated|upgrade|upgraded|docs|documentation|rename|renamed|replace|replaced|remove|removed|delete|deleted|publish|published)\b", lowered):
        return "updated"
    return "changed"


def _categorize_subjects(subjects: list[str]) -> dict[str, list[str]]:
    categories = {name: [] for name in CATEGORY_ORDER}
    seen: set[tuple[str, str]] = set()

    for subject in subjects:
        if _is_version_only_commit(subject):
            continue
        category = _classify_subject(subject)
        normalized = _normalize_subject(subject)
        if not normalized:
            continue
        dedupe_key = (category, normalized)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        categories[category].append(normalized)

    return categories


def _format_changed_files(files: list[str]) -> list[str]:
    if not files:
        return ["- No files changed in this release range."]

    lines = [f"- {path}" for path in files[:15]]
    remaining = len(files) - 15
    if remaining > 0:
        lines.append(f"- ... and {remaining} more files")
    return lines


def _render_release_notes(version: str, previous_tag: str | None) -> str:
    subjects = _list_commit_subjects(previous_tag)
    changed_files = _list_changed_files(previous_tag)
    categories = _categorize_subjects(subjects)

    lines = [f"## Release v{version}", ""]
    if previous_tag is not None:
        lines.append(f"Based on changes since `{previous_tag}`.")
    else:
        lines.append("Based on all changes currently available in the repository history.")
    lines.append("")

    for category_name in CATEGORY_ORDER:
        lines.append(f"### {CATEGORY_TITLES[category_name]}")
        entries = categories[category_name]
        if entries:
            lines.extend(f"- {entry}" for entry in entries)
        else:
            lines.append("- No developer-facing items were classified in this section.")
        lines.append("")

    lines.append("### Developer Notes")
    lines.append(f"- Commit range: `{_commit_range(previous_tag)}`")
    lines.append("- Files touched:")
    lines.extend(_format_changed_files(changed_files))
    lines.append("")
    lines.append("### Install")
    lines.append("```shell")
    lines.append(f"pip install cruxes=={version}")
    lines.append("```")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate categorized release notes from git history."
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Release version. Defaults to project.version in pyproject.toml.",
    )
    args = parser.parse_args()

    try:
        version = args.version or _read_version_from_pyproject()
        previous_tag = _find_previous_tag(version)
        sys.stdout.write(_render_release_notes(version, previous_tag))
        return 0
    except (KeyError, OSError, RuntimeError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"Failed to generate release notes: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())