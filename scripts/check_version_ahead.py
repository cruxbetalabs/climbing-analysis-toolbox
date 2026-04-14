#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tomllib

from packaging.version import InvalidVersion, Version


PREFERRED_BASE_REFS = ("origin/main", "origin/master", "main", "master")


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


def _assert_git_repo() -> None:
    _run_git("rev-parse", "--show-toplevel")


def _get_staged_files() -> list[str]:
    result = _run_git(
        "diff", "--cached", "--name-only", "--diff-filter=ACMR", check=False
    )
    if result.returncode != 0:
        return []
    return [line for line in _decode(result.stdout).splitlines() if line]


def _touches_published_package(paths: list[str]) -> bool:
    for path in paths:
        if path == "pyproject.toml" or path == "setup.py" or path.startswith("src/"):
            return True
    return False


def _read_version_from_toml_bytes(toml_bytes: bytes, source_name: str) -> Version:
    try:
        payload = tomllib.loads(toml_bytes.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Could not parse {source_name}: {exc}") from exc

    try:
        raw_version = payload["project"]["version"]
    except KeyError as exc:
        raise RuntimeError(f"{source_name} does not define project.version") from exc

    try:
        return Version(str(raw_version))
    except InvalidVersion as exc:
        raise RuntimeError(
            f"{source_name} has a non-PEP 440 version: {raw_version}"
        ) from exc


def _read_index_version() -> Version:
    result = _run_git("show", ":pyproject.toml", check=False)
    if result.returncode != 0:
        raise RuntimeError("pyproject.toml is not available in the git index")
    return _read_version_from_toml_bytes(result.stdout, "staged pyproject.toml")


def _resolve_base_ref() -> str | None:
    for ref in PREFERRED_BASE_REFS:
        exists = _run_git("rev-parse", "--verify", "--quiet", ref, check=False)
        if exists.returncode == 0:
            return ref

    upstream = _run_git(
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    )
    if upstream.returncode == 0:
        return _decode(upstream.stdout)

    return None


def _read_remote_version(remote_ref: str) -> Version:
    result = _run_git("show", f"{remote_ref}:pyproject.toml", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Could not read pyproject.toml from {remote_ref}")
    return _read_version_from_toml_bytes(result.stdout, f"{remote_ref}:pyproject.toml")


def main() -> int:
    try:
        _assert_git_repo()
        staged_files = _get_staged_files()
        if not _touches_published_package(staged_files):
            return 0

        base_ref = _resolve_base_ref()
        if base_ref is None:
            print(
                "Skipping version check: no main/master or upstream reference is available yet.",
                file=sys.stderr,
            )
            return 0

        local_version = _read_index_version()
        remote_version = _read_remote_version(base_ref)

        if local_version <= remote_version:
            print(
                "Version check failed: staged pyproject.toml must be higher than "
                f"{base_ref} when commit content touches publishable package files.",
                file=sys.stderr,
            )
            print(
                f"Staged version: {local_version}",
                file=sys.stderr,
            )
            print(
                f"{base_ref} version: {remote_version}",
                file=sys.stderr,
            )
            print(
                "Bump [project].version in pyproject.toml before committing, or refresh your remote-tracking refs with 'git fetch origin' if they are stale.",
                file=sys.stderr,
            )
            return 1

        return 0
    except RuntimeError as exc:
        print(f"Version check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
