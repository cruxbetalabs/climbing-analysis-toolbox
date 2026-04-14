```shell
# local testing
pip install -e .
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This repo uses `pre-commit` to run a local release guard before each commit.

Relevant docs:
- https://pre-commit.com/

The configured hook checks the staged `[project].version` in `pyproject.toml`
against the version in your tracked remote branch and blocks the commit if all
of the following are true:
- the commit touches `pyproject.toml`, `setup.py`, or anything under `src/`
- `pyproject.toml` is staged
- the staged package version is not strictly higher than the remote version

Notes:
- The hook reads the version from the git index, so it validates what is staged,
	not just what is in your working tree.
- The remote comparison uses your branch upstream when available, then falls
	back to `origin/main` or `origin/master`.
- If your remote-tracking refs are stale, refresh them with `git fetch origin`.
- If there is no upstream or matching remote ref yet, the hook skips the check.

Useful commands:

```shell
# run hooks for the currently staged changes
pre-commit run

# run just the version guard
pre-commit run version-ahead-of-remote --all-files

# refresh remote refs before a release-related commit
git fetch origin
```

```shell
# venv -a climbing @ local macbook venv cli
rm -rf dist build *.egg-info
python -m build
twine upload dist/*
```