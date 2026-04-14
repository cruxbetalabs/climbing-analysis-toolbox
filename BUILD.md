```shell
# local testing
pip install -e .
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The local hook compares `[project].version` in `pyproject.toml` against your
tracked remote branch and blocks commits that touch published package files if
the staged version is not higher.

```shell
# venv -a climbing @ local macbook venv cli
rm -rf dist build *.egg-info
python -m build
twine upload dist/*
```