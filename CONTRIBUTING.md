# Contributing

**Find this project useful?** Help us make it even better by submitting any bugs or improvement
suggestions you have as GitHub Issues and Pull Requests.


## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python 3.11+.

    git clone https://github.com/adamhadani/sklearn-hierarchical-classification.git
    cd sklearn-hierarchical-classification
    uv sync --dev
    uv run pre-commit install

`pre-commit` runs ruff (lint + format), mypy and the fast test suite on every commit. CI runs the
same hooks plus the full test suite across all supported Python versions, so a change that passes
`uv run pre-commit run --all-files` locally should pass CI.


## Pull Requests

This project uses [git-flow](https://github.com/nvie/gitflow).

Please submit PRs against the `develop` branch. Add tests for behaviour changes; the existing tests
live under `sklearn_hierarchical_classification/tests/`.


## Releases

Versions are derived from git tags by setuptools-scm. Maintainers release by pushing a tag of the
form `X.Y.Z`, which publishes to PyPI and creates a GitHub Release automatically.
