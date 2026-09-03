---
title: Development
nav_order: 8
---

# Development
{: .no_toc }

1. TOC
{:toc}

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python 3.11+.
Every tool's configuration lives in `pyproject.toml`; linting, formatting, type checking and
the tests are enforced through [pre-commit](https://pre-commit.com/), locally and in CI.

```bash
git clone https://github.com/promptromp/sklearn-hierarchical-classification.git
cd sklearn-hierarchical-classification
uv sync --dev
uv run pre-commit install
```

```bash
uv run pytest -m "not slow"         # what the pre-commit hook runs
uv run pytest --cov --cov-branch    # full suite, as CI runs it
uv run pre-commit run --all-files   # everything CI enforces: ruff, codespell, mypy, hygiene hooks, tests
```

The public functions and methods are annotated and the package ships a `py.typed` marker;
mypy checks every function body (`check_untyped_defs`), annotated or not.

Tests live inside the package, under `sklearn_hierarchical_classification/tests/`, and are
excluded from the wheel. `pytest.mark.slow` marks the tests that download a dataset. Deprecation
warnings fail the suite, so a deprecated scikit-learn, numpy or networkx call is caught before
the API disappears, and total (statement and branch) coverage below 95% fails every
`pytest --cov` run. `codespell` checks the tracked text files as one of the pre-commit hooks.

## Pull requests

The `develop` branch is the integration branch (git-flow); open pull requests against it.
CI lints with the pre-commit configuration, runs the test suite on Python 3.11 through 3.14
and once more on 3.11 against the oldest releases the dependency lower bounds in
`pyproject.toml` admit, and builds the wheel and installs it into an environment without the
dev dependencies to check that it imports, fits, ships its `py.typed` marker and leaves the tests out. Add a test for every
behaviour change. Dependabot opens weekly pull requests for the GitHub Actions and for the
packages in `uv.lock`, minor and patch bumps grouped, majors one at a time.

## Benchmarks

`benchmarks/` holds scripts that are not part of the package:

- `bench.py` times `fit` and `predict` on synthetic sparse data and hierarchies of
  configurable shape;
- `rcv1_benchmark.py` runs RCV1-v2 (fetched through scikit-learn);
- `germeval2019_benchmark.py` and `bgc_benchmark.py` run the two book-blurb datasets
  (downloaded on first use) through the features and protocol shared in `blurbs.py`.

Results and protocol are on the [Benchmarks](benchmarks) page. Keep to the protocol when
adding a configuration: every choice is made on out-of-fold or development data and the test
set is scored once per pre-registered configuration.

## Documentation

This site is built by Jekyll from `docs/` with the
[just-the-docs](https://just-the-docs.com/) theme and published to GitHub Pages by
`.github/workflows/publish-docs.yml` on every push to `develop` that touches `docs/`.
Pages are plain Markdown with front matter (`title`, `nav_order`). To preview locally:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

## Releasing

Versions come from git tags through setuptools-scm; there is no version string in the tree.
Pushing a tag of the form `X.Y.Z` builds the distributions, publishes them to PyPI through
trusted publishing, and creates a GitHub Release:

```bash
git tag 1.4.0
git push origin 1.4.0
```
