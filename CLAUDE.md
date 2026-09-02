# CLAUDE.md

Guidance for Claude Code working in this repository. Keep it short and spend the
space on gotchas; anything discoverable by reading the code does not belong here.

## What this is

A scikit-learn compatible meta-estimator (`HierarchicalClassifier`) for hierarchical
classification: given a class hierarchy as a tree/DAG (`networkx.DiGraph` or adjacency
dict, rooted at the `ROOT` sentinel), it trains a local classifier per parent node and
predicts top-down. `metrics.py` provides hierarchical precision/recall/F-beta. The
package is small and flat: `classifier.py` is the core, `array.py`/`graph.py` are
numpy/scipy/networkx helpers, `validation.py` checks parameter consistency.

## Gotchas

**`test_estimator_inteface` runs scikit-learn's full `check_estimator` suite** (55+
checks against current scikit-learn). This is what forces the estimator contract:
`validate_data(...)` in `fit`/`predict`/`predict_proba` (sets `n_features_in_`), the
mixin order `MetaEstimatorMixin, ClassifierMixin, BaseEstimator`, `__sklearn_tags__`
declaring `input_tags.sparse = True`, and no `sample_weight` parameter on `fit`
(it was previously accepted and silently ignored, which the checks now reject). Do not
"simplify" any of these away.

**Two feature-extraction modes share one code path.** `feature_extraction="preprocessed"`
(default) expects a 2-D feature matrix and works in CSR sparse throughout; `"raw"`
expects a Python sequence of raw samples (e.g. text) and the base estimator must be a
`Pipeline` that does its own feature extraction. In raw mode `X` is a list, so nothing
may call `.shape` on it; the raw-mode helpers (`apply_rollup_Xy_raw`, `_build_features`)
are the places this has gone wrong before.

**Tree vs DAG hierarchies roll up differently** in `_train_local_classifier`: on a tree
the rolled-up labels are flattened; on a DAG a sample can belong to several children so
rows are duplicated via `apply_rollup_Xy(_raw)`. `is_tree_` is computed once in `fit`.

**`rollup_nodes` must skip single-node paths.** networkx >= 3.0 makes
`all_simple_paths(G, s, s)` yield `[[s]]` (older versions yielded nothing), so a sample
labeled with the node currently being trained produces a path with no `path[1]`. The
`len(path) > 1` guards there are load-bearing; this only shows up in the slow
multi-label newsgroups test.

**Single-target nodes fall back to `DummyClassifier(strategy="constant")`**, and the
constant is wrapped as a 1-element list because scikit-learn's parameter validation
rejects numpy scalars for `constant`.

**Warnings from this package are errors in tests.** `filterwarnings` in `pyproject.toml`
turns `DeprecationWarning`/`FutureWarning` attributed to `sklearn_hierarchical_classification`
into failures, so a deprecated scikit-learn/numpy/networkx call in library code fails CI
before the API is removed. Warnings raised inside third-party code (e.g. scikit-learn's own
dataset loaders under new numpy) are deliberately not caught.

**Tests live inside the package** (`sklearn_hierarchical_classification/tests/`) and are
excluded from the wheel via `packages.find.exclude` plus `include-package-data = false`.
The latter is required: with setuptools-scm's file finder, the default would pull the
git-tracked test files back in as package data. `examples/classify_digits.py` imports
`tests.fixtures`, so it only runs from a source checkout.

**`pytest.mark.slow`** marks the 20newsgroups multi-label test (downloads ~14MB, ~30s).
The pre-commit hook runs `-m "not slow"`; CI runs everything with the dataset directory
cached.

**Version comes from git tags only** (setuptools-scm). There is no version string in the
tree; do not add one. Tags are plain `X.Y.Z` (no `v` prefix), matching the existing tag
history. Pushing a tag triggers `publish-to-pypi.yml`, which needs a PyPI trusted publisher
configured for this repo and a `pypi` GitHub environment.

**Tooling versions come from `uv.lock` only.** Ruff/mypy/pytest run in pre-commit as
`local`/`system` hooks invoking `uv run ...`, and CI's lint job runs
`uv run pre-commit run --all-files` with `SKIP=pytest`. `.pre-commit-config.yaml` is the
single source of truth for what is enforced. Upgrade with `uv lock --upgrade`; new checks
go in the pre-commit config, never as bare workflow steps.

**Ruff rule selection uses `extend-select`, not `select`**, layering on top of ruff's
default rule set. All tool config lives in `pyproject.toml`; do not reintroduce
`setup.cfg`, `tox.ini`, `.flake8` or similar.

**`docs/` is a legacy Sphinx setup** published manually from the `gh-pages` branch. It is
not built in CI and its `conf.py` is only linted (with `A001` ignored for `copyright`).

## Commands

```bash
uv sync --dev

# Matches what pre-commit runs (excludes slow tests)
uv run pytest -m "not slow"

# Full suite, as CI runs it
uv run pytest --cov --cov-branch

# Everything CI enforces
uv run pre-commit run --all-files

# Build distributions (version derived from git tag)
uv build
```

Python 3.11+; CI matrix covers 3.11 through 3.14. `develop` is the integration branch
(git-flow); PRs target `develop`.
