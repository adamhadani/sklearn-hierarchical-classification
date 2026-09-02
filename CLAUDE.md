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

**Training selects rows by label; it never materialises per-node feature matrices.**
`_train_local_classifiers` visits every node reachable from the root once. A node's training
set is the samples labeled with any *strict* descendant (samples labeled with the node itself
belong to its parent's set), found through `_rows_by_label` (a label → row-indices dict built
once) and `networkx.descendants`, then sliced out of `X` by index. Row sets are unions, so a
DAG node with two parents contributes its rows once to each ancestor. The fitted model keeps
no reference to the training data; `graph_.nodes[n]` holds only `metafeatures` and
`classifier`. `feature_extraction="raw"` follows the same path with `X` as a Python list
(`[X[i] for i in rows]`), so nothing may call `.shape` on it. Multi-label (`mlb`) only works
in raw mode: `validate_data` rejects a 2-D indicator `y` in preprocessed mode, and the row
selection for `mlb` goes through the indicator columns of `mlb.classes_`.

**`rollup_nodes` uses a descendant map, not `all_simple_paths`.** `children_by_descendant`
maps every strict descendant of the source to the children it lies under, so each child is
listed once per target even when several paths run through it, and targets outside the
subtree (including the source itself) roll up to `[]`. `apply_rollup_Xy` duplicates rows with
one `np.repeat` fancy-index on CSR; do not reintroduce Python loops over rows or nonzeros in
`array.py`.

**Every node classifier is scored through `_local_scores`**, which wraps a raw sample as a
length-1 batch and normalises `decision_function` / `predict_proba` output to a 1-D array
aligned with `clf.classes_` (a binary `decision_function` returns one signed score, which
is expanded to two). Index `probs[local_class_idx]` only; never reintroduce
mode-specific indexing in `_recursive_predict`.

**Tree vs DAG hierarchies roll up differently** in `_train_local_classifier`: on a tree
the rolled-up labels are flattened; on a DAG a sample can belong to several children so
rows are duplicated via `apply_rollup_Xy(_raw)`. `is_tree_` is computed once in `fit`.

**Single-target nodes fall back to `DummyClassifier(strategy="constant")`**, and the
constant is wrapped as a 1-element list because scikit-learn's parameter validation
rejects numpy scalars for `constant`.

**Deprecation warnings are errors in tests, globally.** `filterwarnings` in `pyproject.toml`
turns every `DeprecationWarning`/`FutureWarning` into a failure, so a deprecated
scikit-learn/numpy/networkx call fails CI before the API is removed. The filter has to be
global: scikit-learn attributes its deprecation warnings to `sklearn.*`, so a
package-scoped filter never fires. Known third-party noise (e.g. scikit-learn's dataset
loaders under numpy 2.5) gets an explicit `ignore` entry next to it; add to that list
rather than weakening the `error` entries.

**Tests live inside the package** (`sklearn_hierarchical_classification/tests/`) and are
excluded from the wheel via `packages.find.exclude` plus `include-package-data = false`.
The latter is required: with setuptools-scm's file finder, the default would pull the
git-tracked test files back in as package data. `examples/classify_digits.py` imports
`tests.fixtures`, so it only runs from a source checkout.

**`pytest.mark.slow`** marks the 20newsgroups multi-label test (downloads ~14MB, ~30s).
The pre-commit hook runs `-m "not slow"`; CI runs everything with the dataset directory
cached.

**Version comes from git tags only** (setuptools-scm). There is no version string in the
tree; do not add one (`docs/source/conf.py` reads it from package metadata). Tags are
plain `X.Y.Z` (no `v` prefix), matching the existing tag history, and only tags of that
exact shape trigger the publish jobs in `publish-to-pypi.yml`, which needs a PyPI trusted
publisher configured for this repo and a `pypi` GitHub environment.

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
