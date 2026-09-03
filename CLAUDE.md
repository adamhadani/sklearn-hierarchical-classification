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
`_train_local_classifiers` visits nodes in depth-first order from the root (deterministic, so
seeded base estimators reproduce). A node's training set is the samples labeled with any
*strict* descendant (samples labeled with the node itself belong to its parent's set), taken
from a label → row-indices dict built once for either target encoding and sliced out of `X`
by index through `_rows`, which is the one place that knows raw `X` is a Python list.
Row sets are unions, so a DAG node with two parents contributes its rows once to each
ancestor; on a DAG a sample under several children is repeated by index, so the input type
(dense or sparse) is preserved at every node. Rows are chosen by label only: an all-zero
feature row is still a training sample. The fitted model keeps no reference to the training
data; `graph_.nodes[n]` holds `metafeatures`, `classifier` and, in multi-label mode, the
`trained_classes` learned there. Multi-label (`mlb`) works in
both modes: the indicator `y` is densified up front and validated with `multi_output=True`.
Rolled-up children unknown to the binarizer give all-zero rows, which are dropped; a node
left with none gets no classifier and a warning. The exception is
`training_strategy="inclusive"` (mlb only): every document outside the subtree joins the
node's training set as an all-zero row on purpose, so the local classifier can reject
documents a parent mis-routes to it. Metafeatures still describe the subtree.
`mlb_prediction_threshold` may be an array aligned with `mlb.classes_` (per-class cut-offs
tuned on out-of-fold scores, validated again at predict since they are usually set after fit);
predicting with `-inf` visits every node *learned at fit* and yields the score matrix for
such tuning, which `thresholds.scut_thresholds` and `thresholds.routed_thresholds` consume. On a
DAG that matrix reports the maximum over visited parents, which is exactly what routing compares
to the threshold. Tune per class *locally* (pass `graph`): sibling-trained local classifiers give
meaningless scores to out-of-subtree samples, and a global SCut on the all-node matrix gets worse
than no tuning (RCV1, siblings, C=1, plain OOF micro-F1: global 0.819, none 0.827, local 0.838).
`routed_thresholds` beats local SCut on micro-F1 (RCV1, inclusive, C=0.5, 2-fold cross-tuned
OOF: 0.846 vs 0.839) at equal macro, but only with enough held-out positives per class (on
GermEval every per-class scheme loses to one scalar threshold). `thresholds.route` is the one
emulation of the multi-label walk on a score matrix and tests pin it to `predict` (exact on
trees, a superset on DAGs): compare threshold policies through it, never by re-deriving the walk
in a script. Per-node choice of base classifier family or C was measured and gives nothing on
TF-IDF text (spike, Sept 2026): do not add a selector for it. `mlb_min_root_predictions` forces the
best-scoring root children for samples that would otherwise get no top-level label. In
multi-label mode a node only routes to children that had a positive example in its training
set (`graph_.nodes[n][TRAINED_CLASSES]`): one-vs-rest gives an unlearned class a constant
predictor with decision value 0, which any negative threshold would select for every sample.

**`rollup_nodes` uses a descendant map, not `all_simple_paths`.** `children_by_descendant`
maps every strict descendant of the source to the children it lies under, so each child is
listed once per target even when several paths run through it, and targets outside the
subtree (including the source itself) roll up to `[]`. Roll-up runs once per *distinct* label at a node (`np.unique` + inverse), so
nothing loops over samples in Python during training.

**Every node classifier is scored through `_local_scores`**, which wraps a raw sample as a
length-1 batch and normalises `decision_function` / `predict_proba` output to a 1-D array
aligned with `clf.classes_` (a binary `decision_function` returns one signed score, which
is expanded to two). Index `probs[local_class_idx]` only; never reintroduce
mode-specific indexing in `_recursive_predict`.

**Prediction scores each node once per call, in topological order.** `_predict_top_down`
keeps an inbox of row indices per node; a node is scored in one `_local_scores` call on the
union of rows that reached it from *all* parents, so DAGs get one call per node too
(guarded by `test_predict_scores_each_local_classifier_once_per_call`). Only `predict_proba`
allocates the score matrix. In multi-label mode a node's classifier may report scores for
columns that are not its children (ancestors, siblings); those are recorded but never
routed, which is what keeps the walk finite. A callable `stopping_criteria` is the only
per-sample Python in prediction; it returns True to *stop*, is never consulted at the root,
and early stopping is rejected together with `mlb` at fit time.

**`__sklearn_clone__` passes a fitted `mlb` through unchanged.** scikit-learn's `clone` would
re-instantiate the binarizer unfitted (it is a constructor parameter), and `fit` reads
`mlb.classes_`, so every clone-based tool (grid search, `cross_val_predict`, the tuning loop in the
docs) would fail in multi-label mode. Do not "simplify" the override away. `fit` also validates the
hierarchy (root present, acyclic; unreachable nodes warn) and warns about labels of `y` outside it,
which used to be dropped silently; tests rely on the warning, not an error, because fitting on a
label subset is a legitimate use.

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
tree; do not add one. Tags are
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

**`docs/` is a Jekyll site** (just-the-docs remote theme) built and deployed to GitHub Pages by
`publish-docs.yml` on pushes to `develop` that touch it; the repository's Pages source must be
set to "GitHub Actions" once. Pages are hand-written Markdown, so the API reference and the
parameter tables in `docs/` are not generated: a signature or behaviour change in the package
must be mirrored there (and in the README's benchmark summary when numbers change).

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
