---
title: Upgrading from 1.3.x
nav_order: 7
---

# Upgrading from 1.3.x

Release 1.3.x is the last one published by Globality. Later releases keep the API but change
the behaviours below.

- **Python 3.11+ and scikit-learn 1.6+** are required.
- **`fit()` no longer accepts `sample_weight`.** It was accepted but never used (weights were
  silently ignored), which current scikit-learn estimator checks reject.
- **A node's training set is selected by label only.** Previously a sample whose feature row
  was entirely zero was silently left out of every local classifier (and of the `n_samples`
  metafeature).
- **On a DAG, a sample under several children of a node is used once per child.** Previously
  it was used once per *path*, which duplicated rows and, in the feature-building step, doubled
  feature values.
- **`_select_features(X, y)` is called once per trained node** with that node's rows, and must
  keep the number of columns since prediction passes unselected rows to the local classifiers.
- **The `array` helpers `extract_rows_csr`, `nnz_rows_ix`, `apply_rollup_Xy` and
  `apply_rollup_Xy_raw`** were internal to the old per-node matrix building and were removed.
- **Multi-label `predict()` returns an indicator matrix** over `mlb.classes_` of the nodes
  visited by each sample, matching the `y` passed to `fit` and the columns of `predict_proba()`.
  It previously returned an array of root-prefixed node paths with duplicated entries.
  `predict_proba()` in that mode is now sized by `mlb.classes_`.
- **On a DAG, multi-label `predict_proba()` reports the highest local score** of a class under
  several visited parents (previously their sum), which is the quantity its threshold is
  compared with.
- **Early stopping is rejected together with `mlb`**; it was silently ignored before. A callable
  `stopping_criteria` returns `True` to stop (the code always behaved this way; the docstring
  said the opposite) and is no longer consulted at the root.
- **The default base estimator is `LogisticRegression(solver="lbfgs")`** without the removed
  `multi_class="multinomial"` argument. On scikit-learn 1.6/1.7 binary local classifiers
  therefore use one-vs-rest, so predicted probabilities (and hence `nmlnp` stopping decisions)
  can differ slightly from 1.3.x; on scikit-learn 1.8+ there is no difference.

- **`algorithm="lcn"` is deprecated** and raises a `FutureWarning` at `fit`, together with the
  training strategies reserved for it (`"exclusive"`, `"less_exclusive"`, `"less_inclusive"`,
  `"exclusive_siblings"`). It never had an implementation: a leaf node has no training data of
  its own, so the model fitted was that of `"lcpn"` with `"siblings"`, whatever the training
  strategy said. Drop both parameters to keep it. Both go away in the next major release.

New since 1.3.x, all opt-in: `training_strategy="inclusive"`, per-class
`mlb_prediction_threshold`, `mlb_min_root_predictions`, and the `thresholds` module. See
[Multi-label](multi-label).
