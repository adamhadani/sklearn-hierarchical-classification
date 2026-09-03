---
title: Benchmarks
nav_order: 5
---

# Benchmarks
{: .no_toc }

1. TOC
{:toc}

Two public multi-label text classification benchmarks with published results, run by the
scripts in `benchmarks/`. Every tuned setting is chosen on out-of-fold scores of the training
set or on the development split, and the test set is scored once per configuration that was
fixed beforehand.

## RCV1-v2

Reuters Corpus Volume 1 (Lewis et al., JMLR 2004) as distributed by scikit-learn: 804,414
newswire stories as TF-IDF vectors with 47,236 features, 103 topic codes in a 4-root
hierarchy, labels expanded to include every ancestor. The LYRL2004 split has 23,149 training
and 781,265 test documents. The published reference is an SVM with per-category tuned SCut
thresholds: micro-F1 0.816, macro-F1 0.607.

```bash
uv run python benchmarks/rcv1_benchmark.py --tune --C 0.5
```

Results on a laptop with `LinearSVC` local classifiers (LCPN: one local classifier per parent
node):

| Model | micro-F1 | macro-F1 | hF1 | fit | predict (781k docs) |
|---|---:|---:|---:|---:|---:|
| Flat `OneVsRest(LinearSVC)`, threshold 0 | 0.804 | 0.486 | 0.808 | 3.7 s | 5.7 s |
| LCPN, siblings-trained nodes, threshold 0 (`--training-strategy siblings --min-root 0`) | 0.796 | 0.514 | 0.796 | 2.0 s | 2.3 s |
| LCPN, siblings + per-class local SCut from CV (`... --tune --thresholds scut`) | 0.792 | 0.595 | 0.792 | 10.8 s | 2.4 s |
| LCPN, inclusive-trained nodes + routed thresholds from CV, root fallback (`--tune`) | 0.812 | 0.605 | 0.812 | 30.7 s | 2.3 s |
| Same with `--C 0.5` (chosen on out-of-fold micro-F1) | **0.816** | **0.609** | 0.816 | 29.5 s | 2.4 s |
| Published SVM, per-category tuned thresholds (Lewis et al. 2004) | 0.816 | 0.607 | | | |

What the rows show: siblings-trained nodes are fast and predict fewer labels than the flat
model; per-class thresholds lift macro-F1 by eight points; inclusive training adds two points
of micro-F1 on top because a node can reject documents its parent mis-routed; routed
thresholds edge out local SCut. The whole hierarchy trains in under a minute on one core.

## GermEval 2019 Task 1

German book blurbs from Random House (Remus, Aly and Biemann 2019): 343 genres in a 4-level
tree with 8 root genres, 14,548 / 2,079 / 4,157 training / development / test blurbs. Subtask
B scores the full label set of each blurb; the winning system (TwistBytes, micro-F1 0.6767)
used this library with three TF-IDF views fitted per node and a negative decision threshold.

```bash
uv run python benchmarks/germeval2019_benchmark.py
```

The script builds TF-IDF views fitted once on the training records and shared by every node.
The `text` feature set has word 1-2 grams and character 2-3 grams of the title + blurb;
`text+metadata` adds three views of the fields every participant had at test time: the title
on its own, the author names as tokens, and ISBN publisher prefixes, which identify the imprint
(the task report lists several teams using such metadata; the publisher URL was withheld and is
not used). Local classifiers are inclusive-trained `LinearSVC`s with C=1.5. The feature set,
decision threshold and root fallback are selected on the development split; the model is then
refitted on train + dev and the test set scored once per configuration.

| Configuration | subtask B micro-F1 | subtask A micro-F1 |
|---|---:|---:|
| `text`, threshold 0, no root fallback | 0.590 | 0.819 |
| `text`, dev-selected threshold -0.40, root fallback, siblings-trained nodes | 0.634 | |
| `text`, dev-selected threshold -0.40, root fallback | 0.651 | 0.807 |
| `text+metadata`, dev-selected threshold -0.35, root fallback | **0.725** | **0.872** |
| Published TwistBytes system (per-node vocabularies, three views) | 0.677 | 0.863 (flat model) |

Authors and imprints are the lever: on the development split the author view alone adds five
points and the ISBN view three, while the title view, the publication year, wider character
n-grams and the winning system's word 1-7 gram views each move the score by less than half a
point. Per-class thresholds are not used here: most of the 343 labels have too few development
positives, and in 2-fold cross-tuning within the development split every per-class scheme loses
to one scalar threshold.

## What did not help

Measured and rejected, so that nobody has to repeat it:

- **Choosing the base classifier per node** (by held-out average precision over a ladder of
  linear models) is neutral on RCV1 and slightly negative on GermEval; node "hardness" does
  not predict which classifier wins. The per-node `base_estimator` dict remains available, but
  the library has no selector.
- **Global SCut** (one population for every class) is worse than no tuning at all on RCV1:
  sibling-trained local classifiers give meaningless scores to out-of-subtree samples.
- **Scalar thresholds chosen out-of-fold** on RCV1 do not transfer to the test period, while
  per-class thresholds do.
