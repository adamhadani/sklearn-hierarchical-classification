#!/usr/bin/env python
"""
GermEval 2019 Task 1 benchmark: hierarchical classification of German book blurbs.

Remus, Aly and Biemann (2019). 343 genre labels in a 4-level tree with 8 root genres; 14,548
training, 2,079 development and 4,157 test blurbs. Subtask A scores the root genres, subtask B the
full label set of each blurb (micro-F1 in both). The winning subtask-B system (TwistBytes, Benites
2019, micro-F1 0.6767) used this library with TF-IDF + LinearSVC local classifiers and a negative
decision threshold to trade precision for recall.

Features, classifier and protocol are shared with the Blurb Genre Collection benchmark: see
`blurbs.py`. The winning system fitted its TF-IDF vocabularies per node, on the node's own
subtree, and its word 1-7-gram views were measured here and do not beat the word 1-2 + character
2-3 grams used. The metadata views use fields every participant had at test time (the publisher
URL was withheld and is not used); the task report notes several teams used such metadata, and
authors and imprints are strong genre predictors.

The official data package (CC BY-NC 4.0, University of Hamburg Language Technology group) is
downloaded on first use into ~/scikit_learn_data/germeval2019.

Example:

    uv run python benchmarks/germeval2019_benchmark.py

"""

import re
from pathlib import Path

from blurbs import Blurbs, parse_args, run


GERMEVAL_2019 = Blurbs(
    name="GermEval 2019 Task 1",
    package_url=(
        "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc/"
        "germeval2019t1-public-data-final.zip"
    ),
    cache_dir=Path.home() / "scikit_learn_data" / "germeval2019",
    files={
        "train": "blurbs_train.txt",
        "dev": "blurbs_dev.txt",
        "test": "blurbs_test.txt",
        "hierarchy": "hierarchy.txt",
    },
    topic=re.compile(r"<topic d=\"\d\"[^>]*>(.*?)</topic>"),
    tags={"title": "title", "body": "body", "authors": "authors", "isbn": "isbn"},
    C=1.5,  # the winning system's value
    published=[
        "published test scores: TwistBytes (this library, t=-0.25) subtask B (all labels) micro-F1 0.6767 (1st of 10);",
        "                       TwistBytes flat model subtask A (root genres) micro-F1 0.8634 (2nd)",
    ],
)


if __name__ == "__main__":
    run(GERMEVAL_2019, parse_args(GERMEVAL_2019, __doc__))
