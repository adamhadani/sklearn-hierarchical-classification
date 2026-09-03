#!/usr/bin/env python
"""
Blurb Genre Collection benchmark: hierarchical classification of English book blurbs.

Aly, Remus and Biemann (2019): 91,892 Penguin Random House blurbs with 146 genre labels in a
4-level hierarchy (7 root genres), split 58,715 / 14,785 / 18,394 into train / dev / test, about
3 labels per blurb. It is one of the four standard datasets of the hierarchical text
classification literature, so current neural results are directly comparable: a flat fine-tuned
BERT-base reaches micro-F1 81.4 / macro-F1 64.6 and the best hierarchy-aware encoders about
82.2 / 66.2 (HYDRA, EMNLP 2025, RoBERTa-base; DepthMatch 80.5 / 66.6), against 71.2 F1 for the
SVM baseline of the dataset paper.

Features, classifier and protocol are shared with the GermEval 2019 benchmark: see `blurbs.py`.
The dataset ships with labels for all three splits, and its records carry the same metadata
fields (author, ISBN) as the GermEval data.

The dataset (CC BY-NC, University of Hamburg Language Technology group) is downloaded on first
use into ~/scikit_learn_data/bgc.

Example:

    uv run python benchmarks/bgc_benchmark.py

"""

import re
from pathlib import Path

from blurbs import Blurbs, parse_args, run


BGC = Blurbs(
    name="Blurb Genre Collection (EN)",
    package_url="https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip",
    cache_dir=Path.home() / "scikit_learn_data" / "bgc",
    files={
        "train": "BlurbGenreCollection_EN_train.txt",
        "dev": "BlurbGenreCollection_EN_dev.txt",
        "test": "BlurbGenreCollection_EN_test.txt",
        "hierarchy": "hierarchy.txt",
    },
    topic=re.compile(r"<d\d>(.*?)</d\d>"),
    tags={"title": "title", "body": "body", "authors": "author", "isbn": "isbn"},
    C=1.0,
    published=[
        "published test scores (all labels micro-F1 / macro-F1): SVM baseline of the dataset paper 71.2 F1;",
        "    flat BERT-base 81.4 / 64.6, RoBERTa-base 81.5 / 64.1, HYDRA 82.2 / 66.2 (Karl and Scherp, EMNLP 2025)",
    ],
)


if __name__ == "__main__":
    run(BGC, parse_args(BGC, __doc__))
