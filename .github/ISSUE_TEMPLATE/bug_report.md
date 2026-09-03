---
name: Bug report
about: Something behaves differently from what the documentation says
title: ''
labels: bug
assignees: ''

---

**What happened**
A clear description of the problem, with the full traceback if there is one.

**Minimal example**
The smallest hierarchy, data and classifier configuration that reproduces it:

```python
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT

class_hierarchy = {ROOT: ["A", "B"], "A": ["1", "2"], "B": ["3"]}
clf = HierarchicalClassifier(class_hierarchy=class_hierarchy)
...
```

**Expected behaviour**
What you expected instead.

**Environment**
- sklearn-hierarchical-classification version:
- scikit-learn / numpy / networkx versions:
- Python version and OS:
