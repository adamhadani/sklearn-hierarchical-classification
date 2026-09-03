"""
Constants.

"""

# Special id reserved for an artificial 'root node' that may be added to class hierarchy
# when using a 'one classifier per parent node' strategy.
ROOT = "<ROOT>"

# Dictionary keys used in various places by classifier
CLASSIFIER = "classifier"
DEFAULT = "default"
METAFEATURES = "metafeatures"
TRAINED_CLASSES = "trained_classes"

# Enumeration of valid configuration types
VALID_ALGORITHM = ("lcn", "lcpn")  # "lcn" is deprecated (never implemented); remove in the next major release
VALID_FEATURE_EXTRACTION = ("preprocessed", "raw")
VALID_PREDICTION_DEPTH = ("mlnp", "nmlnp")
VALID_TRAINING_STRATEGY = (
    "siblings",
    "inclusive",
    # The rest are only accepted with the deprecated "lcn" and go with it
    "exclusive",
    "less_exclusive",
    "less_inclusive",
    "exclusive_siblings",
)
