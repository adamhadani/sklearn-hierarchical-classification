"""Validation helpers."""

import numpy as np

from sklearn_hierarchical_classification.constants import (
    VALID_ALGORITHM,
    VALID_FEATURE_EXTRACTION,
    VALID_PREDICTION_DEPTH,
    VALID_TRAINING_STRATEGY,
)


class ParameterValidator:
    """Parameter validation logic for the HierarchicalClassifier class."""

    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __call__(self):
        return self._validate()

    def _validate(self):
        if self.algorithm not in VALID_ALGORITHM:
            raise TypeError(
                "'algorithm' must be set to one of: {}.".format(
                    ", ".join(VALID_ALGORITHM),
                )
            )

        if self.algorithm == "lcn" and not self.training_strategy:
            raise TypeError("""When 'algorithm' is set to "lcn", 'training_strategy' must be set.""")

        if self.algorithm == "lcpn" and self.training_strategy not in (None, "siblings", "inclusive"):
            raise TypeError(
                """When 'algorithm' is set to "lcpn", 'training_strategy' must be None, "siblings" (the default:
                a node's classifier is trained on the documents of its subtree) or "inclusive" (documents from
                outside the subtree are added as negatives)."""
            )

        if self.algorithm == "lcpn" and self.training_strategy == "inclusive" and self.mlb is None:
            raise TypeError(
                """'training_strategy' "inclusive" requires 'mlb': out-of-subtree documents are negatives for
                every child, which only a multi-label (indicator) local classifier can express."""
            )

        if self.training_strategy and self.training_strategy not in VALID_TRAINING_STRATEGY:
            raise TypeError(
                "'training_strategy' must be set to one of: {}.".format(
                    ", ".join(VALID_TRAINING_STRATEGY),
                )
            )

        if self.prediction_depth not in VALID_PREDICTION_DEPTH:
            raise TypeError(
                "'prediction_depth' must be set to one of: {}.".format(
                    ", ".join(VALID_PREDICTION_DEPTH),
                )
            )

        if (self.prediction_depth == "nmlnp") ^ (self.stopping_criteria is not None):
            raise TypeError(
                """When 'prediction_depth' is set to "nmlnp", 'stopping_criteria' must be set
                to a float or callable. Conversely, stopping_criteria should not be specified
                when prediction_depth is not set to "nmlnp"."""
            )

        if self.stopping_criteria is not None and not any(
            (
                isinstance(self.stopping_criteria, float),
                callable(self.stopping_criteria),
            )
        ):
            raise TypeError("""'stopping_criteria' must be set to a float or a callable.""")

        if self.mlb is not None and (self.prediction_depth == "nmlnp" or self.stopping_criteria is not None):
            raise TypeError(
                """Early stopping ('prediction_depth' set to "nmlnp" or a 'stopping_criteria') is only defined
                for single-label prediction and cannot be combined with 'mlb'."""
            )

        if not isinstance(self.mlb_min_root_predictions, int | np.integer) or self.mlb_min_root_predictions < 0:
            raise TypeError("'mlb_min_root_predictions' must be a non-negative integer.")

        if self.feature_extraction not in VALID_FEATURE_EXTRACTION:
            raise TypeError(
                "'feature_extraction' must be set to one of: {}.".format(
                    ", ".join(VALID_FEATURE_EXTRACTION),
                )
            )


def validate_parameters(instance):
    return ParameterValidator(instance)()


def is_estimator(obj):
    if hasattr(obj.__class__, "fit"):
        return True
