"""Test validation logic."""

import pytest
from hamcrest import assert_that, calling, raises
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn_hierarchical_classification.tests.fixtures import make_classifier_and_data


def test_parameter_validation():
    """Test parameter validation checks for consistent assignment."""
    test_cases = [
        {
            "prediction_depth": "nmlnp",
            "stopping_criteria": None,
        },
        {
            "prediction_depth": "nmlnp",
            "stopping_criteria": "not_a_float_or_a_callable",
        },
        {
            "prediction_depth": "mlnp",
            "stopping_criteria": 123.4,
        },
        {
            "prediction_depth": "some_invalid_prediction_depth_value",
        },
        {
            "algorithm": "lcpn",
            "training_strategy": "exclusive",
        },
        {
            "algorithm": "some_invalid_algorithm_value",
        },
    ]

    for classifier_kwargs in test_cases:
        clf, (X, y) = make_classifier_and_data(**classifier_kwargs)
        assert_that(calling(clf.fit).with_args(X=X, y=y), raises(TypeError))


@pytest.mark.parametrize("training_strategy", [None, "some_invalid_training_strategy"])
def test_lcn_parameters_are_validated_after_the_deprecation_warning(training_strategy):
    """The deprecated "lcn" algorithm warns first, then its parameters are checked as before."""
    clf, (X, y) = make_classifier_and_data(algorithm="lcn", training_strategy=training_strategy)

    with pytest.warns(FutureWarning, match="'lcn'"):
        assert_that(calling(clf.fit).with_args(X=X, y=y), raises(TypeError))


def test_nmlnp_is_rejected_with_multi_label_binarizer():
    """Early stopping is only defined for single-label prediction; with `mlb` it would be silently
    ignored, so fit must refuse the combination."""
    clf, (X, y) = make_classifier_and_data(prediction_depth="nmlnp", stopping_criteria=0.5, mlb=MultiLabelBinarizer())

    assert_that(calling(clf.fit).with_args(X=X, y=y), raises(TypeError, "mlb"))
