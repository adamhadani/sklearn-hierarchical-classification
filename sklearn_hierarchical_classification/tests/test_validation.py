"""Test validation logic."""

from hamcrest import assert_that, calling, raises

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
            "algorithm": "lcn",
            "training_strategy": None,
        },
        {
            "algorithm": "lcn",
            "training_strategy": "some_invalid_training_strategy",
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
