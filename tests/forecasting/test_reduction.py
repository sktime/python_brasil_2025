"""Test the sktime contract for Prophet and HierarchicalProphet."""

import pytest  # noqa: F401
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks

from tsbook.forecasting.reduction import ReductionForecaster


@parametrize_with_checks([ReductionForecaster])
def test_sktime_api_compliance(obj, test_name):
    """Test the sktime contract for ReductionForecaster."""
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)
