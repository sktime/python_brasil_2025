"""Test suite for the ReductionForecaster implementation."""

import numpy as np
import pandas as pd
import pytest  # noqa: F401
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks

from tsbook.forecasting.reduction import ReductionForecaster


@parametrize_with_checks([ReductionForecaster])
def test_sktime_api_compliance(obj, test_name):
    """Test the sktime contract for ReductionForecaster."""
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)


def test_reduction_insample_predictions_cover_training():
    """Ensure in-sample predictions return finite values for recent history."""

    n = 30
    idx = pd.RangeIndex(n, name="time")
    y = pd.Series(np.arange(n, dtype=float), index=idx, name="y")

    forecaster = ReductionForecaster(
        estimator=LinearRegression(),
        window_length=5,
        steps_ahead=3,
    )
    forecaster.fit(y)

    fh = ForecastingHorizon([-5, -1, 0, 1, 2], is_relative=True)
    y_pred = forecaster.predict(fh)

    insample_steps = [-5, -1, 0]
    insample_times = [len(idx) - 1 + step for step in insample_steps]

    insample_forecasts = y_pred.loc[insample_times]
    assert not insample_forecasts.isna().any()

    observed_values = y.loc[insample_times]
    assert np.allclose(insample_forecasts.values, observed_values.values, atol=1e-6)

    assert len(y_pred) == len(fh)
