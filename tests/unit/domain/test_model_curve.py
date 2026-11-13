"""Tests for ModelCurve."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from curvefit.domain.data_series import DataSeries
from curvefit.domain.model_curve import ModelCurve
from curvefit.domain.variable import Variable


def _make_series() -> DataSeries:
    return DataSeries(
        independent_variables=Variable("x", name="Distance", unit="m"),
        dependent_variables=Variable("y", name="Response", unit="Pa"),
        independent_data=np.array([[0.0], [0.5], [1.0]]),
        dependent_data=np.array([[1.0], [2.0], [3.0]]),
    )


def test_composes_a_copied_data_series_for_presentation() -> None:
    series = _make_series()

    curve = ModelCurve(series)

    assert curve.sample_count == 3
    np.testing.assert_array_equal(curve.independent_data, series.independent_data)
    np.testing.assert_array_equal(curve.dependent_data, series.dependent_data)
    assert curve.data_series is not series


def test_does_not_follow_source_variable_changes() -> None:
    series = _make_series()
    curve = ModelCurve(series)

    series.independent_variables[0].configure(name="Edited", unit="cm")

    assert curve.independent_variables[0].name == "Distance"
    assert curve.independent_variables[0].unit == "m"


def test_returned_data_series_cannot_mutate_curve_variables() -> None:
    curve = ModelCurve(_make_series())
    exported = curve.data_series

    exported.independent_variables[0].configure(name="Edited")

    assert curve.independent_variables[0].name == "Distance"


def test_numerical_matrices_are_read_only() -> None:
    curve = ModelCurve(_make_series())

    with pytest.raises(ValueError):
        curve.dependent_data[0, 0] = 10.0


def test_wrapper_is_immutable() -> None:
    curve = ModelCurve(_make_series())

    with pytest.raises(FrozenInstanceError):
        curve._data_series = _make_series()


def test_rejects_non_data_series_input() -> None:
    with pytest.raises(ValueError, match="must be a DataSeries"):
        ModelCurve(object())
