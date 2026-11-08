"""Tests for Dataset-specific DataSeries validation and dimension policy."""

import numpy as np
import pytest

from curvefit.domain.data_series import DataSeries
from curvefit.domain.dataset import DatasetValidator
from curvefit.domain.variable import Variable


def make_series(inputs: int = 1, outputs: int = 1) -> DataSeries:
    return DataSeries(
        independent_variables=[Variable(f"x_{i}") for i in range(inputs)],
        dependent_variables=[Variable(f"y_{i}") for i in range(outputs)],
        independent_data=np.ones((2, inputs)),
        dependent_data=np.ones((2, outputs)),
    )


class TestDataSeries:
    def test_returns_valid_series_without_copying(self) -> None:
        series = make_series()
        assert DatasetValidator.data_series(series) is series

    @pytest.mark.parametrize("value", [None, "series", 1, [], {}, np.ones((2, 2))])
    def test_rejects_non_data_series(self, value: object) -> None:
        with pytest.raises(ValueError, match="data_series must be a DataSeries"):
            DatasetValidator.data_series(value)

    @pytest.mark.parametrize("inputs, outputs", [(1, 1), (2, 1), (1, 2), (2, 3)])
    def test_accepts_dimensions_when_counts_are_unspecified(self, inputs: int, outputs: int) -> None:
        series = make_series(inputs, outputs)
        assert DatasetValidator.data_series(series) is series

    @pytest.mark.parametrize("inputs, outputs", [(1, 1), (2, 3)])
    def test_accepts_matching_counts(self, inputs: int, outputs: int) -> None:
        series = make_series(inputs, outputs)
        assert DatasetValidator.data_series(
            series, expected_independent_count=inputs, expected_dependent_count=outputs,
        ) is series

    @pytest.mark.parametrize("actual, expected", [(1, 2), (2, 1)])
    def test_rejects_independent_count_mismatch(self, actual: int, expected: int) -> None:
        series = make_series(inputs=actual)
        with pytest.raises(ValueError, match="unsupported number of independent variables"):
            DatasetValidator.data_series(series, expected_independent_count=expected)

    @pytest.mark.parametrize("actual, expected", [(1, 2), (2, 1)])
    def test_rejects_dependent_count_mismatch(self, actual: int, expected: int) -> None:
        series = make_series(outputs=actual)
        with pytest.raises(ValueError, match="unsupported number of dependent variables"):
            DatasetValidator.data_series(series, expected_dependent_count=expected)

    def test_independent_constraint_leaves_dependent_count_unrestricted(self) -> None:
        series = make_series(1, 3)
        assert DatasetValidator.data_series(series, expected_independent_count=1) is series

    def test_dependent_constraint_leaves_independent_count_unrestricted(self) -> None:
        series = make_series(3, 1)
        assert DatasetValidator.data_series(series, expected_dependent_count=1) is series
