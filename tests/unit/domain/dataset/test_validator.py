"""
Check whether a DataSeries and its variable counts are accepted.

These tests call DatasetValidator.data_series directly, allowing each test to
choose which counts to require. The Dataset constructor requires one input
variable and one output variable.

NumPy creates example arrays. pytest.mark.parametrize runs a separate case for
each listed value or combination. pytest.raises requires the specified error;
match checks part of its message. assert checks the expected result.
"""

import numpy as np
import pytest

from curvefit.domain.data_series import DataSeries
from curvefit.domain.dataset import DatasetValidator
from curvefit.domain.variable import Variable


def make_series(inputs: int = 1, outputs: int = 1) -> DataSeries:
    """
    Create two observations with the requested numbers of inputs and outputs.
    Give each variable a different symbol and fill the arrays with 1, since
    these tests concern column counts.
    """
    return DataSeries(
        independent_variables=[Variable(f"x_{i}") for i in range(inputs)],
        dependent_variables=[Variable(f"y_{i}") for i in range(outputs)],
        independent_data=np.ones((2, inputs)),
        dependent_data=np.ones((2, outputs)),
    )


class TestDataSeries:
    """Check the supplied series type and the requested variable counts."""
    def test_returns_valid_series_without_copying(self) -> None:
        """
        Pass a valid series to the validator. Check that it returns the same
        object rather than creating a copy.
        """
        series = make_series()
        assert DatasetValidator.data_series(series) is series

    @pytest.mark.parametrize("value", [None, "series", 1, [], {}, np.ones((2, 2))])
    def test_rejects_non_data_series(self, value: object) -> None:
        """
        Pass each listed invalid input. Expect ValueError because the supplied
        value must be a DataSeries.
        """
        with pytest.raises(ValueError, match="data_series must be a DataSeries"):
            DatasetValidator.data_series(value)

    @pytest.mark.parametrize("inputs, outputs", [(1, 1), (2, 1), (1, 2), (2, 3)])
    def test_accepts_dimensions_when_counts_are_unspecified(self, inputs: int, outputs: int) -> None:
        """
        Prepare series with different input and output counts. Check that all
        are accepted when no counts are required.
        """
        series = make_series(inputs, outputs)
        assert DatasetValidator.data_series(series) is series

    @pytest.mark.parametrize("inputs, outputs", [(1, 1), (2, 3)])
    def test_accepts_matching_counts(self, inputs: int, outputs: int) -> None:
        """
        Require exactly the counts present in the series. Check that both
        single-variable and multiple-variable series are accepted.
        """
        series = make_series(inputs, outputs)
        assert DatasetValidator.data_series(
            series, expected_independent_count=inputs, expected_dependent_count=outputs,
        ) is series

    @pytest.mark.parametrize("actual, expected", [(1, 2), (2, 1)])
    def test_rejects_independent_count_mismatch(self, actual: int, expected: int) -> None:
        """
        Prepare one input count and require a different count. Check rejection
        when there are either too few or too many input variables.
        """
        series = make_series(inputs=actual)
        with pytest.raises(ValueError, match="unsupported number of independent variables"):
            DatasetValidator.data_series(series, expected_independent_count=expected)

    @pytest.mark.parametrize("actual, expected", [(1, 2), (2, 1)])
    def test_rejects_dependent_count_mismatch(self, actual: int, expected: int) -> None:
        """
        Prepare one output count and require a different count. Check rejection
        when there are either too few or too many output variables.
        """
        series = make_series(outputs=actual)
        with pytest.raises(ValueError, match="unsupported number of dependent variables"):
            DatasetValidator.data_series(series, expected_dependent_count=expected)

    def test_independent_constraint_leaves_dependent_count_unrestricted(self) -> None:
        """
        Require one input in a series with one input and three outputs. Check
        that an input requirement does not restrict outputs.
        """
        series = make_series(1, 3)
        assert DatasetValidator.data_series(series, expected_independent_count=1) is series

    def test_dependent_constraint_leaves_independent_count_unrestricted(self) -> None:
        """
        Require one output in a series with three inputs and one output. Check
        that an output requirement does not restrict inputs.
        """
        series = make_series(3, 1)
        assert DatasetValidator.data_series(series, expected_dependent_count=1) is series
