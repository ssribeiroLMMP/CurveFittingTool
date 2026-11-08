"""Unit tests for DataSeries construction and numerical structure."""

import numpy as np
import pytest

from curvefit.domain.data_series import DataSeries
from curvefit.domain.variable import Variable


class TestConstruction:
    def test_accepts_single_variables_without_explicit_tuple(self) -> None:
        x = Variable(symbol="x")
        y = Variable(symbol="y")

        series = DataSeries(
            independent_variables=x,
            dependent_variables=y,
            independent_data=np.array([[1.0], [2.0]]),
            dependent_data=np.array([[3.0], [4.0]]),
        )

        assert series.independent_variables == (x,)
        assert series.dependent_variables == (y,)

    def test_normalizes_integer_matrices_to_float64(self) -> None:
        series = DataSeries(
            independent_variables=Variable(symbol="x"),
            dependent_variables=Variable(symbol="y"),
            independent_data=np.array([[1], [2]], dtype=np.int32),
            dependent_data=np.array([[3], [4]], dtype=np.int64),
        )

        assert series.independent_data.dtype == np.float64
        assert series.dependent_data.dtype == np.float64

    def test_copies_input_matrices(self) -> None:
        independent = np.array([[1.0], [2.0]])
        dependent = np.array([[3.0], [4.0]])

        series = DataSeries(
            independent_variables=Variable(symbol="x"),
            dependent_variables=Variable(symbol="y"),
            independent_data=independent,
            dependent_data=dependent,
        )

        independent[0, 0] = 99.0
        dependent[0, 0] = 88.0

        assert series.independent_data[0, 0] == 1.0
        assert series.dependent_data[0, 0] == 3.0


class TestDataSeriesInvariants:
    def test_rejects_same_symbol_for_independent_and_dependent(self) -> None:
        with pytest.raises(
            ValueError,
            match="must use distinct symbols",
        ):
            DataSeries(
                independent_variables=Variable(symbol="x"),
                dependent_variables=Variable(symbol="x"),
                independent_data=np.array([[1.0], [2.0]]),
                dependent_data=np.array([[3.0], [4.0]]),
            )

    def test_rejects_different_sample_counts(self) -> None:
        with pytest.raises(
            ValueError,
            match="same number of observations",
        ):
            DataSeries(
                independent_variables=Variable(symbol="x"),
                dependent_variables=Variable(symbol="y"),
                independent_data=np.array([[1.0], [2.0]]),
                dependent_data=np.array([[3.0]]),
            )

    def test_rejects_column_count_different_from_variable_count(self) -> None:
        with pytest.raises(
            ValueError,
            match="one column per variable",
        ):
            DataSeries(
                independent_variables=Variable(symbol="x"),
                dependent_variables=Variable(symbol="y"),
                independent_data=np.array(
                    [[1.0, 10.0], [2.0, 20.0]],
                ),
                dependent_data=np.array([[3.0], [4.0]]),
            )
