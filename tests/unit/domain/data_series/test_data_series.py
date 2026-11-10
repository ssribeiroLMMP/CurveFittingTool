"""
Check DataSeries creation from variables and numerical arrays.

Each array row represents an observation; each column corresponds to a
variable. Tests prepare examples and check the resulting series or the expected
error when the supplied data do not match.

NumPy creates the arrays. pytest.raises requires construction to raise the
specified error; match checks part of its message. assert checks the result.
"""

import numpy as np
import pytest

from curvefit.domain.data_series import DataSeries
from curvefit.domain.variable import Variable


class TestConstruction:
    """Check how supplied variables and arrays are stored in the series."""
    def test_accepts_single_variables_without_explicit_tuple(self) -> None:
        """
        Supply one Variable object for each side. Check that the series stores
        each one in a tuple containing one item.
        """
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
        """
        Create input and output arrays with different integer types. Check that
        both are stored as float64 numbers.
        """
        series = DataSeries(
            independent_variables=Variable(symbol="x"),
            dependent_variables=Variable(symbol="y"),
            independent_data=np.array([[1], [2]], dtype=np.int32),
            dependent_data=np.array([[3], [4]], dtype=np.int64),
        )

        assert series.independent_data.dtype == np.float64
        assert series.dependent_data.dtype == np.float64

    def test_copies_input_matrices(self) -> None:
        """
        Change the original arrays after creating the series. Check that values
        stored in the series still match the originally supplied values.
        """
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
    """Check that symbols, rows, and columns match before creating a series."""
    def test_rejects_same_symbol_for_independent_and_dependent(self) -> None:
        """
        Use x as both the input and output symbol. Expect an error because the
        quantities on each side need distinct symbols.
        """
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
        """
        Supply two input observations but only one output observation. Expect
        an error because each input row needs a corresponding output row.
        """
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
        """
        Supply two input columns but describe only one variable. Expect an
        error because every column needs a corresponding variable.
        """
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
