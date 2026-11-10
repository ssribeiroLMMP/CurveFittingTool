"""
Check the rules used to validate a DataSeries one at a time.

The groups below cover variables, arrays, and their correspondence. Each test
calls the method responsible for the rule being checked.

NumPy prepares arrays and compares their values. pytest.mark.parametrize
repeats a test for each listed input. pytest.raises requires an error; match
checks part of its message. assert checks the expected condition. For methods
that only check a relationship, returning None without raising an error means
the relationship was accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from curvefit.domain.data_series import DataSeriesValidator
from curvefit.domain.variable import Variable


class TestVariables:
    """Check which variables are accepted and how their order is preserved."""

    def test_normalizes_single_variable_to_tuple(self) -> None:
        """Supply one Variable and check that it is returned inside a tuple."""
        variable = Variable(symbol="x")

        result = DataSeriesValidator.variables(variable)

        assert result == (variable,)

    def test_preserves_sequence_order_as_tuple(self) -> None:
        """
        Supply two variables in a list. Check that the result is a tuple with
        the same variables in the same order.
        """
        x1 = Variable(symbol="x_1")
        x2 = Variable(symbol="x_2")

        result = DataSeriesValidator.variables([x1, x2])

        assert result == (x1, x2)
        assert isinstance(result, tuple)

    @pytest.mark.parametrize(
        "values",
        [
            [],
            (),
        ],
    )
    def test_rejects_empty_collection(self, values) -> None:
        """
        Try an empty list and an empty tuple. Both must be rejected because at
        least one variable is required.
        """
        with pytest.raises(
            ValueError,
            match="variable collection cannot be empty",
        ):
            DataSeriesValidator.variables(values)

    @pytest.mark.parametrize(
        "values",
        [
            "x",
            b"x",
            1,
            None,
        ],
    )
    def test_rejects_invalid_collection_type(self, values) -> None:
        """
        Try text, bytes, a number, and None. None of these describes a Variable
        or a sequence of Variable objects.
        """
        with pytest.raises(
            ValueError,
            match="variables must be a Variable or a sequence",
        ):
            DataSeriesValidator.variables(values)

    def test_rejects_non_variable_member(self) -> None:
        """
        Mix a Variable and a string in the same list. Check that every item
        must be a Variable object.
        """
        variable = Variable(symbol="x")

        with pytest.raises(
            ValueError,
            match="must contain only Variable objects",
        ):
            DataSeriesValidator.variables([variable, "y"])

    def test_rejects_duplicate_symbols(self) -> None:
        """
        Create two variables with different names but the same symbol x. Check
        that different names do not make repeated symbols acceptable.
        """
        first = Variable(symbol="x", name="First")
        second = Variable(symbol="x", name="Second")

        with pytest.raises(
            ValueError,
            match="cannot contain duplicate symbols",
        ):
            DataSeriesValidator.variables([first, second])

    def test_accepts_expected_count(self) -> None:
        """
        Supply two variables and require two. Check that the matching count is
        accepted.
        """
        variables = [
            Variable(symbol="x_1"),
            Variable(symbol="x_2"),
        ]

        result = DataSeriesValidator.variables(
            variables,
            expected_count=2,
        )

        assert len(result) == 2

    def test_rejects_unexpected_count(self) -> None:
        """
        Supply two variables when only one is required. Check that the
        mismatched count raises an error.
        """
        variables = [
            Variable(symbol="x_1"),
            Variable(symbol="x_2"),
        ]

        with pytest.raises(
            ValueError,
            match=r"must contain exactly 1 variable\(s\)",
        ):
            DataSeriesValidator.variables(
                variables,
                expected_count=1,
            )


class TestDataMatrix:
    """Check the shape, contents, and copying of supplied arrays."""

    def test_accepts_numeric_two_dimensional_matrix(self) -> None:
        """
        Supply an integer array with two rows and two columns. Check that its
        values are preserved and converted to float64.
        """
        values = np.array(
            [[1, 2], [3, 4]],
            dtype=np.int64,
        )

        result = DataSeriesValidator.data_matrix(values)

        np.testing.assert_array_equal(
            result,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        assert result.dtype == np.float64

    def test_returns_independent_copy(self) -> None:
        """
        Change the array returned by the validator. Check that the original
        array is unchanged, showing that the data were copied.
        """
        values = np.array([[1.0], [2.0]])

        result = DataSeriesValidator.data_matrix(values)
        result[0, 0] = 99.0

        assert values[0, 0] == 1.0

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.0],
            ((1.0,), (2.0,)),
            None,
        ],
    )
    def test_rejects_non_numpy_input(self, values) -> None:
        """
        Supply a list, a tuple, and None. Check that the method requires a
        NumPy array even when a sequence contains numbers.
        """
        with pytest.raises(
            ValueError,
            match="data matrix must be a NumPy array",
        ):
            DataSeriesValidator.data_matrix(values)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1.0, 2.0]),
            np.array([[[1.0]]]),
        ],
    )
    def test_rejects_non_two_dimensional_array(self, values) -> None:
        """
        Supply a one-dimensional vector and a three-dimensional array. Both
        must be rejected because the data must have rows and columns.
        """
        with pytest.raises(
            ValueError,
            match="data matrix must be two-dimensional",
        ):
            DataSeriesValidator.data_matrix(values)

    def test_rejects_zero_observations(self) -> None:
        """
        Create an array with no rows. Check that at least one observation is
        required.
        """
        values = np.empty((0, 1))

        with pytest.raises(
            ValueError,
            match="at least one observation",
        ):
            DataSeriesValidator.data_matrix(values)

    def test_rejects_zero_variables(self) -> None:
        """
        Create an array with three rows and no columns. Check that at least one
        variable is required.
        """
        values = np.empty((3, 0))

        with pytest.raises(
            ValueError,
            match="at least one variable",
        ):
            DataSeriesValidator.data_matrix(values)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([["1"], ["2"]]),
            np.array([[True], [False]]),
            np.array([[object()], [object()]], dtype=object),
        ],
    )
    def test_rejects_non_numeric_values(self, values) -> None:
        """
        Try arrays containing text, True/False values, and generic objects.
        Check that these contents are rejected.
        """
        with pytest.raises(
            ValueError,
            match="must contain numeric values",
        ):
            DataSeriesValidator.data_matrix(values)

    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    def test_rejects_non_finite_values(self, invalid_value) -> None:
        """
        Include NaN, positive infinity, or negative infinity alongside a valid
        value. Each case must be rejected.
        """
        values = np.array([[1.0], [invalid_value]])

        with pytest.raises(
            ValueError,
            match="must contain only finite values",
        ):
            DataSeriesValidator.data_matrix(values)


class TestRelationships:
    """Check correspondence between variables, columns, and observations."""

    def test_matching_variable_count_accepts_correspondence(self) -> None:
        """
        Associate two variables with a two-column array. Check that the
        relationship is accepted without an error.
        """
        variables = (
            Variable(symbol="x_1"),
            Variable(symbol="x_2"),
        )
        data = np.ones((4, 2), dtype=np.float64)

        assert DataSeriesValidator.matching_variable_count(
            variables,
            data,
        ) is None

    def test_matching_variable_count_rejects_mismatch(self) -> None:
        """
        Associate one variable with a two-column array. Check that the missing
        variable for the second column causes an error.
        """
        variables = (Variable(symbol="x"),)
        data = np.ones((4, 2), dtype=np.float64)

        with pytest.raises(
            ValueError,
            match="one column per variable",
        ):
            DataSeriesValidator.matching_variable_count(
                variables,
                data,
            )

    def test_matching_sample_count_accepts_equal_rows(self) -> None:
        """
        Compare arrays with four rows each but different column counts. Check
        that this rule requires only matching observation counts.
        """
        independent = np.ones((4, 2), dtype=np.float64)
        dependent = np.ones((4, 1), dtype=np.float64)

        assert DataSeriesValidator.matching_sample_count(
            independent,
            dependent,
        ) is None

    def test_matching_sample_count_rejects_different_rows(self) -> None:
        """
        Compare four input rows with three output rows. Check that the missing
        corresponding observation causes an error.
        """
        independent = np.ones((4, 1), dtype=np.float64)
        dependent = np.ones((3, 1), dtype=np.float64)

        with pytest.raises(
            ValueError,
            match="same number of observations",
        ):
            DataSeriesValidator.matching_sample_count(
                independent,
                dependent,
            )

    def test_distinct_symbols_accepts_disjoint_collections(self) -> None:
        """
        Use x_1 and x_2 for inputs and y for the output. Check that distinct
        symbols across the two sides are accepted.
        """
        independent = (
            Variable(symbol="x_1"),
            Variable(symbol="x_2"),
        )
        dependent = (Variable(symbol="y"),)

        assert DataSeriesValidator.distinct_symbols(
            independent,
            dependent,
        ) is None

    def test_distinct_symbols_rejects_overlap(self) -> None:
        """
        Use x for both input and output. Check that repeating a symbol across
        the two sides causes an error.
        """
        independent = (Variable(symbol="x"),)
        dependent = (Variable(symbol="x"),)

        with pytest.raises(
            ValueError,
            match="must use distinct symbols",
        ):
            DataSeriesValidator.distinct_symbols(
                independent,
                dependent,
            )
