"""Unit tests for DatasetValidator."""

from __future__ import annotations

import numpy as np
import pytest

from curvefit.domain.dataset import DatasetValidator
from curvefit.domain.variable import Variable


class TestVariables:
    """Tests for variable collection validation and normalization."""

    def test_normalizes_single_variable_to_tuple(self) -> None:
        variable = Variable(symbol="x")

        result = DatasetValidator.variables(variable)

        assert result == (variable,)

    def test_preserves_sequence_order_as_tuple(self) -> None:
        x1 = Variable(symbol="x_1")
        x2 = Variable(symbol="x_2")

        result = DatasetValidator.variables([x1, x2])

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
        with pytest.raises(
            ValueError,
            match="variable collection cannot be empty",
        ):
            DatasetValidator.variables(values)

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
        with pytest.raises(
            ValueError,
            match="variables must be a Variable or a sequence",
        ):
            DatasetValidator.variables(values)

    def test_rejects_non_variable_member(self) -> None:
        variable = Variable(symbol="x")

        with pytest.raises(
            ValueError,
            match="must contain only Variable objects",
        ):
            DatasetValidator.variables([variable, "y"])

    def test_rejects_duplicate_symbols(self) -> None:
        first = Variable(symbol="x", name="First")
        second = Variable(symbol="x", name="Second")

        with pytest.raises(
            ValueError,
            match="cannot contain duplicate symbols",
        ):
            DatasetValidator.variables([first, second])

    def test_accepts_expected_count(self) -> None:
        variables = [
            Variable(symbol="x_1"),
            Variable(symbol="x_2"),
        ]

        result = DatasetValidator.variables(
            variables,
            expected_count=2,
        )

        assert len(result) == 2

    def test_rejects_unexpected_count(self) -> None:
        variables = [
            Variable(symbol="x_1"),
            Variable(symbol="x_2"),
        ]

        with pytest.raises(
            ValueError,
            match=r"must contain exactly 1 variable\(s\)",
        ):
            DatasetValidator.variables(
                variables,
                expected_count=1,
            )


class TestDataMatrix:
    """Tests for canonical numerical matrix validation."""

    def test_accepts_numeric_two_dimensional_matrix(self) -> None:
        values = np.array(
            [[1, 2], [3, 4]],
            dtype=np.int64,
        )

        result = DatasetValidator.data_matrix(values)

        np.testing.assert_array_equal(
            result,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        assert result.dtype == np.float64

    def test_returns_independent_copy(self) -> None:
        values = np.array([[1.0], [2.0]])

        result = DatasetValidator.data_matrix(values)
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
        with pytest.raises(
            ValueError,
            match="data matrix must be a NumPy array",
        ):
            DatasetValidator.data_matrix(values)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1.0, 2.0]),
            np.array([[[1.0]]]),
        ],
    )
    def test_rejects_non_two_dimensional_array(self, values) -> None:
        with pytest.raises(
            ValueError,
            match="data matrix must be two-dimensional",
        ):
            DatasetValidator.data_matrix(values)

    def test_rejects_zero_observations(self) -> None:
        values = np.empty((0, 1))

        with pytest.raises(
            ValueError,
            match="at least one observation",
        ):
            DatasetValidator.data_matrix(values)

    def test_rejects_zero_variables(self) -> None:
        values = np.empty((3, 0))

        with pytest.raises(
            ValueError,
            match="at least one variable",
        ):
            DatasetValidator.data_matrix(values)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([["1"], ["2"]]),
            np.array([[True], [False]]),
            np.array([[object()], [object()]], dtype=object),
        ],
    )
    def test_rejects_non_numeric_values(self, values) -> None:
        with pytest.raises(
            ValueError,
            match="must contain numeric values",
        ):
            DatasetValidator.data_matrix(values)

    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    def test_rejects_non_finite_values(self, invalid_value) -> None:
        values = np.array([[1.0], [invalid_value]])

        with pytest.raises(
            ValueError,
            match="must contain only finite values",
        ):
            DatasetValidator.data_matrix(values)


class TestRelationships:
    """Tests for relationships among variables and numerical matrices."""

    def test_matching_variable_count_accepts_correspondence(self) -> None:
        variables = (
            Variable(symbol="x_1"),
            Variable(symbol="x_2"),
        )
        data = np.ones((4, 2), dtype=np.float64)

        assert DatasetValidator.matching_variable_count(
            variables,
            data,
        ) is None

    def test_matching_variable_count_rejects_mismatch(self) -> None:
        variables = (Variable(symbol="x"),)
        data = np.ones((4, 2), dtype=np.float64)

        with pytest.raises(
            ValueError,
            match="one column per variable",
        ):
            DatasetValidator.matching_variable_count(
                variables,
                data,
            )

    def test_matching_sample_count_accepts_equal_rows(self) -> None:
        independent = np.ones((4, 2), dtype=np.float64)
        dependent = np.ones((4, 1), dtype=np.float64)

        assert DatasetValidator.matching_sample_count(
            independent,
            dependent,
        ) is None

    def test_matching_sample_count_rejects_different_rows(self) -> None:
        independent = np.ones((4, 1), dtype=np.float64)
        dependent = np.ones((3, 1), dtype=np.float64)

        with pytest.raises(
            ValueError,
            match="same number of observations",
        ):
            DatasetValidator.matching_sample_count(
                independent,
                dependent,
            )

    def test_distinct_symbols_accepts_disjoint_collections(self) -> None:
        independent = (
            Variable(symbol="x_1"),
            Variable(symbol="x_2"),
        )
        dependent = (Variable(symbol="y"),)

        assert DatasetValidator.distinct_symbols(
            independent,
            dependent,
        ) is None

    def test_distinct_symbols_rejects_overlap(self) -> None:
        independent = (Variable(symbol="x"),)
        dependent = (Variable(symbol="x"),)

        with pytest.raises(
            ValueError,
            match="must use distinct symbols",
        ):
            DatasetValidator.distinct_symbols(
                independent,
                dependent,
            )
