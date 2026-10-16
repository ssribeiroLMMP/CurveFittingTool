"""Unit tests for the Dataset domain entity."""

from __future__ import annotations

import numpy as np
import pytest

from curvefit.domain.dataset import Dataset
from curvefit.domain.variable import Variable


def make_dataset(
    *,
    metadata=None,
) -> Dataset:
    """Create a valid single-input single-output Dataset for tests."""
    return Dataset(
        independent_variables=Variable(
            symbol="x",
            name="Time",
            unit="s",
        ),
        dependent_variables=Variable(
            symbol="y",
            name="Response",
            unit="Pa",
        ),
        independent_data=np.array([[1.0], [2.0], [3.0]]),
        dependent_data=np.array([[10.0], [20.0], [30.0]]),
        metadata=metadata,
    )


class TestConstruction:
    """Tests for Dataset construction and canonical state."""

    def test_accepts_single_variables_without_explicit_tuple(self) -> None:
        x = Variable(symbol="x")
        y = Variable(symbol="y")

        dataset = Dataset(
            independent_variables=x,
            dependent_variables=y,
            independent_data=np.array([[1.0], [2.0]]),
            dependent_data=np.array([[3.0], [4.0]]),
        )

        assert dataset.independent_variables == (x,)
        assert dataset.dependent_variables == (y,)

    def test_normalizes_integer_matrices_to_float64(self) -> None:
        dataset = Dataset(
            independent_variables=Variable(symbol="x"),
            dependent_variables=Variable(symbol="y"),
            independent_data=np.array([[1], [2]], dtype=np.int32),
            dependent_data=np.array([[3], [4]], dtype=np.int64),
        )

        assert dataset.independent_data.dtype == np.float64
        assert dataset.dependent_data.dtype == np.float64

    def test_copies_input_matrices(self) -> None:
        independent = np.array([[1.0], [2.0]])
        dependent = np.array([[3.0], [4.0]])

        dataset = Dataset(
            independent_variables=Variable(symbol="x"),
            dependent_variables=Variable(symbol="y"),
            independent_data=independent,
            dependent_data=dependent,
        )

        independent[0, 0] = 99.0
        dependent[0, 0] = 88.0

        assert dataset.independent_data[0, 0] == 1.0
        assert dataset.dependent_data[0, 0] == 3.0

    def test_matrices_are_read_only(self) -> None:
        dataset = make_dataset()

        assert dataset.independent_data.flags.writeable is False
        assert dataset.dependent_data.flags.writeable is False

        with pytest.raises(ValueError):
            dataset.independent_data[0, 0] = 99.0

        with pytest.raises(ValueError):
            dataset.dependent_data[0, 0] = 99.0

    def test_reports_counts_from_canonical_matrices(self) -> None:
        dataset = make_dataset()

        assert dataset.sample_count == 3
        assert dataset.independent_variable_count == 1
        assert dataset.dependent_variable_count == 1

    def test_default_metadata_is_empty(self) -> None:
        dataset = make_dataset()

        assert dataset.metadata == {}

    def test_metadata_is_copied_on_input(self) -> None:
        metadata = {"source": "experiment"}
        dataset = make_dataset(metadata=metadata)

        metadata["source"] = "changed"

        assert dataset.metadata == {"source": "experiment"}

    def test_metadata_property_returns_copy(self) -> None:
        dataset = make_dataset(metadata={"source": "experiment"})

        exported = dataset.metadata
        exported["source"] = "changed"

        assert dataset.metadata == {"source": "experiment"}

    @pytest.mark.parametrize(
        "metadata",
        [
            "experiment",
            1,
            ["source", "experiment"],
        ],
    )
    def test_rejects_invalid_metadata(self, metadata) -> None:
        with pytest.raises(
            ValueError,
            match="metadata must be a mapping or None",
        ):
            make_dataset(metadata=metadata)


class TestDatasetInvariants:
    """Tests for constraints enforced by Dataset construction."""

    def test_rejects_same_symbol_for_independent_and_dependent(self) -> None:
        with pytest.raises(
            ValueError,
            match="must use distinct symbols",
        ):
            Dataset(
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
            Dataset(
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
            Dataset(
                independent_variables=Variable(symbol="x"),
                dependent_variables=Variable(symbol="y"),
                independent_data=np.array(
                    [[1.0, 10.0], [2.0, 20.0]],
                ),
                dependent_data=np.array([[3.0], [4.0]]),
            )

    def test_v1_rejects_multiple_independent_variables(self) -> None:
        x1 = Variable(symbol="x_1")
        x2 = Variable(symbol="x_2")

        with pytest.raises(
            ValueError,
            match=r"exactly 1 variable\(s\)",
        ):
            Dataset(
                independent_variables=[x1, x2],
                dependent_variables=Variable(symbol="y"),
                independent_data=np.array(
                    [[1.0, 10.0], [2.0, 20.0]],
                ),
                dependent_data=np.array([[3.0], [4.0]]),
            )

    def test_v1_rejects_multiple_dependent_variables(self) -> None:
        y1 = Variable(symbol="y_1")
        y2 = Variable(symbol="y_2")

        with pytest.raises(
            ValueError,
            match=r"exactly 1 variable\(s\)",
        ):
            Dataset(
                independent_variables=Variable(symbol="x"),
                dependent_variables=[y1, y2],
                independent_data=np.array([[1.0], [2.0]]),
                dependent_data=np.array(
                    [[3.0, 30.0], [4.0, 40.0]],
                ),
            )


class TestColumns:
    """Tests for convenient one-dimensional column access."""

    def test_independent_column_returns_expected_read_only_view(self) -> None:
        dataset = make_dataset()

        column = dataset.independent_column(0)

        np.testing.assert_array_equal(
            column,
            np.array([1.0, 2.0, 3.0]),
        )
        assert column.ndim == 1
        assert column.flags.writeable is False

    def test_dependent_column_returns_expected_read_only_view(self) -> None:
        dataset = make_dataset()

        column = dataset.dependent_column(0)

        np.testing.assert_array_equal(
            column,
            np.array([10.0, 20.0, 30.0]),
        )
        assert column.ndim == 1
        assert column.flags.writeable is False

    @pytest.mark.parametrize("index", [-1, 1])
    def test_independent_column_rejects_out_of_range_index(
        self,
        index,
    ) -> None:
        dataset = make_dataset()

        with pytest.raises(
            IndexError,
            match="independent variable index is out of range",
        ):
            dataset.independent_column(index)

    @pytest.mark.parametrize("index", [-1, 1])
    def test_dependent_column_rejects_out_of_range_index(
        self,
        index,
    ) -> None:
        dataset = make_dataset()

        with pytest.raises(
            IndexError,
            match="dependent variable index is out of range",
        ):
            dataset.dependent_column(index)


class TestCopy:
    """Tests for Dataset copying behavior."""

    def test_copy_creates_distinct_dataset_and_arrays(self) -> None:
        dataset = make_dataset(metadata={"source": "experiment"})

        copied = dataset.copy()

        assert copied is not dataset
        assert copied.independent_data is not dataset.independent_data
        assert copied.dependent_data is not dataset.dependent_data
        np.testing.assert_array_equal(
            copied.independent_data,
            dataset.independent_data,
        )
        np.testing.assert_array_equal(
            copied.dependent_data,
            dataset.dependent_data,
        )
        assert copied.metadata == dataset.metadata

    def test_copy_reuses_variable_objects_by_contract(self) -> None:
        dataset = make_dataset()

        copied = dataset.copy()

        assert (
            copied.independent_variables[0]
            is dataset.independent_variables[0]
        )
        assert (
            copied.dependent_variables[0]
            is dataset.dependent_variables[0]
        )

    def test_copied_matrices_remain_read_only(self) -> None:
        dataset = make_dataset()

        copied = dataset.copy()

        assert copied.independent_data.flags.writeable is False
        assert copied.dependent_data.flags.writeable is False


class TestRepresentation:
    """Tests for developer-oriented Dataset representation."""

    def test_repr_reports_dataset_dimensions(self) -> None:
        dataset = make_dataset()

        representation = repr(dataset)

        assert representation == (
            "Dataset(samples=3, independent_variables=1, "
            "dependent_variables=1)"
        )
