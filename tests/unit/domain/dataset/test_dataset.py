"""
Check how Dataset combines a data series with experiment information.

make_series and make_dataset create fresh examples on every call. Each test
prepares an example, performs an operation, and checks the result. The classes
below group tests about the same subject.

NumPy creates and compares arrays. pytest runs the tests; parametrize repeats a
test for each listed input. pytest.raises requires the expected error, and
match checks part of its message. assert checks a condition and fails the test
if it is false. Mapping and Any describe accepted helper arguments.
MappingProxyType provides a read-only dictionary to check that experiment
information can also come from other collections of keys and values.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from curvefit.domain.data_series import DataSeries
from curvefit.domain.dataset import Dataset
from curvefit.domain.variable import Variable


def make_series() -> DataSeries:
    """
    Create three pairs of values, with time as input and response as output.
    Each call creates a new series so that tests do not depend on each other.
    """
    return DataSeries(
        independent_variables=Variable("x", name="Time", unit="s"),
        dependent_variables=Variable("y", name="Response", unit="Pa"),
        independent_data=np.array([[1.0], [2.0], [3.0]]),
        dependent_data=np.array([[10.0], [20.0], [30.0]]),
    )


def make_dataset(*, metadata: Mapping[str, Any] | None = None) -> Dataset:
    """
    Create a Dataset with the example series and the supplied experiment
    information. Tests use this helper when they do not need different
    numerical data.
    """
    return Dataset(data_series=make_series(), metadata=metadata)


class TestConstruction:
    """Check Dataset creation, numerical data, and experiment information."""
    def test_accepts_data_series_and_delegates_data_and_variables(self) -> None:
        """
        Create a Dataset from an existing series. Check that it keeps that
        series and exposes its variables and values.
        """
        series = make_series()
        dataset = Dataset(data_series=series)

        # "is" checks for the same object; "==" compares values.
        assert dataset.data_series is series
        assert dataset.independent_variables == series.independent_variables
        assert dataset.dependent_variables == series.dependent_variables
        assert dataset.independent_variables[0] is series.independent_variables[0]
        assert dataset.dependent_variables[0] is series.dependent_variables[0]
        np.testing.assert_array_equal(dataset.independent_data, series.independent_data)
        np.testing.assert_array_equal(dataset.dependent_data, series.dependent_data)

    @pytest.mark.parametrize("value", [None, "series", 1, {}, np.ones((2, 2))])
    def test_rejects_non_data_series(self, value: object) -> None:
        """
        Try creating a Dataset with each listed value. Each must raise
        ValueError because it is not a DataSeries.
        """
        with pytest.raises(ValueError, match="data_series must be a DataSeries"):
            Dataset(data_series=value)

    def test_data_series_property_cannot_be_reassigned(self) -> None:
        """
        Try replacing the series through the public property. Expect
        AttributeError and check that the original series is still stored.
        """
        dataset = make_dataset()
        original = dataset.data_series
        with pytest.raises(AttributeError):
            dataset.data_series = make_series()
        assert dataset.data_series is original

    def test_matrices_are_read_only(self) -> None:
        """
        Check that writing is disabled and that direct assignments to both
        arrays raise ValueError. This does not test re-enabling writing with
        setflags.
        """
        dataset = make_dataset()
        assert not dataset.independent_data.flags.writeable
        assert not dataset.dependent_data.flags.writeable
        with pytest.raises(ValueError):
            dataset.independent_data[0, 0] = 99.0
        with pytest.raises(ValueError):
            dataset.dependent_data[0, 0] = 99.0

    def test_reports_counts_from_canonical_matrices(self) -> None:
        """
        Check that the example reports three observations, one input variable,
        and one output variable.
        """
        dataset = make_dataset()
        assert dataset.sample_count == 3
        assert dataset.independent_variable_count == 1
        assert dataset.dependent_variable_count == 1

    def test_accepts_one_observation(self) -> None:
        """
        Create a series containing just one pair of values. Check that Dataset
        accepts it and keeps arrays with one row and one column.
        """
        series = DataSeries(
            Variable("x"), Variable("y"), np.array([[1.0]]), np.array([[2.0]])
        )
        dataset = Dataset(series)
        assert dataset.sample_count == 1
        assert dataset.independent_data.shape == (1, 1)
        assert dataset.dependent_data.shape == (1, 1)

    def test_default_metadata_is_empty(self) -> None:
        """Check that omitted experiment information produces an empty dictionary."""
        assert make_dataset().metadata == {}

    def test_accepts_non_dict_mapping_metadata(self) -> None:
        """
        Supply a read-only collection of experiment information. Check that
        Dataset accepts its keys and values.
        """
        metadata = MappingProxyType({"source": "experiment"})
        assert make_dataset(metadata=metadata).metadata == dict(metadata)

    def test_metadata_is_copied_on_input(self) -> None:
        """
        Change the supplied dictionary after creating the Dataset. Check that
        the stored experiment information keeps its original value.
        """
        metadata = {"source": "experiment"}
        dataset = make_dataset(metadata=metadata)
        # Change only the dictionary used to prepare the test.
        metadata["source"] = "changed"
        assert dataset.metadata == {"source": "experiment"}

    def test_metadata_property_returns_copy(self) -> None:
        """
        Change the dictionary returned by metadata. Check that the experiment
        information stored in Dataset remains unchanged.
        """
        dataset = make_dataset(metadata={"source": "experiment"})
        exported = dataset.metadata
        # Change the returned dictionary, then read Dataset metadata again.
        exported["source"] = "changed"
        assert dataset.metadata == {"source": "experiment"}

    @pytest.mark.parametrize("metadata", ["experiment", 1, ["source", "experiment"]])
    def test_rejects_invalid_metadata(self, metadata: object) -> None:
        """
        Try text, a number, and a list as experiment information. Expect an
        error for each input because none provides a dictionary-like collection
        of keys and values.
        """
        with pytest.raises(ValueError, match="metadata must be a mapping or None"):
            make_dataset(metadata=metadata)


class TestDatasetInvariants:
    """Check the current limit of one input variable and one output variable."""
    def test_v1_rejects_multiple_independent_variables(self) -> None:
        # The series accepts two inputs; Dataset must reject that count.
        """
        Prepare a valid series with two input variables. Check that Dataset
        rejects it because this version allows only one input.
        """
        series = DataSeries(
            independent_variables=[Variable("x_1"), Variable("x_2")],
            dependent_variables=Variable("y"),
            independent_data=np.array([[1.0, 10.0], [2.0, 20.0]]),
            dependent_data=np.array([[3.0], [4.0]]),
        )
        with pytest.raises(ValueError, match="unsupported number of independent variables"):
            Dataset(series)

    def test_v1_rejects_multiple_dependent_variables(self) -> None:
        """
        Prepare a valid series with two output variables. Check that Dataset
        rejects it because this version allows only one output.
        """
        series = DataSeries(
            independent_variables=Variable("x"),
            dependent_variables=[Variable("y_1"), Variable("y_2")],
            independent_data=np.array([[1.0], [2.0]]),
            dependent_data=np.array([[3.0, 30.0], [4.0, 40.0]]),
        )
        with pytest.raises(ValueError, match="unsupported number of dependent variables"):
            Dataset(series)


class TestColumns:
    """Check column access and rejection of positions that do not exist."""
    @pytest.mark.parametrize(
        "method, expected",
        [("independent_column", [1.0, 2.0, 3.0]),
         ("dependent_column", [10.0, 20.0, 30.0])],
    )
    def test_column_returns_expected_read_only_view(self, method: str, expected: list[float]) -> None:
        """
        Run the same check for the input column and the output column. Verify
        the values, one-dimensional shape, and disabled writing.
        """
        # getattr selects the method using the name supplied for this case.
        column = getattr(make_dataset(), method)(0)
        np.testing.assert_array_equal(column, expected)
        assert column.ndim == 1
        assert not column.flags.writeable

    # The two lists produce four cases: two indices for each side.
    @pytest.mark.parametrize("index", [-1, 1])
    @pytest.mark.parametrize("role", ["independent", "dependent"])
    def test_column_rejects_out_of_range_index(self, role: str, index: int) -> None:
        """
        Try positions -1 and 1 for both input and output columns. Only column 0
        exists, so every attempt must raise IndexError.
        """
        dataset = make_dataset()
        with pytest.raises(IndexError, match=f"{role} variable index is out of range"):
            getattr(dataset, f"{role}_column")(index)


class TestCopy:
    """Check what is copied and what is shared when copying a Dataset."""
    def test_copy_creates_distinct_dataset_series_and_arrays(self) -> None:
        """
        Copy a Dataset and check that the new objects and arrays are separate
        from the originals while preserving their values and experiment
        information.
        """
        dataset = make_dataset(metadata={"source": "experiment"})
        copied = dataset.copy()
        assert copied is not dataset
        assert copied.data_series is not dataset.data_series
        # Check that the arrays do not share the same memory.
        assert not np.shares_memory(copied.independent_data, dataset.independent_data)
        assert not np.shares_memory(copied.dependent_data, dataset.dependent_data)
        np.testing.assert_array_equal(copied.independent_data, dataset.independent_data)
        np.testing.assert_array_equal(copied.dependent_data, dataset.dependent_data)
        assert copied.metadata == dataset.metadata

    def test_copy_reuses_variable_objects_by_contract(self) -> None:
        """
        Check that the copy uses the same Variable objects, since they describe
        the same input and output quantities.
        """
        dataset = make_dataset()
        copied = dataset.copy()
        assert copied.independent_variables[0] is dataset.independent_variables[0]
        assert copied.dependent_variables[0] is dataset.dependent_variables[0]

    def test_copied_matrices_remain_read_only(self) -> None:
        """Check that copying a Dataset does not enable writing to the copied arrays."""
        copied = make_dataset().copy()
        assert not copied.independent_data.flags.writeable
        assert not copied.dependent_data.flags.writeable


class TestRepresentation:
    """Check the text that describes the dimensions of a Dataset."""
    def test_repr_reports_dataset_dimensions(self) -> None:
        """
        Compare repr output with the expected description: three observations,
        one input, and one output.
        """
        assert repr(make_dataset()) == (
            "Dataset(samples=3, independent_variables=1, dependent_variables=1)"
        )
