"""Unit tests for Dataset composition and experimental semantics."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from curvefit.domain.data_series import DataSeries
from curvefit.domain.dataset import Dataset
from curvefit.domain.variable import Variable


def make_series() -> DataSeries:
    return DataSeries(
        independent_variables=Variable("x", name="Time", unit="s"),
        dependent_variables=Variable("y", name="Response", unit="Pa"),
        independent_data=np.array([[1.0], [2.0], [3.0]]),
        dependent_data=np.array([[10.0], [20.0], [30.0]]),
    )


def make_dataset(*, metadata: Mapping[str, Any] | None = None) -> Dataset:
    return Dataset(data_series=make_series(), metadata=metadata)


class TestConstruction:
    def test_accepts_data_series_and_delegates_data_and_variables(self) -> None:
        series = make_series()
        dataset = Dataset(data_series=series)

        assert dataset.data_series is series
        assert dataset.independent_variables == series.independent_variables
        assert dataset.dependent_variables == series.dependent_variables
        assert dataset.independent_variables[0] is series.independent_variables[0]
        assert dataset.dependent_variables[0] is series.dependent_variables[0]
        np.testing.assert_array_equal(dataset.independent_data, series.independent_data)
        np.testing.assert_array_equal(dataset.dependent_data, series.dependent_data)

    @pytest.mark.parametrize("value", [None, "series", 1, {}, np.ones((2, 2))])
    def test_rejects_non_data_series(self, value: object) -> None:
        with pytest.raises(ValueError, match="data_series must be a DataSeries"):
            Dataset(data_series=value)

    def test_data_series_property_cannot_be_reassigned(self) -> None:
        dataset = make_dataset()
        original = dataset.data_series
        with pytest.raises(AttributeError):
            dataset.data_series = make_series()
        assert dataset.data_series is original

    def test_matrices_are_read_only(self) -> None:
        dataset = make_dataset()
        assert not dataset.independent_data.flags.writeable
        assert not dataset.dependent_data.flags.writeable
        with pytest.raises(ValueError):
            dataset.independent_data[0, 0] = 99.0
        with pytest.raises(ValueError):
            dataset.dependent_data[0, 0] = 99.0

    def test_reports_counts_from_canonical_matrices(self) -> None:
        dataset = make_dataset()
        assert dataset.sample_count == 3
        assert dataset.independent_variable_count == 1
        assert dataset.dependent_variable_count == 1

    def test_accepts_one_observation(self) -> None:
        series = DataSeries(
            Variable("x"), Variable("y"), np.array([[1.0]]), np.array([[2.0]])
        )
        dataset = Dataset(series)
        assert dataset.sample_count == 1
        assert dataset.independent_data.shape == (1, 1)
        assert dataset.dependent_data.shape == (1, 1)

    def test_default_metadata_is_empty(self) -> None:
        assert make_dataset().metadata == {}

    def test_accepts_non_dict_mapping_metadata(self) -> None:
        metadata = MappingProxyType({"source": "experiment"})
        assert make_dataset(metadata=metadata).metadata == dict(metadata)

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

    @pytest.mark.parametrize("metadata", ["experiment", 1, ["source", "experiment"]])
    def test_rejects_invalid_metadata(self, metadata: object) -> None:
        with pytest.raises(ValueError, match="metadata must be a mapping or None"):
            make_dataset(metadata=metadata)


class TestDatasetInvariants:
    def test_v1_rejects_multiple_independent_variables(self) -> None:
        series = DataSeries(
            independent_variables=[Variable("x_1"), Variable("x_2")],
            dependent_variables=Variable("y"),
            independent_data=np.array([[1.0, 10.0], [2.0, 20.0]]),
            dependent_data=np.array([[3.0], [4.0]]),
        )
        with pytest.raises(ValueError, match="unsupported number of independent variables"):
            Dataset(series)

    def test_v1_rejects_multiple_dependent_variables(self) -> None:
        series = DataSeries(
            independent_variables=Variable("x"),
            dependent_variables=[Variable("y_1"), Variable("y_2")],
            independent_data=np.array([[1.0], [2.0]]),
            dependent_data=np.array([[3.0, 30.0], [4.0, 40.0]]),
        )
        with pytest.raises(ValueError, match="unsupported number of dependent variables"):
            Dataset(series)


class TestColumns:
    @pytest.mark.parametrize(
        "method, expected",
        [("independent_column", [1.0, 2.0, 3.0]),
         ("dependent_column", [10.0, 20.0, 30.0])],
    )
    def test_column_returns_expected_read_only_view(self, method: str, expected: list[float]) -> None:
        column = getattr(make_dataset(), method)(0)
        np.testing.assert_array_equal(column, expected)
        assert column.ndim == 1
        assert not column.flags.writeable

    @pytest.mark.parametrize("index", [-1, 1])
    @pytest.mark.parametrize("role", ["independent", "dependent"])
    def test_column_rejects_out_of_range_index(self, role: str, index: int) -> None:
        dataset = make_dataset()
        with pytest.raises(IndexError, match=f"{role} variable index is out of range"):
            getattr(dataset, f"{role}_column")(index)


class TestCopy:
    def test_copy_creates_distinct_dataset_series_and_arrays(self) -> None:
        dataset = make_dataset(metadata={"source": "experiment"})
        copied = dataset.copy()
        assert copied is not dataset
        assert copied.data_series is not dataset.data_series
        assert not np.shares_memory(copied.independent_data, dataset.independent_data)
        assert not np.shares_memory(copied.dependent_data, dataset.dependent_data)
        np.testing.assert_array_equal(copied.independent_data, dataset.independent_data)
        np.testing.assert_array_equal(copied.dependent_data, dataset.dependent_data)
        assert copied.metadata == dataset.metadata

    def test_copy_reuses_variable_objects_by_contract(self) -> None:
        dataset = make_dataset()
        copied = dataset.copy()
        assert copied.independent_variables[0] is dataset.independent_variables[0]
        assert copied.dependent_variables[0] is dataset.dependent_variables[0]

    def test_copied_matrices_remain_read_only(self) -> None:
        copied = make_dataset().copy()
        assert not copied.independent_data.flags.writeable
        assert not copied.dependent_data.flags.writeable


class TestRepresentation:
    def test_repr_reports_dataset_dimensions(self) -> None:
        assert repr(make_dataset()) == (
            "Dataset(samples=3, independent_variables=1, dependent_variables=1)"
        )
