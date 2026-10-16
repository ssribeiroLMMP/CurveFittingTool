"""Domain entity representing an experimental dataset.

Dataset assigns experimental semantics and optional metadata to a generic
DataSeries. Numerical structure and variable-to-column relationships belong to
DataSeries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from curvefit.domain.data_series import DataSeries
from curvefit.domain.dataset.validator import DatasetValidator
from curvefit.domain.variable import Variable


class Dataset:
    """Represent experimental data through a validated DataSeries.

    Parameters
    ----------
    data_series:
        Generic paired numerical series containing the experimental variables
        and values.
    metadata:
        Optional application-defined metadata copied into the Dataset.

    Notes
    -----
    Version 1 supports exactly one independent and one dependent variable.
    DataSeries itself remains dimensionally general so it can also represent
    future multidimensional data and model-generated curves.
    """

    SUPPORTED_INDEPENDENT_VARIABLE_COUNT = 1
    SUPPORTED_DEPENDENT_VARIABLE_COUNT = 1

    def __init__(
        self,
        data_series: DataSeries,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        validated_series = DatasetValidator.data_series(
            data_series,
            expected_independent_count=self.SUPPORTED_INDEPENDENT_VARIABLE_COUNT,
            expected_dependent_count=self.SUPPORTED_DEPENDENT_VARIABLE_COUNT,
        )

        self._data_series = validated_series
        self._metadata = self._normalize_metadata(metadata)

    @property
    def data_series(self) -> DataSeries:
        """Return the experimental DataSeries."""
        return self._data_series

    @property
    def independent_variables(self) -> tuple[Variable, ...]:
        """Return the ordered independent Variable objects."""
        return self._data_series.independent_variables

    @property
    def dependent_variables(self) -> tuple[Variable, ...]:
        """Return the ordered dependent Variable objects."""
        return self._data_series.dependent_variables

    @property
    def independent_data(self) -> NDArray[np.float64]:
        """Return the read-only independent-data matrix."""
        return self._data_series.independent_data

    @property
    def dependent_data(self) -> NDArray[np.float64]:
        """Return the read-only dependent-data matrix."""
        return self._data_series.dependent_data

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a shallow copy of the Dataset metadata."""
        return self._metadata.copy()

    @property
    def sample_count(self) -> int:
        """Return the number of experimental observations."""
        return self._data_series.sample_count

    @property
    def independent_variable_count(self) -> int:
        """Return the number of independent variables."""
        return self._data_series.independent_variable_count

    @property
    def dependent_variable_count(self) -> int:
        """Return the number of dependent variables."""
        return self._data_series.dependent_variable_count

    def independent_column(self, index: int) -> NDArray[np.float64]:
        """Return one independent-variable data column."""
        return self._data_series.independent_column(index)

    def dependent_column(self, index: int) -> NDArray[np.float64]:
        """Return one dependent-variable data column."""
        return self._data_series.dependent_column(index)

    def copy(self) -> Dataset:
        """Create an independent numerical copy of the Dataset."""
        return Dataset(
            data_series=self._data_series.copy(),
            metadata=self._metadata.copy(),
        )

    @staticmethod
    def _normalize_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate and copy optional Dataset metadata."""
        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping or None.")

        return dict(metadata)

    def __repr__(self) -> str:
        """Return a concise developer-oriented Dataset representation."""
        return (
            f"{self.__class__.__name__}("
            f"samples={self.sample_count}, "
            f"independent_variables={self.independent_variable_count}, "
            f"dependent_variables={self.dependent_variable_count})"
        )
