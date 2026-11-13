"""Presentation-oriented model curve composed from a DataSeries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from curvefit.domain.data_series import DataSeries
from curvefit.domain.variable import Variable


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ModelCurve:
    """Represent model predictions evaluated on a presentation grid."""

    _data_series: DataSeries = field(repr=False)

    def __init__(self, data_series: DataSeries) -> None:
        if not isinstance(data_series, DataSeries):
            raise ValueError("data_series must be a DataSeries object.")

        object.__setattr__(
            self,
            "_data_series",
            _copy_data_series(data_series),
        )

    @property
    def data_series(self) -> DataSeries:
        """Return an independent copy of the composed data series."""
        return _copy_data_series(self._data_series)

    @property
    def independent_variables(self) -> tuple[Variable, ...]:
        """Return independent copies of the curve input variables."""
        return tuple(
            variable.copy()
            for variable in self._data_series.independent_variables
        )

    @property
    def dependent_variables(self) -> tuple[Variable, ...]:
        """Return independent copies of the curve output variables."""
        return tuple(
            variable.copy()
            for variable in self._data_series.dependent_variables
        )

    @property
    def independent_data(self) -> NDArray[np.float64]:
        """Return the read-only presentation input matrix."""
        return self._data_series.independent_data

    @property
    def dependent_data(self) -> NDArray[np.float64]:
        """Return the read-only presentation prediction matrix."""
        return self._data_series.dependent_data

    @property
    def sample_count(self) -> int:
        """Return the number of points in the presentation curve."""
        return self._data_series.sample_count


def _copy_data_series(data_series: DataSeries) -> DataSeries:
    """Copy a data series without retaining mutable Variable objects."""
    return DataSeries(
        independent_variables=tuple(
            variable.copy()
            for variable in data_series.independent_variables
        ),
        dependent_variables=tuple(
            variable.copy()
            for variable in data_series.dependent_variables
        ),
        independent_data=data_series.independent_data,
        dependent_data=data_series.dependent_data,
    )
