"""Generic paired numerical series used by domain entities.

DataSeries represents the mathematical structure (X, Y) independently of the
origin or meaning of the values. Dataset and model-generated curves can compose
this entity without duplicating matrix and variable handling.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from curvefit.domain.data_series.validator import DataSeriesValidator
from curvefit.domain.variable import Variable


class DataSeries:
    """Represent paired independent and dependent numerical data.

    Parameters
    ----------
    independent_variables:
        One Variable or an ordered sequence describing the columns of
        ``independent_data``.
    dependent_variables:
        One Variable or an ordered sequence describing the columns of
        ``dependent_data``.
    independent_data:
        Two-dimensional numerical matrix whose rows represent samples and
        columns represent independent variables.
    dependent_data:
        Two-dimensional numerical matrix whose rows represent samples and
        columns represent dependent variables.

    Notes
    -----
    Numerical matrices are copied, converted to ``float64`` and made read-only
    during construction. DataSeries does not assign experimental, fitted or
    simulated semantics to the values it stores.
    """

    def __init__(
        self,
        independent_variables: Variable | Sequence[Variable],
        dependent_variables: Variable | Sequence[Variable],
        independent_data: NDArray[np.number],
        dependent_data: NDArray[np.number],
    ) -> None:
        validated_independent_variables = DataSeriesValidator.variables(
            independent_variables,
        )
        validated_dependent_variables = DataSeriesValidator.variables(
            dependent_variables,
        )

        DataSeriesValidator.distinct_symbols(
            validated_independent_variables,
            validated_dependent_variables,
        )

        validated_independent_data = DataSeriesValidator.data_matrix(
            independent_data,
        )
        validated_dependent_data = DataSeriesValidator.data_matrix(
            dependent_data,
        )

        DataSeriesValidator.matching_variable_count(
            validated_independent_variables,
            validated_independent_data,
        )
        DataSeriesValidator.matching_variable_count(
            validated_dependent_variables,
            validated_dependent_data,
        )
        DataSeriesValidator.matching_sample_count(
            validated_independent_data,
            validated_dependent_data,
        )

        validated_independent_data.setflags(write=False)
        validated_dependent_data.setflags(write=False)

        self._independent_variables = validated_independent_variables
        self._dependent_variables = validated_dependent_variables
        self._independent_data = validated_independent_data
        self._dependent_data = validated_dependent_data

    @property
    def independent_variables(self) -> tuple[Variable, ...]:
        """Return the ordered independent Variable objects."""
        return self._independent_variables

    @property
    def dependent_variables(self) -> tuple[Variable, ...]:
        """Return the ordered dependent Variable objects."""
        return self._dependent_variables

    @property
    def independent_data(self) -> NDArray[np.float64]:
        """Return the read-only independent-data matrix."""
        return self._independent_data

    @property
    def dependent_data(self) -> NDArray[np.float64]:
        """Return the read-only dependent-data matrix."""
        return self._dependent_data

    @property
    def sample_count(self) -> int:
        """Return the number of paired samples."""
        return self._independent_data.shape[0]

    @property
    def independent_variable_count(self) -> int:
        """Return the number of independent variables."""
        return self._independent_data.shape[1]

    @property
    def dependent_variable_count(self) -> int:
        """Return the number of dependent variables."""
        return self._dependent_data.shape[1]

    def independent_column(self, index: int) -> NDArray[np.float64]:
        """Return one independent-variable column as a read-only view."""
        if not 0 <= index < self.independent_variable_count:
            raise IndexError("independent variable index is out of range.")
        return self._independent_data[:, index]

    def dependent_column(self, index: int) -> NDArray[np.float64]:
        """Return one dependent-variable column as a read-only view."""
        if not 0 <= index < self.dependent_variable_count:
            raise IndexError("dependent variable index is out of range.")
        return self._dependent_data[:, index]

    def copy(self) -> DataSeries:
        """Return an independent numerical copy of the data series.

        Variable objects are reused because the copy represents the same
        mathematical quantities.
        """
        return DataSeries(
            independent_variables=self._independent_variables,
            dependent_variables=self._dependent_variables,
            independent_data=self._independent_data.copy(),
            dependent_data=self._dependent_data.copy(),
        )

    def __repr__(self) -> str:
        """Return a concise developer-oriented representation."""
        return (
            f"{self.__class__.__name__}("
            f"samples={self.sample_count}, "
            f"independent_variables={self.independent_variable_count}, "
            f"dependent_variables={self.dependent_variable_count})"
        )
