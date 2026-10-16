"""Validation rules for generic numerical data series.

A DataSeries stores paired independent and dependent numerical matrices using
canonical two-dimensional NumPy arrays. Rows represent corresponding samples
and columns represent mathematical variables.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from curvefit.domain.variable import Variable


class DataSeriesValidator:
    """Validate variables and numerical matrices used by DataSeries."""

    @staticmethod
    def variables(
        values: Variable | Sequence[Variable],
        *,
        expected_count: int | None = None,
    ) -> tuple[Variable, ...]:
        """Validate and normalize one or more mathematical variables."""
        if isinstance(values, Variable):
            normalized = (values,)
        elif (
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
        ):
            normalized = tuple(values)
        else:
            raise ValueError(
                "variables must be a Variable or a sequence of "
                "Variable objects."
            )

        if not normalized:
            raise ValueError("variable collection cannot be empty.")

        if not all(isinstance(variable, Variable) for variable in normalized):
            raise ValueError(
                "variable collection must contain only Variable objects."
            )

        symbols = tuple(variable.symbol for variable in normalized)
        if len(set(symbols)) != len(symbols):
            raise ValueError(
                "variable collection cannot contain duplicate symbols."
            )

        if expected_count is not None and len(normalized) != expected_count:
            raise ValueError(
                "variable collection must contain exactly "
                f"{expected_count} variable(s)."
            )

        return normalized

    @staticmethod
    def data_matrix(
        values: NDArray[np.number],
    ) -> NDArray[np.float64]:
        """Validate and normalize a two-dimensional numerical matrix."""
        if not isinstance(values, np.ndarray):
            raise ValueError("data matrix must be a NumPy array.")

        if values.ndim != 2:
            raise ValueError("data matrix must be two-dimensional.")

        if values.shape[0] == 0:
            raise ValueError(
                "data matrix must contain at least one observation."
            )

        if values.shape[1] == 0:
            raise ValueError("data matrix must contain at least one variable.")

        if (
            not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            raise ValueError("data matrix must contain numeric values.")

        normalized = values.astype(np.float64, copy=True)

        if not np.all(np.isfinite(normalized)):
            raise ValueError("data matrix must contain only finite values.")

        return normalized

    @staticmethod
    def matching_variable_count(
        variables: tuple[Variable, ...],
        data: NDArray[np.float64],
    ) -> None:
        """Validate one-to-one correspondence between variables and columns."""
        if data.shape[1] != len(variables):
            raise ValueError(
                "data matrix must contain one column per variable."
            )

    @staticmethod
    def matching_sample_count(
        independent_data: NDArray[np.float64],
        dependent_data: NDArray[np.float64],
    ) -> None:
        """Validate that paired matrices contain the same number of samples."""
        if independent_data.shape[0] != dependent_data.shape[0]:
            raise ValueError(
                "independent and dependent data must contain the same "
                "number of observations."
            )

    @staticmethod
    def distinct_symbols(
        independent_variables: tuple[Variable, ...],
        dependent_variables: tuple[Variable, ...],
    ) -> None:
        """Validate that independent and dependent symbols do not overlap."""
        independent_symbols = {
            variable.symbol for variable in independent_variables
        }
        dependent_symbols = {
            variable.symbol for variable in dependent_variables
        }

        if independent_symbols & dependent_symbols:
            raise ValueError(
                "independent and dependent variables must use distinct symbols."
            )
