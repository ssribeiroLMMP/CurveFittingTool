"""Map model Parameter objects to solver-oriented numerical vectors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from curvefit.domain.parameter import Parameter


class ParameterMapper:
    """Map model parameters to the numerical representation used by a solver.

    The mapper captures the parameter configuration at construction time.
    Parameters marked with ``vary=False`` are excluded from solver vectors
    and retain their initial values when the complete parameter vector is
    reconstructed.
    """

    def __init__(
        self,
        parameters: Sequence[Parameter],
    ) -> None:
        """Create a mapper from an ordered parameter collection.

        Parameters
        ----------
        parameters:
            Ordered sequence of model parameters.

        Raises
        ------
        ValueError
            If the collection contains an object other than Parameter.
        """
        normalized = tuple(parameters)

        if not all(
            isinstance(parameter, Parameter)
            for parameter in normalized
        ):
            raise ValueError(
                "parameters must contain only Parameter objects."
            )

        self._varying_indices = tuple(
            index
            for index, parameter in enumerate(normalized)
            if parameter.vary
        )

        self._fixed_indices = tuple(
            index
            for index, parameter in enumerate(normalized)
            if not parameter.vary
        )

        self._full_initial_values = np.asarray(
            [
                parameter.initial_value
                for parameter in normalized
            ],
            dtype=np.float64,
        )

        self._initial_values = np.asarray(
            [
                normalized[index].initial_value
                for index in self._varying_indices
            ],
            dtype=np.float64,
        )

        self._bounds = np.asarray(
            [
                [
                    normalized[index].min_value
                    for index in self._varying_indices
                ],
                [
                    normalized[index].max_value
                    for index in self._varying_indices
                ],
            ],
            dtype=np.float64,
        )

        self._full_initial_values.setflags(write=False)
        self._initial_values.setflags(write=False)
        self._bounds.setflags(write=False)

    @property
    def varying_indices(self) -> tuple[int, ...]:
        """Return the indices of parameters optimized by the solver."""
        return self._varying_indices

    @property
    def fixed_indices(self) -> tuple[int, ...]:
        """Return the indices of parameters held fixed during fitting."""
        return self._fixed_indices

    def initial_values(self) -> NDArray[np.float64]:
        """Return the initial solver vector for varying parameters."""
        return self._initial_values

    def bounds(self) -> NDArray[np.float64]:
        """Return lower and upper bounds for varying parameters.

        Returns
        -------
        NDArray[np.float64]
            Matrix with shape ``(2, n_varying_parameters)``. Row 0 contains
            lower bounds and row 1 contains upper bounds.
        """
        return self._bounds

    def expand_values(
        self,
        varying_values: NDArray[np.number],
    ) -> NDArray[np.float64]:
        """Reconstruct the complete ordered parameter vector.

        Parameters
        ----------
        varying_values:
            One-dimensional vector containing one value for each parameter
            optimized by the solver.

        Returns
        -------
        NDArray[np.float64]
            Complete parameter vector in the original model order. Fixed
            parameters use their configured initial values.

        Raises
        ------
        ValueError
            If ``varying_values`` is not one-dimensional or its size does not
            match the number of varying parameters.
        """
        values = np.asarray(
            varying_values,
            dtype=np.float64,
        )

        if values.ndim != 1:
            raise ValueError(
                "varying_values must be one-dimensional."
            )

        if values.size != len(self._varying_indices):
            raise ValueError(
                "varying_values size must match the number "
                "of varying parameters."
            )

        full_values = self._full_initial_values.copy()

        if self._varying_indices:
            full_values[list(self._varying_indices)] = values

        return full_values
