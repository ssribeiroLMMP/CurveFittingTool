"""Helpers for building vectorized numerical model functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sympy import Expr, Symbol, lambdify

NumericalModelFunction = Callable[
    [ArrayLike, ArrayLike],
    NDArray[np.float64],
]


def build_numeric_function(
    expression: Expr,
    independent_symbol: Symbol,
    parameter_symbols: Sequence[Symbol],
) -> NumericalModelFunction:
    """Build a vectorized numerical function from a symbolic expression.

    The returned function accepts an ``(n, 1)`` input matrix and a
    one-dimensional parameter vector ordered as ``parameter_symbols``. It
    returns an ``(n, 1)`` prediction matrix.
    """
    parameter_symbols = tuple(parameter_symbols)
    evaluate_expression = lambdify(
        (independent_symbol, *parameter_symbols),
        expression,
        modules="numpy",
    )

    def evaluate(
        independent_data: ArrayLike,
        parameter_values: ArrayLike,
    ) -> NDArray[np.float64]:
        inputs = np.asarray(independent_data, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != 1:
            raise ValueError(
                "independent_data must have shape (sample_count, 1)."
            )

        parameters = np.asarray(parameter_values, dtype=np.float64)
        if parameters.ndim != 1 or parameters.size != len(parameter_symbols):
            raise ValueError(
                "parameter_values must be one-dimensional and contain "
                "one value for each model parameter."
            )

        predictions = np.asarray(
            evaluate_expression(inputs[:, 0], *parameters),
            dtype=np.float64,
        )
        if predictions.ndim == 0:
            predictions = np.full(inputs.shape[0], predictions.item())
        if predictions.shape != (inputs.shape[0],):
            raise ValueError(
                "numeric model function must return one value per input row."
            )

        return predictions.reshape(-1, 1)

    return evaluate
