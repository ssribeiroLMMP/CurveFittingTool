"""Deterministic residual and goodness-of-fit calculations."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray


def calculate_residuals(
    observed: NDArray[np.number],
    predicted: NDArray[np.number],
) -> NDArray[np.float64]:
    """Return the read-only residual matrix ``observed - predicted``."""
    observed_values, predicted_values = _validated_pair(observed, predicted)
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = observed_values - predicted_values
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must contain only finite values.")
    residuals.setflags(write=False)
    return residuals


def calculate_fit_metrics(
    observed: NDArray[np.number],
    predicted: NDArray[np.number],
) -> Mapping[str, float]:
    """Calculate RSS, RMSE, and R-squared for paired numerical matrices.

    For constant observations, R-squared is defined as ``1.0`` for perfect
    predictions and ``0.0`` otherwise. This keeps the metric finite and
    deterministic while acknowledging that the usual ratio is undefined.
    """
    observed_values, predicted_values = _validated_pair(observed, predicted)

    with np.errstate(over="ignore", invalid="ignore"):
        residuals = observed_values - predicted_values
        rss = float(np.sum(np.square(residuals), dtype=np.float64))
        rmse = float(np.sqrt(rss / residuals.size))
        centered = observed_values - np.mean(observed_values, dtype=np.float64)
        total_sum_of_squares = float(
            np.sum(np.square(centered), dtype=np.float64)
        )

    if not all(np.isfinite(value) for value in (rss, rmse, total_sum_of_squares)):
        raise ValueError("fit metrics must be finite.")

    if total_sum_of_squares == 0.0:
        r_squared = 1.0 if rss == 0.0 else 0.0
    else:
        r_squared = 1.0 - rss / total_sum_of_squares

    if not np.isfinite(r_squared):
        raise ValueError("fit metrics must be finite.")

    return MappingProxyType(
        {
            "rss": rss,
            "rmse": rmse,
            "r_squared": r_squared,
        }
    )


def _validated_pair(
    observed: NDArray[np.number],
    predicted: NDArray[np.number],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and copy paired two-dimensional numerical matrices."""
    observed_values = _validated_matrix(observed, "observed")
    predicted_values = _validated_matrix(predicted, "predicted")

    if observed_values.shape != predicted_values.shape:
        raise ValueError("observed and predicted must have matching shapes.")

    return observed_values, predicted_values


def _validated_matrix(
    values: NDArray[np.number],
    field_name: str,
) -> NDArray[np.float64]:
    """Validate a non-empty finite two-dimensional numerical matrix."""
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{field_name} must be a NumPy array.")
    if values.ndim != 2:
        raise ValueError(f"{field_name} must be two-dimensional.")
    if values.size == 0:
        raise ValueError(f"{field_name} cannot be empty.")
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.bool_)
    ):
        raise ValueError(f"{field_name} must contain numeric values.")

    normalized = values.astype(np.float64, copy=True)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{field_name} must contain only finite values.")
    return normalized
