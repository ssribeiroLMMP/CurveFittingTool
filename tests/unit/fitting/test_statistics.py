"""Tests for deterministic fitting statistics."""

from types import MappingProxyType

import numpy as np
import pytest

from curvefit.fitting.statistics import (
    calculate_fit_metrics,
    calculate_residuals,
)


def test_calculates_residuals_rss_rmse_and_r_squared() -> None:
    observed = np.array([[1.0], [2.0], [3.0]])
    predicted = np.array([[1.0], [3.0], [2.0]])

    residuals = calculate_residuals(observed, predicted)
    metrics = calculate_fit_metrics(observed, predicted)

    np.testing.assert_array_equal(residuals, [[0.0], [-1.0], [1.0]])
    assert metrics["rss"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert metrics["r_squared"] == pytest.approx(0.0)


def test_perfect_predictions_have_expected_metrics() -> None:
    observed = np.array([[2.0], [4.0], [6.0]])

    metrics = calculate_fit_metrics(observed, observed)

    assert metrics == {"rss": 0.0, "rmse": 0.0, "r_squared": 1.0}


def test_constant_observations_use_finite_r_squared_convention() -> None:
    observed = np.full((3, 1), 2.0)

    perfect = calculate_fit_metrics(observed, observed)
    imperfect = calculate_fit_metrics(observed, np.full((3, 1), 3.0))

    assert perfect["r_squared"] == 1.0
    assert imperfect["r_squared"] == 0.0


def test_supports_multiple_dependent_columns() -> None:
    observed = np.array([[1.0, 2.0], [3.0, 4.0]])
    predicted = np.array([[1.0, 1.0], [2.0, 4.0]])

    metrics = calculate_fit_metrics(observed, predicted)

    assert metrics["rss"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))


def test_results_are_immutable_and_inputs_are_not_modified() -> None:
    observed = np.array([[1.0], [2.0]])
    predicted = np.array([[1.0], [3.0]])
    original_observed = observed.copy()
    original_predicted = predicted.copy()

    residuals = calculate_residuals(observed, predicted)
    metrics = calculate_fit_metrics(observed, predicted)

    np.testing.assert_array_equal(observed, original_observed)
    np.testing.assert_array_equal(predicted, original_predicted)
    assert residuals.flags.writeable is False
    assert isinstance(metrics, MappingProxyType)
    with pytest.raises(ValueError):
        residuals[0, 0] = 1.0
    with pytest.raises(TypeError):
        metrics["rss"] = 1.0


@pytest.mark.parametrize(
    ("observed", "predicted", "message"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), "two-dimensional"),
        (np.empty((0, 1)), np.empty((0, 1)), "cannot be empty"),
        (np.array([[1.0]]), np.array([[1.0], [2.0]]), "matching shapes"),
        (np.array([[np.nan]]), np.array([[1.0]]), "finite values"),
        (np.array([[1.0]]), np.array([[np.inf]]), "finite values"),
        (np.array([[True]]), np.array([[False]]), "numeric values"),
    ],
)
def test_rejects_invalid_matrices(
    observed: np.ndarray,
    predicted: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_fit_metrics(observed, predicted)


def test_rejects_metrics_that_overflow() -> None:
    observed = np.array([[np.finfo(np.float64).max]])
    predicted = np.array([[-np.finfo(np.float64).max]])

    with pytest.raises(ValueError, match="metrics must be finite"):
        calculate_fit_metrics(observed, predicted)


def test_rejects_residuals_that_overflow() -> None:
    observed = np.array([[np.finfo(np.float64).max]])
    predicted = np.array([[-np.finfo(np.float64).max]])

    with pytest.raises(ValueError, match="residuals must contain only finite"):
        calculate_residuals(observed, predicted)
