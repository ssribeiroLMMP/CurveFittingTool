"""Tests for FitResult."""

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from uuid import UUID, uuid4

import numpy as np
import pytest
from sympy import symbols

from curvefit.domain.data_series import DataSeries
from curvefit.domain.fit_result import FitResult, FitStatus
from curvefit.domain.model_curve import ModelCurve
from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.parameter import Parameter
from curvefit.domain.parameter_result import ParameterResult
from curvefit.domain.snapshots import ModelSnapshot
from curvefit.domain.variable import Variable


def _make_model() -> ModelDefinition:
    x, a, b = symbols("x a b")
    return ModelDefinition(
        name="Linear",
        independent_variables=Variable("x", name="Distance", unit="m"),
        dependent_variables=Variable("y", name="Response", unit="Pa"),
        parameters=(
            Parameter("a", name="Slope", initial_value=1.0),
            Parameter("b", name="Intercept", initial_value=0.0, vary=False),
        ),
        symbolic_expressions=a * x + b,
        numeric_function=lambda inputs, values: inputs,
    )


def _make_series(*, dense: bool = False) -> DataSeries:
    independent = (
        np.linspace(0.0, 2.0, 5).reshape(-1, 1)
        if dense
        else np.array([[0.0], [1.0], [2.0]])
    )
    return DataSeries(
        independent_variables=Variable("x", name="Distance", unit="m"),
        dependent_variables=Variable("y", name="Response", unit="Pa"),
        independent_data=independent,
        dependent_data=2.0 * independent + 1.0,
    )


def _make_result(**overrides: object) -> FitResult:
    model = _make_model()
    predicted_series = _make_series()
    arguments = {
        "model_snapshot": ModelSnapshot.from_model(model),
        "parameter_results": (
            ParameterResult("a", 2.0, 0.1),
            ParameterResult("b", 1.0, None),
        ),
        "predicted_series": predicted_series,
        "model_curve": ModelCurve(_make_series(dense=True)),
        "residuals": np.zeros_like(predicted_series.dependent_data),
        "covariance": np.array([[0.01]]),
        "metrics": {"rss": 0.0, "rmse": 0.0, "r_squared": 1.0},
        "status": FitStatus.SUCCESS,
        "message": "Fit completed",
        "metadata": {"method": "least_squares"},
    }
    arguments.update(overrides)
    return FitResult(**arguments)


def test_stores_complete_execution_record() -> None:
    result = _make_result()

    assert isinstance(result.fit_id, UUID)
    assert result.model_snapshot.name == "Linear"
    assert tuple(item.symbol for item in result.parameter_results) == ("a", "b")
    assert result.predicted_series.sample_count == 3
    assert result.model_curve.sample_count == 5
    assert result.metrics == {"rss": 0.0, "rmse": 0.0, "r_squared": 1.0}
    assert result.status is FitStatus.SUCCESS
    assert result.success is True
    assert result.message == "Fit completed"
    assert result.metadata == {"method": "least_squares"}


def test_generates_a_distinct_fit_id_for_each_execution() -> None:
    assert _make_result().fit_id != _make_result().fit_id


def test_accepts_controlled_restoration_of_fit_id() -> None:
    fit_id = uuid4()

    assert _make_result(fit_id=fit_id).fit_id == fit_id


def test_copies_arrays_and_keeps_them_read_only() -> None:
    residuals = np.zeros((3, 1))
    covariance = np.array([[0.01]])
    result = _make_result(residuals=residuals, covariance=covariance)

    residuals[0, 0] = 9.0
    covariance[0, 0] = 9.0

    assert result.residuals[0, 0] == 0.0
    assert result.covariance[0, 0] == 0.01
    with pytest.raises(ValueError):
        result.residuals[0, 0] = 1.0
    with pytest.raises(ValueError):
        result.covariance[0, 0] = 1.0


def test_copies_predicted_series_and_its_variables() -> None:
    series = _make_series()
    result = _make_result(predicted_series=series)

    series.independent_variables[0].configure(name="Edited")
    exported = result.predicted_series
    exported.independent_variables[0].configure(name="Also edited")

    assert result.predicted_series.independent_variables[0].name == "Distance"


def test_metrics_are_immutable_and_metadata_is_copied() -> None:
    metrics = {"rss": 0.0}
    metadata = {"method": "least_squares", "tags": ["synthetic"]}
    result = _make_result(metrics=metrics, metadata=metadata)

    metrics["rss"] = 4.0
    metadata["method"] = "edited"
    metadata["tags"].append("edited")
    exported_metadata = result.metadata
    exported_metadata["tags"].append("exported")

    assert result.metrics["rss"] == 0.0
    assert result.metadata["method"] == "least_squares"
    assert result.metadata["tags"] == ["synthetic"]
    assert isinstance(result.metrics, MappingProxyType)
    with pytest.raises(TypeError):
        result.metrics["rss"] = 2.0


def test_record_is_immutable() -> None:
    result = _make_result()

    with pytest.raises(FrozenInstanceError):
        result._status = FitStatus.FAILURE


def test_failure_status_drives_success_property() -> None:
    result = _make_result(status="failure", message="Solver stopped")

    assert result.status is FitStatus.FAILURE
    assert result.success is False


def test_parameter_results_must_match_snapshot_order() -> None:
    with pytest.raises(ValueError, match="must match model snapshot"):
        _make_result(
            parameter_results=(
                ParameterResult("b", 1.0),
                ParameterResult("a", 2.0),
            )
        )


def test_series_symbols_must_match_snapshot() -> None:
    series = DataSeries(
        independent_variables=Variable("t"),
        dependent_variables=Variable("y"),
        independent_data=np.array([[0.0], [1.0], [2.0]]),
        dependent_data=np.array([[1.0], [3.0], [5.0]]),
    )

    with pytest.raises(ValueError, match="symbols must match"):
        _make_result(predicted_series=series)


def test_residual_shape_must_match_predictions() -> None:
    with pytest.raises(ValueError, match="dependent-data shape"):
        _make_result(residuals=np.zeros((2, 1)))


def test_covariance_shape_must_match_varying_parameters() -> None:
    with pytest.raises(ValueError, match="each varying parameter"):
        _make_result(covariance=np.eye(2))


@pytest.mark.parametrize(
    "metrics",
    [
        {"": 1.0},
        {"rss": np.inf},
        {"rss": True},
        {"rss": 1.0, " rss ": 2.0},
    ],
)
def test_rejects_invalid_metrics(metrics: object) -> None:
    with pytest.raises(ValueError):
        _make_result(metrics=metrics)


def test_rejects_invalid_fit_id() -> None:
    with pytest.raises(ValueError, match="fit_id"):
        _make_result(fit_id="not-a-uuid")
