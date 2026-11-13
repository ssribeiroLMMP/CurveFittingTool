"""Unit tests for LmfitEngine behavior and failure handling."""

from types import SimpleNamespace

import numpy as np
import pytest
from sympy import symbols

from curvefit.domain.data_series import DataSeries
from curvefit.domain.dataset import Dataset
from curvefit.domain.exceptions import FitError
from curvefit.domain.fit_result import FitResult, FitStatus
from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.parameter import Parameter
from curvefit.domain.variable import Variable
from curvefit.fitting import LmfitEngine
from curvefit.models.linear import create_linear_model


def _linear_dataset(
    *,
    slope: float = 3.0,
    intercept: float = 1.0,
) -> Dataset:
    independent = np.linspace(-2.0, 2.0, 9).reshape(-1, 1)
    deterministic_noise = 0.001 * (
        np.square(independent) - np.mean(np.square(independent))
    )
    return Dataset(
        DataSeries(
            independent_variables=Variable("x"),
            dependent_variables=Variable("y"),
            independent_data=independent,
            dependent_data=(
                slope * independent
                + intercept
                + deterministic_noise
            ),
        )
    )


def test_fits_linear_data_and_returns_complete_result() -> None:
    model = create_linear_model()

    result = LmfitEngine(curve_point_count=51).fit(
        _linear_dataset(),
        model,
    )

    assert isinstance(result, FitResult)
    assert result.status is FitStatus.SUCCESS
    assert result.success is True
    assert result.model_snapshot.model_id == model.model_id
    assert result.parameter_results[0].value == pytest.approx(3.0)
    assert result.parameter_results[1].value == pytest.approx(1.0)
    assert all(item.stderr is not None for item in result.parameter_results)
    assert result.covariance is not None
    assert result.covariance.shape == (2, 2)
    assert result.predicted_series.sample_count == 9
    assert result.model_curve.sample_count == 51
    assert result.metrics["rss"] > 0.0
    assert result.metrics["rmse"] < 0.01
    assert result.metadata["engine"] == "lmfit"


def test_preserves_fixed_parameter_and_excludes_it_from_covariance() -> None:
    model = create_linear_model()
    model.parameters[1].configure(initial_value=1.0, vary=False)

    result = LmfitEngine().fit(_linear_dataset(), model)

    assert result.parameter_results[0].value == pytest.approx(3.0)
    assert result.parameter_results[1].value == 1.0
    assert result.parameter_results[1].stderr is None
    assert result.parameter_results[0].stderr is not None
    assert result.covariance is not None
    assert result.covariance.shape == (1, 1)


def test_honors_parameter_bounds() -> None:
    model = create_linear_model()
    model.parameters[0].configure(
        initial_value=1.0,
        min_value=0.0,
        max_value=2.0,
    )
    model.parameters[1].configure(initial_value=1.0, vary=False)

    result = LmfitEngine().fit(_linear_dataset(slope=3.0), model)

    assert result.parameter_results[0].value == pytest.approx(2.0, abs=1e-6)
    assert result.parameter_results[0].value <= 2.0
    assert result.parameter_results[1].value == 1.0


def test_evaluates_model_when_all_parameters_are_fixed() -> None:
    model = create_linear_model()
    model.parameters[0].configure(initial_value=3.0, vary=False)
    model.parameters[1].configure(initial_value=1.0, vary=False)

    result = LmfitEngine(curve_point_count=10).fit(
        _linear_dataset(),
        model,
    )

    assert tuple(item.value for item in result.parameter_results) == (3.0, 1.0)
    assert all(item.stderr is None for item in result.parameter_results)
    assert result.covariance is None
    assert result.metrics["rss"] > 0.0
    assert result.metadata["method"] == "fixed_parameters"


def test_does_not_write_fitted_values_into_model_parameters() -> None:
    model = create_linear_model()
    original_values = tuple(
        parameter.initial_value
        for parameter in model.parameters
    )

    LmfitEngine().fit(_linear_dataset(), model)

    assert tuple(
        parameter.initial_value
        for parameter in model.parameters
    ) == original_values


def test_rejects_dataset_with_different_variable_symbols() -> None:
    independent = np.array([[0.0], [1.0]])
    dataset = Dataset(
        DataSeries(
            independent_variables=Variable("t"),
            dependent_variables=Variable("y"),
            independent_data=independent,
            dependent_data=independent,
        )
    )

    with pytest.raises(FitError, match="variable symbols must match"):
        LmfitEngine().fit(dataset, create_linear_model())


@pytest.mark.parametrize(
    ("numeric_function", "message"),
    [
        (lambda inputs, values: inputs[:, 0], "returned shape"),
        (
            lambda inputs, values: np.full(inputs.shape, np.nan),
            "non-finite predictions",
        ),
    ],
)
def test_reports_invalid_model_output(
    numeric_function,
    message: str,
) -> None:
    x, a = symbols("x a")
    model = ModelDefinition(
        name="Invalid output",
        independent_variables=Variable("x"),
        dependent_variables=Variable("y"),
        parameters=Parameter("a"),
        symbolic_expressions=a * x,
        numeric_function=numeric_function,
    )

    with pytest.raises(FitError, match=message):
        LmfitEngine().fit(_linear_dataset(), model)


def test_reports_solver_non_convergence(monkeypatch) -> None:
    def failed_minimize(*args, **kwargs):
        return SimpleNamespace(success=False, message="iteration limit")

    monkeypatch.setattr(
        "curvefit.fitting.lmfit_engine.lmfit.minimize",
        failed_minimize,
    )

    with pytest.raises(FitError, match="did not converge.*iteration limit"):
        LmfitEngine().fit(_linear_dataset(), create_linear_model())


@pytest.mark.parametrize("point_count", [1, 0, -1, 2.5, True])
def test_rejects_invalid_curve_point_count(point_count: object) -> None:
    with pytest.raises(ValueError, match="integer of at least 2"):
        LmfitEngine(curve_point_count=point_count)
