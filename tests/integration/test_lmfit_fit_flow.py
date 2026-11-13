"""Integration tests for synthetic model fitting through LmfitEngine."""

import numpy as np
import pytest

from curvefit.domain.data_series import DataSeries
from curvefit.domain.dataset import Dataset
from curvefit.domain.variable import Variable
from curvefit.fitting import LmfitEngine
from curvefit.models.exponential import create_exponential_model
from curvefit.models.linear import create_linear_model


def _dataset(
    independent: np.ndarray,
    dependent: np.ndarray,
) -> Dataset:
    return Dataset(
        DataSeries(
            independent_variables=Variable("x"),
            dependent_variables=Variable("y"),
            independent_data=independent,
            dependent_data=dependent,
        )
    )


def test_recovers_known_linear_parameters() -> None:
    independent = np.linspace(-5.0, 5.0, 25).reshape(-1, 1)
    dataset = _dataset(independent, 2.5 * independent - 0.75)

    result = LmfitEngine().fit(dataset, create_linear_model())

    assert result.parameter_results[0].value == pytest.approx(2.5, abs=1e-10)
    assert result.parameter_results[1].value == pytest.approx(-0.75, abs=1e-10)
    assert result.metrics["rmse"] == pytest.approx(0.0, abs=1e-10)


def test_recovers_known_exponential_parameters() -> None:
    independent = np.linspace(0.0, 4.0, 40).reshape(-1, 1)
    expected = 2.0 * np.exp(-0.7 * independent) + 0.5
    model = create_exponential_model()
    model.parameters[0].configure(
        initial_value=1.5,
        min_value=0.0,
        max_value=5.0,
    )
    model.parameters[1].configure(
        initial_value=-0.5,
        min_value=-2.0,
        max_value=0.0,
    )
    model.parameters[2].configure(
        initial_value=0.25,
        min_value=-1.0,
        max_value=2.0,
    )

    result = LmfitEngine().fit(_dataset(independent, expected), model)

    assert result.parameter_results[0].value == pytest.approx(2.0, rel=1e-6)
    assert result.parameter_results[1].value == pytest.approx(-0.7, rel=1e-6)
    assert result.parameter_results[2].value == pytest.approx(0.5, rel=1e-6)
    assert result.metrics["r_squared"] == pytest.approx(1.0, abs=1e-10)
