"""Tests for built-in polynomial model factories."""

import numpy as np
import pytest
from sympy import symbols

from curvefit.models.polynomial import (
    create_polynomial_model,
    polynomial_model_metadata,
)


@pytest.mark.parametrize(
    ("degree", "name", "expected"),
    [
        (2, "Quadratic", [[3.0], [10.0]]),
        (3, "Cubic", [[6.0], [34.0]]),
    ],
)
def test_polynomial_factory_creates_evaluable_model(
    degree: int,
    name: str,
    expected: list[list[float]],
) -> None:
    """Create quadratic and cubic models with ordered coefficient parameters."""
    model = create_polynomial_model(degree)
    independent_data = np.array([[1.0], [2.0]])
    parameter_values = np.arange(degree + 1, dtype=np.float64)

    predictions = model.numeric_function(independent_data, parameter_values)

    assert model.name == name
    assert len(model.parameters) == degree + 1
    assert tuple(parameter.symbol for parameter in model.parameters) == tuple(
        f"a_{power}"
        for power in range(degree + 1)
    )
    np.testing.assert_allclose(predictions, expected)


@pytest.mark.parametrize("degree", [0, 1, -2, 2.5, True])
def test_polynomial_factory_rejects_unsupported_degree(degree: object) -> None:
    """Require a non-boolean integer degree of at least two."""
    with pytest.raises(ValueError, match="degree must be an integer"):
        create_polynomial_model(degree)


def test_polynomial_metadata_uses_stable_degree_key() -> None:
    """Derive a stable discovery key from the polynomial degree."""
    metadata = polynomial_model_metadata(2)

    assert metadata.key == "polynomial_2"
    assert metadata.name == "Quadratic"
    assert metadata.family == "polynomial"


def test_polynomial_symbolic_expression_matches_coefficients() -> None:
    """Build the symbolic polynomial in ascending coefficient order."""
    x = symbols("x")

    model = create_polynomial_model(2)

    assert model.symbolic_expressions == (x**2 * symbols("a_2") + x * symbols("a_1") + symbols("a_0"),)
