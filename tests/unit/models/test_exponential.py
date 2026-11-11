"""Tests for the built-in exponential model factory."""

import numpy as np
from sympy import exp, symbols

from curvefit.models.exponential import (
    EXPONENTIAL_MODEL_METADATA,
    create_exponential_model,
)


def test_exponential_factory_builds_expected_expression_and_parameters() -> None:
    """Create the offset exponential model with ordered parameters."""
    x, a, b, c = symbols("x a b c")

    model = create_exponential_model()

    assert model.name == "Exponential"
    assert model.symbolic_expressions == (a * exp(b * x) + c,)
    assert tuple(parameter.symbol for parameter in model.parameters) == (
        "a",
        "b",
        "c",
    )
    assert EXPONENTIAL_MODEL_METADATA.key == "exponential"


def test_exponential_numeric_function_evaluates_vector_inputs() -> None:
    """Return vectorized growth predictions in a one-column matrix."""
    model = create_exponential_model()

    predictions = model.numeric_function(
        np.array([[0.0], [1.0]]),
        np.array([2.0, 0.5, 1.0]),
    )

    np.testing.assert_allclose(
        predictions,
        [[3.0], [2.0 * np.exp(0.5) + 1.0]],
    )
    assert predictions.shape == (2, 1)


def test_exponential_factory_returns_fresh_instances() -> None:
    """Keep model parameter configurations independent between instances."""
    first = create_exponential_model()
    second = create_exponential_model()

    assert first.model_id != second.model_id
    assert first.parameters[0] is not second.parameters[0]
