"""Tests for the built-in sine and cosine model factories."""

from collections.abc import Callable

import numpy as np
import pytest
from sympy import Expr, cos, sin, symbols

from curvefit.domain.model_definition import ModelDefinition
from curvefit.models.catalog import ModelMetadata
from curvefit.models.trigonometric import (
    COSINE_MODEL_METADATA,
    SINE_MODEL_METADATA,
    create_cosine_model,
    create_sine_model,
)


@pytest.mark.parametrize(
    ("factory", "metadata", "function"),
    [
        (create_sine_model, SINE_MODEL_METADATA, sin),
        (create_cosine_model, COSINE_MODEL_METADATA, cos),
    ],
)
def test_trigonometric_factory_builds_expected_model(
    factory: Callable[[], ModelDefinition],
    metadata: ModelMetadata,
    function: Callable[[Expr], Expr],
) -> None:
    """Create sine and cosine definitions with common parameter semantics."""
    x, a, b, c, d = symbols("x a b c d")

    model = factory()

    assert model.name == metadata.name
    assert model.symbolic_expressions == (a * function(b * x + c) + d,)
    assert tuple(parameter.symbol for parameter in model.parameters) == (
        "a",
        "b",
        "c",
        "d",
    )
    assert metadata.family == "trigonometric"


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (create_sine_model, [[1.0], [3.0]]),
        (create_cosine_model, [[3.0], [1.0]]),
    ],
)
def test_trigonometric_numeric_function_evaluates_vectors(
    factory: Callable[[], ModelDefinition],
    expected: list[list[float]],
) -> None:
    """Evaluate vectorized sinusoidal predictions with amplitude and offset."""
    model = factory()

    predictions = model.numeric_function(
        np.array([[0.0], [np.pi / 2.0]]),
        np.array([2.0, 1.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(predictions, expected, atol=1e-12)
    assert predictions.shape == (2, 1)
