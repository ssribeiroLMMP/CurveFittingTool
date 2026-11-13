"""Tests for the built-in linear model factory."""

import numpy as np
from sympy import symbols

from curvefit.models.linear import (
    LINEAR_MODEL_METADATA,
    create_linear_model,
)


class TestLinearModel:
    """Check the linear model's structure and numerical behavior."""

    def test_creates_expected_definition(self) -> None:
        """Create a named model with the expected equation and parameters."""
        x, a, b = symbols("x a b")

        model = create_linear_model()

        assert model.name == "Linear"
        assert model.independent_variables[0].symbol == "x"
        assert model.dependent_variables[0].symbol == "y"
        assert tuple(parameter.symbol for parameter in model.parameters) == (
            "a",
            "b",
        )
        assert model.symbolic_expressions == (a * x + b,)
        assert LINEAR_MODEL_METADATA.key == "linear"
        assert LINEAR_MODEL_METADATA.family == "linear"

    def test_numeric_function_returns_column_predictions(self) -> None:
        """Evaluate the line for a matrix of inputs and ordered parameters."""
        model = create_linear_model()

        predictions = model.numeric_function(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([3.0, 2.0]),
        )

        np.testing.assert_allclose(predictions, [[2.0], [5.0], [8.0]])
        assert predictions.shape == (3, 1)

    def test_rejects_input_with_the_wrong_shape(self) -> None:
        """Require model inputs to use the canonical one-column matrix shape."""
        model = create_linear_model()

        with np.testing.assert_raises_regex(ValueError, "shape"):
            model.numeric_function(np.array([1.0, 2.0]), np.array([1.0, 0.0]))

    def test_rejects_a_parameter_vector_with_the_wrong_size(self) -> None:
        """Require one numerical value for each ordered model parameter."""
        model = create_linear_model()

        with np.testing.assert_raises_regex(ValueError, "one value"):
            model.numeric_function(np.array([[1.0]]), np.array([1.0]))
