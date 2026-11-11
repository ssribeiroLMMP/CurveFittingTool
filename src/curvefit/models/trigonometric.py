"""Built-in trigonometric regression model factories."""

from __future__ import annotations

from collections.abc import Callable

from sympy import cos, sin, symbols
from sympy.core.expr import Expr

from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.parameter import Parameter
from curvefit.domain.variable import Variable
from curvefit.models._numerical import build_numeric_function
from curvefit.models.catalog import ModelMetadata

SINE_MODEL_METADATA = ModelMetadata(
    key="sine",
    name="Sine",
    family="trigonometric",
    description="A sinusoidal model with amplitude, frequency, phase, and offset.",
)

COSINE_MODEL_METADATA = ModelMetadata(
    key="cosine",
    name="Cosine",
    family="trigonometric",
    description="A cosinusoidal model with amplitude, frequency, phase, and offset.",
)


def create_sine_model() -> ModelDefinition:
    """Create a fresh sinusoidal model for ``y = a*sin(b*x + c) + d``."""
    return _create_trigonometric_model(
        name=SINE_MODEL_METADATA.name,
        function=sin,
    )


def create_cosine_model() -> ModelDefinition:
    """Create a fresh cosinusoidal model for ``y = a*cos(b*x + c) + d``."""
    return _create_trigonometric_model(
        name=COSINE_MODEL_METADATA.name,
        function=cos,
    )


def _create_trigonometric_model(
    *,
    name: str,
    function: Callable[[Expr], Expr],
) -> ModelDefinition:
    """Create a fresh trigonometric definition with common parameter semantics."""
    x, a, b, c, d = symbols("x a b c d")
    expression = a * function(b * x + c) + d
    parameters = (
        Parameter("a", name="Amplitude", initial_value=1.0),
        Parameter("b", name="Angular frequency", initial_value=1.0),
        Parameter("c", name="Phase", initial_value=0.0),
        Parameter("d", name="Offset", initial_value=0.0),
    )
    return ModelDefinition(
        name=name,
        independent_variables=Variable("x", name="Independent variable"),
        dependent_variables=Variable("y", name="Dependent variable"),
        parameters=parameters,
        symbolic_expressions=expression,
        numeric_function=build_numeric_function(expression, x, (a, b, c, d)),
    )
