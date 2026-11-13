"""Built-in linear regression model."""

from __future__ import annotations

from sympy import symbols

from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.parameter import Parameter
from curvefit.domain.variable import Variable
from curvefit.models._numerical import build_numeric_function
from curvefit.models.catalog import ModelMetadata

LINEAR_MODEL_METADATA = ModelMetadata(
    key="linear",
    name="Linear",
    family="linear",
    description="A straight line with slope and intercept.",
)


def create_linear_model() -> ModelDefinition:
    """Create a fresh linear model definition for ``y = a*x + b``."""
    x, a, b = symbols("x a b")
    expression = a * x + b
    parameters = (
        Parameter("a", name="Slope", initial_value=1.0),
        Parameter("b", name="Intercept", initial_value=0.0),
    )
    return ModelDefinition(
        name=LINEAR_MODEL_METADATA.name,
        independent_variables=Variable("x", name="Independent variable"),
        dependent_variables=Variable("y", name="Dependent variable"),
        parameters=parameters,
        symbolic_expressions=expression,
        numeric_function=build_numeric_function(expression, x, (a, b)),
    )
