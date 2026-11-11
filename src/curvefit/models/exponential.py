"""Built-in exponential regression model."""

from __future__ import annotations

from sympy import exp, symbols

from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.parameter import Parameter
from curvefit.domain.variable import Variable
from curvefit.models._numerical import build_numeric_function
from curvefit.models.catalog import ModelMetadata

EXPONENTIAL_MODEL_METADATA = ModelMetadata(
    key="exponential",
    name="Exponential",
    family="exponential",
    description="Exponential growth or decay with an offset.",
)


def create_exponential_model() -> ModelDefinition:
    """Create a fresh exponential model definition for ``y = a*exp(b*x) + c``."""
    x, a, b, c = symbols("x a b c")
    expression = a * exp(b * x) + c
    parameters = (
        Parameter("a", name="Amplitude", initial_value=1.0),
        Parameter("b", name="Growth rate", initial_value=1.0),
        Parameter("c", name="Offset", initial_value=0.0),
    )
    return ModelDefinition(
        name=EXPONENTIAL_MODEL_METADATA.name,
        independent_variables=Variable("x", name="Independent variable"),
        dependent_variables=Variable("y", name="Dependent variable"),
        parameters=parameters,
        symbolic_expressions=expression,
        numeric_function=build_numeric_function(expression, x, (a, b, c)),
    )
