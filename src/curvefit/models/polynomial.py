"""Built-in polynomial regression model factories."""

from __future__ import annotations

from sympy import Symbol, symbols

from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.parameter import Parameter
from curvefit.domain.variable import Variable
from curvefit.models._numerical import build_numeric_function
from curvefit.models.catalog import ModelMetadata


def create_polynomial_model(degree: int) -> ModelDefinition:
    """Create a fresh polynomial model definition for the requested degree.

    Parameters
    ----------
    degree:
        Polynomial degree. Values below two are reserved for lower-degree
        models, which have their own catalog entries.

    Raises
    ------
    ValueError
        If ``degree`` is not an integer greater than or equal to two.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 2:
        raise ValueError("degree must be an integer greater than or equal to 2.")

    x = symbols("x")
    parameter_symbols = tuple(
        Symbol(f"a_{power}")
        for power in range(degree + 1)
    )
    expression = sum(
        coefficient * x**power
        for power, coefficient in enumerate(parameter_symbols)
    )
    parameters = tuple(
        Parameter(
            str(symbol),
            name=f"Coefficient for x^{power}",
            initial_value=1.0 if power == degree else 0.0,
        )
        for power, symbol in enumerate(parameter_symbols)
    )
    metadata = polynomial_model_metadata(degree)

    return ModelDefinition(
        name=metadata.name,
        independent_variables=Variable("x", name="Independent variable"),
        dependent_variables=Variable("y", name="Dependent variable"),
        parameters=parameters,
        symbolic_expressions=expression,
        numeric_function=build_numeric_function(
            expression,
            x,
            parameter_symbols,
        ),
    )


def polynomial_model_metadata(degree: int) -> ModelMetadata:
    """Create catalog metadata for a supported polynomial degree."""
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 2:
        raise ValueError("degree must be an integer greater than or equal to 2.")

    names = {2: "Quadratic", 3: "Cubic"}
    name = names.get(degree, f"Polynomial (degree {degree})")
    return ModelMetadata(
        key=f"polynomial_{degree}",
        name=name,
        family="polynomial",
        description=(
            f"A polynomial model with degree {degree} and "
            f"{degree + 1} coefficients."
        ),
    )
