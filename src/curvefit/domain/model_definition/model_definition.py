"""Domain entity representing a validated mathematical model definition.

A ModelDefinition stores the mathematical representation already produced by
the Model Engine. LaTeX is an input representation and is therefore not stored
by this domain entity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID, uuid4

from sympy import Expr

from curvefit.domain.model_definition.validator import ModelDefinitionValidator
from curvefit.domain.parameter import Parameter
from curvefit.domain.variable import Variable


class ModelDefinition:
    """Represent a validated mathematical model.

    Parameters
    ----------
    name:
        Human-readable model name.
    independent_variables:
        One Variable or an ordered sequence of independent variables.
    dependent_variables:
        One Variable or an ordered sequence of dependent variables.
    parameters:
        One Parameter, an ordered sequence of parameters or ``None`` for a
        model without adjustable parameters.
    symbolic_expressions:
        One SymPy expression or an ordered sequence containing one expression
        for each dependent variable.
    numeric_function:
        Callable implementing the numerical model.

    Notes
    -----
    Version 1 supports exactly one independent and one dependent variable.
    Collections are nevertheless stored as tuples so that future versions can
    support multiple model inputs and outputs without changing the structural
    representation.

    The numerical function conceptually evaluates ``Y_hat = F(X, parameters)``
    using the same matrix convention adopted by Dataset.

    LaTeX is not stored by ModelDefinition. It is parsed by the Model Engine
    before the ModelDefinition is constructed.
    """

    SUPPORTED_INDEPENDENT_VARIABLE_COUNT = 1
    SUPPORTED_DEPENDENT_VARIABLE_COUNT = 1

    def __init__(
        self,
        name: str,
        independent_variables: Variable | Sequence[Variable],
        dependent_variables: Variable | Sequence[Variable],
        parameters: Parameter | Sequence[Parameter] | None,
        symbolic_expressions: Expr | Sequence[Expr],
        numeric_function: Callable[..., Any],
        *,
        model_id: UUID | None = None,
    ) -> None:
        self._model_id = model_id or uuid4()

        validated_name = ModelDefinitionValidator.name(name)

        validated_independent_variables = ModelDefinitionValidator.variables(
            independent_variables,
            expected_count=self.SUPPORTED_INDEPENDENT_VARIABLE_COUNT,
        )

        validated_dependent_variables = ModelDefinitionValidator.variables(
            dependent_variables,
            expected_count=self.SUPPORTED_DEPENDENT_VARIABLE_COUNT,
        )

        validated_parameters = ModelDefinitionValidator.parameters(
            parameters,
        )

        validated_symbolic_expressions = (
            ModelDefinitionValidator.symbolic_expressions(
                symbolic_expressions,
            )
        )

        validated_numeric_function = ModelDefinitionValidator.numeric_function(
            numeric_function,
        )

        ModelDefinitionValidator.matching_expression_count(
            validated_dependent_variables,
            validated_symbolic_expressions,
        )

        ModelDefinitionValidator.distinct_symbols(
            validated_independent_variables,
            validated_dependent_variables,
            validated_parameters,
        )

        self._name = validated_name
        self._independent_variables = validated_independent_variables
        self._dependent_variables = validated_dependent_variables
        self._parameters = validated_parameters
        self._symbolic_expressions = validated_symbolic_expressions
        self._numeric_function = validated_numeric_function

    @property
    def model_id(self) -> UUID:
        """Return the unique immutable identifier of this model definition."""
        return self._model_id

    @property
    def name(self) -> str:
        """Return the human-readable model name."""
        return self._name

    @property
    def independent_variables(self) -> tuple[Variable, ...]:
        """Return the ordered independent variables."""
        return self._independent_variables

    @property
    def dependent_variables(self) -> tuple[Variable, ...]:
        """Return the ordered dependent variables."""
        return self._dependent_variables

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Return the ordered model parameters."""
        return self._parameters

    @property
    def symbolic_expressions(self) -> tuple[Expr, ...]:
        """Return the ordered symbolic expressions of the model outputs."""
        return self._symbolic_expressions

    @property
    def numeric_function(self) -> Callable[..., Any]:
        """Return the numerical function used to evaluate the model."""
        return self._numeric_function

    def copy(self) -> "ModelDefinition":
        """Return an independent copy of the model definition.

        Returns
        -------
        ModelDefinition
            New model definition with copied Variable and Parameter objects.

        Notes
        -----
        Symbolic expressions and the numerical function are reused because
        they represent the mathematical structure of the model. Variable and
        Parameter objects are copied so that changes to their configurable
        state in the copied model do not affect the original model.
        """
        return ModelDefinition(
            name=self._name,
            independent_variables=tuple(
                variable.copy()
                for variable in self._independent_variables
            ),
            dependent_variables=tuple(
                variable.copy()
                for variable in self._dependent_variables
            ),
            parameters=tuple(
                parameter.copy()
                for parameter in self._parameters
            ),
            symbolic_expressions=self._symbolic_expressions,
            numeric_function=self._numeric_function,
        )

    def __repr__(self) -> str:
        """Return a concise developer representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"independent_variables={len(self._independent_variables)}, "
            f"dependent_variables={len(self._dependent_variables)}, "
            f"parameters={len(self._parameters)})"
        )
