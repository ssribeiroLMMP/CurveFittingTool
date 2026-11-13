"""Immutable historical representation of a mathematical model."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sympy import srepr

from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.model_definition.validator import ModelDefinitionValidator
from curvefit.domain.snapshots.parameter_snapshot import ParameterSnapshot
from curvefit.domain.snapshots.variable_snapshot import VariableSnapshot
from curvefit.domain.validation import TextValidator


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    """Preserve the complete model configuration used by one execution."""

    model_id: UUID
    name: str
    independent_variables: tuple[VariableSnapshot, ...]
    dependent_variables: tuple[VariableSnapshot, ...]
    parameters: tuple[ParameterSnapshot, ...]
    symbolic_expressions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and normalize the historical model representation."""
        if not isinstance(self.model_id, UUID):
            raise ValueError("model_id must be a UUID.")

        name = ModelDefinitionValidator.name(self.name)
        independent_variables = self._validated_items(
            self.independent_variables,
            VariableSnapshot,
            "independent_variables",
            allow_empty=False,
        )
        dependent_variables = self._validated_items(
            self.dependent_variables,
            VariableSnapshot,
            "dependent_variables",
            allow_empty=False,
        )
        parameters = self._validated_items(
            self.parameters,
            ParameterSnapshot,
            "parameters",
            allow_empty=True,
        )
        symbolic_expressions = self._validated_expressions(
            self.symbolic_expressions,
        )

        if len(symbolic_expressions) != len(dependent_variables):
            raise ValueError(
                "symbolic_expressions must contain one expression for each "
                "dependent variable."
            )

        symbols = tuple(
            item.symbol
            for item in (
                *independent_variables,
                *dependent_variables,
                *parameters,
            )
        )
        if len(symbols) != len(set(symbols)):
            raise ValueError(
                "variable and parameter symbols must be distinct within "
                "a model snapshot."
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "independent_variables",
            independent_variables,
        )
        object.__setattr__(
            self,
            "dependent_variables",
            dependent_variables,
        )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "symbolic_expressions",
            symbolic_expressions,
        )

    @classmethod
    def from_model(cls, model: ModelDefinition) -> ModelSnapshot:
        """Capture a ModelDefinition without retaining mutable references."""
        if not isinstance(model, ModelDefinition):
            raise ValueError("model must be a ModelDefinition object.")

        return cls(
            model_id=model.model_id,
            name=model.name,
            independent_variables=tuple(
                VariableSnapshot.from_variable(variable)
                for variable in model.independent_variables
            ),
            dependent_variables=tuple(
                VariableSnapshot.from_variable(variable)
                for variable in model.dependent_variables
            ),
            parameters=tuple(
                ParameterSnapshot.from_parameter(parameter)
                for parameter in model.parameters
            ),
            symbolic_expressions=tuple(
                srepr(expression)
                for expression in model.symbolic_expressions
            ),
        )

    @staticmethod
    def _validated_items(
        values: object,
        expected_type: type,
        field_name: str,
        *,
        allow_empty: bool,
    ) -> tuple:
        """Return a tuple containing only the expected snapshot type."""
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{field_name} must be a tuple.")

        try:
            normalized = tuple(values)
        except TypeError:
            raise ValueError(f"{field_name} must be a tuple.") from None

        if not allow_empty and not normalized:
            raise ValueError(f"{field_name} cannot be empty.")
        if not all(isinstance(value, expected_type) for value in normalized):
            raise ValueError(
                f"{field_name} must contain only {expected_type.__name__} objects."
            )

        return normalized

    @staticmethod
    def _validated_expressions(values: object) -> tuple[str, ...]:
        """Validate canonical symbolic expression strings."""
        if isinstance(values, (str, bytes)):
            values = (values,)

        try:
            normalized = tuple(values)
        except TypeError:
            raise ValueError(
                "symbolic_expressions must be a tuple of strings."
            ) from None

        if not normalized:
            raise ValueError("symbolic_expressions cannot be empty.")

        return tuple(
            TextValidator.required(
                value,
                field_name="symbolic expression",
            )
            for value in normalized
        )
