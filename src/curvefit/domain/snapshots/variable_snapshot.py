"""Immutable historical representation of a model variable."""

from __future__ import annotations

from dataclasses import dataclass

from curvefit.domain.variable import Variable
from curvefit.domain.variable.validator import VariableValidator


@dataclass(frozen=True, slots=True)
class VariableSnapshot:
    """Preserve one variable's identity and metadata for an execution."""

    symbol: str
    name: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the captured variable state."""
        object.__setattr__(
            self,
            "symbol",
            VariableValidator.symbol(self.symbol),
        )
        object.__setattr__(
            self,
            "name",
            VariableValidator.name(self.name),
        )
        object.__setattr__(
            self,
            "unit",
            VariableValidator.unit(self.unit),
        )

    @classmethod
    def from_variable(cls, variable: Variable) -> VariableSnapshot:
        """Capture the current state of a Variable without retaining it."""
        if not isinstance(variable, Variable):
            raise ValueError("variable must be a Variable object.")

        return cls(
            symbol=variable.symbol,
            name=variable.name,
            unit=variable.unit,
        )
