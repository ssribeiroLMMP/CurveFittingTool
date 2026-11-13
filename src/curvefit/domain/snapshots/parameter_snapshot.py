"""Immutable historical representation of a model parameter."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from numbers import Real

from curvefit.domain.parameter import Parameter
from curvefit.domain.parameter.validator import ParameterValidator


@dataclass(frozen=True, slots=True)
class ParameterSnapshot:
    """Preserve one parameter's configuration for a fitting execution."""

    symbol: str
    name: str | None = None
    initial_value: Real = 1.0
    min_value: Real = -inf
    max_value: Real = inf
    vary: bool = True
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the captured parameter configuration."""
        symbol = ParameterValidator.symbol(self.symbol)
        name = ParameterValidator.name(self.name)
        minimum, maximum = ParameterValidator.bounds(
            self.min_value,
            self.max_value,
        )
        initial_value = ParameterValidator.initial_value(
            self.initial_value,
            minimum,
            maximum,
        )
        vary = ParameterValidator.vary(self.vary)
        unit = ParameterValidator.unit(self.unit)

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "initial_value", initial_value)
        object.__setattr__(self, "min_value", minimum)
        object.__setattr__(self, "max_value", maximum)
        object.__setattr__(self, "vary", vary)
        object.__setattr__(self, "unit", unit)

    @classmethod
    def from_parameter(cls, parameter: Parameter) -> ParameterSnapshot:
        """Capture the current state of a Parameter without retaining it."""
        if not isinstance(parameter, Parameter):
            raise ValueError("parameter must be a Parameter object.")

        return cls(
            symbol=parameter.symbol,
            name=parameter.name,
            initial_value=parameter.initial_value,
            min_value=parameter.min_value,
            max_value=parameter.max_value,
            vary=parameter.vary,
            unit=parameter.unit,
        )
