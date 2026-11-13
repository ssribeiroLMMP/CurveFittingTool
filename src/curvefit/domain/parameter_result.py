"""Immutable numerical result for one fitted model parameter."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from curvefit.domain.parameter.validator import ParameterValidator
from curvefit.domain.validation import NumericValidator


@dataclass(frozen=True, slots=True)
class ParameterResult:
    """Store the fitted value and optional uncertainty of one parameter."""

    symbol: str
    value: Real
    stderr: Real | None = None

    def __post_init__(self) -> None:
        """Validate and normalize numerical result values."""
        symbol = ParameterValidator.symbol(self.symbol)
        value = NumericValidator.finite(self.value, field_name="value")
        stderr = (
            None
            if self.stderr is None
            else NumericValidator.non_negative(
                self.stderr,
                field_name="stderr",
            )
        )

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "stderr", stderr)
