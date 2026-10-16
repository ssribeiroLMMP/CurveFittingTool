"""Domain entity representing a curve-fitting model parameter.

A Parameter stores the mathematical identity, descriptive metadata and
optimization configuration associated with one adjustable quantity of a
mathematical model.

Validation is delegated to ParameterValidator so that this entity remains
focused on state and domain behavior.
"""

from __future__ import annotations

from math import inf, isfinite
from numbers import Real

from curvefit.domain.parameter.validator import ParameterValidator


class Parameter:
    """Represent an adjustable parameter of a mathematical fitting model.

    Parameters
    ----------
    symbol:
        Mathematical identity of the parameter. It may contain LaTeX-oriented
        notation and remains immutable throughout the lifetime of the object.
    name:
        Optional human-readable description used by interfaces, reports and
        plots.
    initial_value:
        Finite starting value supplied to the fitting engine.
    min_value:
        Lower admissible bound. Negative infinity is allowed.
    max_value:
        Upper admissible bound. Positive infinity is allowed.
    vary:
        Indicates whether the fitting engine may modify the parameter.
    unit:
        Optional human-readable physical unit.

    Notes
    -----
    This entity does not perform fitting and does not depend on numerical
    optimization libraries or presentation frameworks.

    The mathematical ``symbol`` defines the identity of the object and cannot
    be modified after construction.
    """

    DEFAULT_NAME = None
    DEFAULT_INITIAL_VALUE = 1.0
    DEFAULT_MIN_VALUE = -inf
    DEFAULT_MAX_VALUE = inf
    DEFAULT_VARY = True
    DEFAULT_UNIT = None

    def __init__(
        self,
        symbol: str,
        name: str | None = DEFAULT_NAME,
        initial_value: Real = DEFAULT_INITIAL_VALUE,
        min_value: Real = DEFAULT_MIN_VALUE,
        max_value: Real = DEFAULT_MAX_VALUE,
        vary: bool = DEFAULT_VARY,
        unit: str | None = DEFAULT_UNIT,
    ) -> None:
        self._symbol = ParameterValidator.symbol(symbol)

        self._apply_configuration(
            name=name,
            initial_value=initial_value,
            min_value=min_value,
            max_value=max_value,
            vary=vary,
            unit=unit,
        )

    @property
    def symbol(self) -> str:
        """Return the immutable mathematical symbol of the parameter."""
        return self._symbol

    @property
    def name(self) -> str | None:
        """Return the optional human-readable parameter name."""
        return self._name

    @property
    def initial_value(self) -> float:
        """Return the starting value supplied to the fitting engine."""
        return self._initial_value

    @property
    def min_value(self) -> float:
        """Return the configured lower parameter bound."""
        return self._min_value

    @property
    def max_value(self) -> float:
        """Return the configured upper parameter bound."""
        return self._max_value

    @property
    def bounds(self) -> tuple[float, float]:
        """Return parameter bounds as ``(minimum, maximum)``."""
        return self._min_value, self._max_value

    @property
    def has_bounds(self) -> bool:
        """Return whether at least one finite bound constrains the parameter."""
        return isfinite(self._min_value) or isfinite(self._max_value)

    @property
    def vary(self) -> bool:
        """Return whether the fitting engine may vary this parameter."""
        return self._vary

    @property
    def unit(self) -> str | None:
        """Return the optional physical unit."""
        return self._unit

    def configure(
        self,
        *,
        name: str | None = DEFAULT_NAME,
        initial_value: Real = DEFAULT_INITIAL_VALUE,
        min_value: Real = DEFAULT_MIN_VALUE,
        max_value: Real = DEFAULT_MAX_VALUE,
        vary: bool = DEFAULT_VARY,
        unit: str | None = DEFAULT_UNIT,
    ) -> None:
        """Replace the complete parameter configuration.

        Parameters
        ----------
        name:
            Optional human-readable parameter description.
        initial_value:
            Finite starting value supplied to the fitting engine.
        min_value:
            Lower admissible bound.
        max_value:
            Upper admissible bound.
        vary:
            Indicates whether the fitting engine may modify the parameter.
        unit:
            Optional physical unit.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any supplied value violates the parameter domain rules.

        Notes
        -----
        ``configure`` replaces the complete configurable state rather than
        applying a partial patch.

        Calling ``configure()`` without arguments restores the default
        configuration while preserving the immutable mathematical ``symbol``.

        All proposed values are validated before the current state is modified,
        so failed validation leaves the object unchanged.
        """
        self._apply_configuration(
            name=name,
            initial_value=initial_value,
            min_value=min_value,
            max_value=max_value,
            vary=vary,
            unit=unit,
        )

    def copy(self) -> Parameter:
        """Create an independent copy of the parameter.

        Returns
        -------
        Parameter
            New instance containing the same identity and configuration.
        """
        return Parameter(
            symbol=self._symbol,
            name=self._name,
            initial_value=self._initial_value,
            min_value=self._min_value,
            max_value=self._max_value,
            vary=self._vary,
            unit=self._unit,
        )

    def _apply_configuration(
        self,
        *,
        name: str | None,
        initial_value: Real,
        min_value: Real,
        max_value: Real,
        vary: bool,
        unit: str | None,
    ) -> None:
        """Validate and atomically apply a complete parameter configuration.

        Parameters
        ----------
        name:
            Optional descriptive parameter name.
        initial_value:
            Proposed fitting initial value.
        min_value:
            Proposed lower bound.
        max_value:
            Proposed upper bound.
        vary:
            Proposed optimization variation flag.
        unit:
            Optional physical unit.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any proposed property violates the parameter domain rules.

        Notes
        -----
        This private operation centralizes configuration logic used both by
        construction and reconfiguration, avoiding duplicated validation while
        ensuring that the object is only mutated after every value has been
        successfully validated.
        """
        validated_name = ParameterValidator.name(name)

        validated_min, validated_max = ParameterValidator.bounds(
            min_value,
            max_value,
        )

        validated_initial = ParameterValidator.initial_value(
            initial_value,
            validated_min,
            validated_max,
        )

        validated_vary = ParameterValidator.vary(vary)

        validated_unit = ParameterValidator.unit(unit)

        self._name = validated_name
        self._initial_value = validated_initial
        self._min_value = validated_min
        self._max_value = validated_max
        self._vary = validated_vary
        self._unit = validated_unit

    def __repr__(self) -> str:
        """Return a developer-oriented representation of the parameter."""
        return (
            f"{self.__class__.__name__}("
            f"symbol={self._symbol!r}, "
            f"name={self._name!r}, "
            f"initial_value={self._initial_value!r}, "
            f"min_value={self._min_value!r}, "
            f"max_value={self._max_value!r}, "
            f"vary={self._vary!r}, "
            f"unit={self._unit!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare parameters by identity and configuration."""
        if not isinstance(other, Parameter):
            return NotImplemented

        return (
            self._symbol == other._symbol
            and self._name == other._name
            and self._initial_value == other._initial_value
            and self._min_value == other._min_value
            and self._max_value == other._max_value
            and self._vary == other._vary
            and self._unit == other._unit
        )
