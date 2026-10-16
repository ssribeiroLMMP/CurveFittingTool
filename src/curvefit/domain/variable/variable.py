"""Domain entity representing a mathematical variable.

A Variable stores the mathematical identity and descriptive metadata of an
independent or dependent quantity used by a dataset and mathematical model.

Validation is delegated to VariableValidator so that this entity remains
focused on state and domain behavior.
"""

from __future__ import annotations

from curvefit.domain.variable.validator import VariableValidator


class Variable:
    """Represent a mathematical variable used by datasets and models.

    Parameters
    ----------
    symbol:
        Mathematical identity of the variable. The symbol is immutable after
        object construction.
    name:
        Optional human-readable description used by interfaces, plots and
        reports. Empty or whitespace-only strings are normalized to ``None``.
    unit:
        Optional human-readable physical unit. Empty or whitespace-only
        strings are normalized to ``None``.

    Notes
    -----
    ``Variable`` stores metadata for one mathematical variable. It does not
    contain experimental data values and does not determine whether the
    variable is independent or dependent. That role is established by the
    object that references it, such as Dataset or ModelDefinition.

    The mathematical ``symbol`` defines the identity of the object and cannot
    be modified after construction.
    """

    DEFAULT_NAME = None
    DEFAULT_UNIT = None

    def __init__(
        self,
        symbol: str,
        name: str | None = DEFAULT_NAME,
        unit: str | None = DEFAULT_UNIT,
    ) -> None:
        self._symbol = VariableValidator.symbol(symbol)

        self._apply_configuration(
            name=name,
            unit=unit,
        )

    @property
    def symbol(self) -> str:
        """Return the immutable mathematical symbol of the variable."""
        return self._symbol

    @property
    def name(self) -> str | None:
        """Return the optional human-readable variable name."""
        return self._name

    @property
    def unit(self) -> str | None:
        """Return the optional physical unit associated with the variable."""
        return self._unit

    def configure(
        self,
        *,
        name: str | None = DEFAULT_NAME,
        unit: str | None = DEFAULT_UNIT,
    ) -> None:
        """Replace the complete configurable metadata of the variable.

        Parameters
        ----------
        name:
            Optional human-readable variable description. Empty or
            whitespace-only strings are normalized to ``None``.
        unit:
            Optional physical unit. Empty or whitespace-only strings are
            normalized to ``None``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``name`` or ``unit`` is neither a string nor ``None``.

        Notes
        -----
        Calling ``configure()`` without arguments restores all configurable
        properties to their default values while preserving ``symbol``.

        All proposed values are validated before the current state is
        modified. A failed configuration therefore leaves the object
        unchanged.
        """
        self._apply_configuration(
            name=name,
            unit=unit,
        )

    def copy(self) -> Variable:
        """Create an independent copy of the variable.

        Returns
        -------
        Variable
            New instance containing the same mathematical identity and
            metadata.
        """
        return Variable(
            symbol=self._symbol,
            name=self._name,
            unit=self._unit,
        )

    def _apply_configuration(
        self,
        *,
        name: str | None,
        unit: str | None,
    ) -> None:
        """Validate and atomically apply variable metadata.

        Parameters
        ----------
        name:
            Proposed optional human-readable variable description. Empty or
            whitespace-only strings are normalized to ``None``.
        unit:
            Proposed optional physical unit. Empty or whitespace-only strings
            are normalized to ``None``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If either value is neither a string nor ``None``.

        Notes
        -----
        Both values are validated before either internal attribute is
        modified.
        """
        validated_name = VariableValidator.name(name)
        validated_unit = VariableValidator.unit(unit)

        self._name = validated_name
        self._unit = validated_unit

    def __repr__(self) -> str:
        """Return a developer-oriented representation of the variable."""
        return (
            f"{self.__class__.__name__}("
            f"symbol={self._symbol!r}, "
            f"name={self._name!r}, "
            f"unit={self._unit!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare variables by mathematical identity and metadata."""
        if not isinstance(other, Variable):
            return NotImplemented

        return (
            self._symbol == other._symbol
            and self._name == other._name
            and self._unit == other._unit
        )
