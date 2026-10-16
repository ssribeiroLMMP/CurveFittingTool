"""Validation rules specific to mathematical variables.

This module defines the validation facade used by ``Variable``. Generic
textual and mathematical-symbol validation is delegated to reusable domain
validators so that the Variable entity remains independent of validation
implementation details.
"""

from __future__ import annotations

from curvefit.domain.validation import SymbolValidator, TextValidator


class VariableValidator:
    """Validate values associated with Variable entities.

    This class composes reusable domain validators and exposes only the
    validation operations required by ``Variable``.

    Full mathematical-expression parsing is intentionally outside this
    validator and remains a responsibility of the Model Engine.
    """

    @staticmethod
    def symbol(value: str) -> str:
        """Validate and normalize a variable mathematical symbol.

        Parameters
        ----------
        value:
            Mathematical symbol identifying the variable.

        Returns
        -------
        str
            Validated atomic mathematical symbol.

        Raises
        ------
        ValueError
            If the value is not a supported atomic mathematical symbol.
        """
        return SymbolValidator.validate(
            value,
            field_name="symbol",
        )

    @staticmethod
    def name(value: str | None) -> str | None:
        """Validate and normalize an optional descriptive variable name.

        Parameters
        ----------
        value:
            Optional human-readable description used by interfaces, plots and
            reports. Empty or whitespace-only strings are normalized to
            ``None``.

        Returns
        -------
        str | None
            Normalized descriptive name, or ``None`` when the value is absent
            or empty.

        Raises
        ------
        ValueError
            If the supplied value is neither a string nor ``None``.
        """
        return TextValidator.optional(
            value,
            field_name="name",
        )

    @staticmethod
    def unit(value: str | None) -> str | None:
        """Validate and normalize an optional physical unit.

        Parameters
        ----------
        value:
            Optional human-readable physical unit. Empty or whitespace-only
            strings are normalized to ``None``.

        Returns
        -------
        str | None
            Normalized unit, or ``None`` when the value is absent or empty.

        Raises
        ------
        ValueError
            If the supplied value is neither a string nor ``None``.
        """
        return TextValidator.optional(
            value,
            field_name="unit",
        )
