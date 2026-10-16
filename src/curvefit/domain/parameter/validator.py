"""Validation rules specific to Parameter entities.

The validator composes reusable numeric, text and symbol validators and
centralizes all rules related to parameter identity and configuration.
"""

from __future__ import annotations

from numbers import Real

from curvefit.domain.validation import (
    NumericValidator,
    SymbolValidator,
    TextValidator,
)


class ParameterValidator:
    """Validate Parameter identity and configuration.

    This validator contains only rules that are meaningful in the context of
    curve-fitting parameters. Generic numeric rules are delegated to
    ``NumericValidator``.

    Atomic symbol syntax is delegated to ``SymbolValidator``. Full LaTeX
    expressions and model semantics remain responsibilities of the Model Engine.

    The class is stateless and is not intended to be instantiated.
    """

    @staticmethod
    def symbol(value: str) -> str:
        """Validate and normalize a parameter mathematical symbol.

        Parameters
        ----------
        value:
            Mathematical symbol associated with the parameter.

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
        """Validate and normalize an optional descriptive parameter name.

        Parameters
        ----------
        value:
            Human-readable description used by the GUI, plots and reports.

        Returns
        -------
        str | None
            Normalized descriptive name or ``None`` when unspecified.
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
            Human-readable physical unit associated with the parameter.

        Returns
        -------
        str | None
            Normalized unit or ``None`` when unspecified.
        """
        return TextValidator.optional(
            value,
            field_name="unit",
        )

    @staticmethod
    def bounds(
        minimum: Real,
        maximum: Real,
    ) -> tuple[float, float]:
        """Validate and normalize parameter bounds.

        Parameters
        ----------
        minimum:
            Lower admissible parameter bound. Negative infinity is allowed.
        maximum:
            Upper admissible parameter bound. Positive infinity is allowed.

        Returns
        -------
        tuple[float, float]
            Normalized bounds as ``(minimum, maximum)``.

        Raises
        ------
        ValueError
            If either bound is invalid or if ``minimum > maximum``.
        """
        normalized_minimum = NumericValidator.real(
            minimum,
            field_name="min_value",
        )

        normalized_maximum = NumericValidator.real(
            maximum,
            field_name="max_value",
        )

        return NumericValidator.less_than_or_equal(
            normalized_minimum,
            normalized_maximum,
            left_name="min_value",
            right_name="max_value",
        )

    @staticmethod
    def initial_value(
        value: Real,
        minimum: Real,
        maximum: Real,
    ) -> float:
        """Validate an optimizer initial value against parameter bounds.

        Parameters
        ----------
        value:
            Initial estimate supplied to the fitting engine.
        minimum:
            Lower admissible parameter bound.
        maximum:
            Upper admissible parameter bound.

        Returns
        -------
        float
            Validated finite initial value.

        Raises
        ------
        ValueError
            If the value is non-numeric, non-finite or outside the configured
            parameter bounds.
        """
        return NumericValidator.between(
            value,
            minimum,
            maximum,
            field_name="initial_value",
            finite_value=True,
        )

    @staticmethod
    def vary(value: bool) -> bool:
        """Validate whether a parameter is allowed to vary during fitting.

        Parameters
        ----------
        value:
            Flag indicating whether the fitting engine may change the
            parameter value.

        Returns
        -------
        bool
            Validated boolean flag.

        Raises
        ------
        ValueError
            If ``value`` is not a boolean.
        """
        if not isinstance(value, bool):
            raise ValueError("vary must be a boolean.")

        return value
