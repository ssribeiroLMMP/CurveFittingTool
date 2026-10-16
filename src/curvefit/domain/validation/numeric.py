"""Reusable validation rules for numeric domain values.

The validators defined in this module are intentionally independent of
specific domain entities. They provide small, composable validation
operations that can be reused by parameter, dataset, result and other
domain-specific validators.
"""

from __future__ import annotations

from math import isfinite, isnan
from numbers import Real


class NumericValidator:
    """Provide reusable validation operations for numeric values.

    This class contains stateless validation methods intended to be composed
    by domain-specific validators.

    The methods return normalized ``float`` values whenever appropriate so
    that callers do not need to repeat numeric conversion logic.

    The class must remain independent of domain entities and third-party
    numerical libraries.
    """

    @staticmethod
    def real(
        value: Real,
        *,
        field_name: str = "value",
    ) -> float:
        """Validate that a value is a real number and normalize it to float.

        Parameters
        ----------
        value:
            Value to validate.
        field_name:
            Human-readable field name used in validation error messages.

        Returns
        -------
        float
            Validated value converted to ``float``.

        Raises
        ------
        ValueError
            If the value is not a real number, is boolean or is NaN.
        """
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{field_name} must be a real numeric value."
            )

        normalized = float(value)

        if isnan(normalized):
            raise ValueError(
                f"{field_name} cannot be NaN."
            )

        return normalized

    @classmethod
    def finite(
        cls,
        value: Real,
        *,
        field_name: str = "value",
    ) -> float:
        """Validate that a numeric value is finite.

        Parameters
        ----------
        value:
            Numeric value to validate.
        field_name:
            Human-readable field name used in validation error messages.

        Returns
        -------
        float
            Validated finite value.

        Raises
        ------
        ValueError
            If the value is not a valid real number or is infinite.
        """
        normalized = cls.real(
            value,
            field_name=field_name,
        )

        if not isfinite(normalized):
            raise ValueError(
                f"{field_name} must be finite."
            )

        return normalized

    @classmethod
    def non_negative(
        cls,
        value: Real,
        *,
        field_name: str = "value",
        finite_only: bool = True,
    ) -> float:
        """Validate that a numeric value is greater than or equal to zero.

        Parameters
        ----------
        value:
            Numeric value to validate.
        field_name:
            Human-readable field name used in validation error messages.
        finite_only:
            When ``True``, infinite values are rejected.

        Returns
        -------
        float
            Validated non-negative value.

        Raises
        ------
        ValueError
            If the value is negative or otherwise invalid.
        """
        validator = cls.finite if finite_only else cls.real

        normalized = validator(
            value,
            field_name=field_name,
        )

        if normalized < 0:
            raise ValueError(
                f"{field_name} cannot be negative."
            )

        return normalized

    @classmethod
    def between(
        cls,
        value: Real,
        minimum: Real,
        maximum: Real,
        *,
        field_name: str = "value",
        inclusive: bool = True,
        finite_value: bool = False,
    ) -> float:
        """Validate that a numeric value lies within a numeric interval.

        Parameters
        ----------
        value:
            Numeric value to validate.
        minimum:
            Lower interval boundary.
        maximum:
            Upper interval boundary.
        field_name:
            Human-readable field name used in validation error messages.
        inclusive:
            When ``True``, interval boundaries are accepted. When ``False``,
            the value must lie strictly inside the interval.
        finite_value:
            When ``True``, the validated value itself must be finite.

        Returns
        -------
        float
            Validated and normalized value.

        Raises
        ------
        ValueError
            If the interval is invalid or the value lies outside it.
        """
        normalized_minimum = cls.real(
            minimum,
            field_name="minimum",
        )

        normalized_maximum = cls.real(
            maximum,
            field_name="maximum",
        )

        cls.less_than_or_equal(
            normalized_minimum,
            normalized_maximum,
            left_name="minimum",
            right_name="maximum",
        )

        validator = cls.finite if finite_value else cls.real

        normalized_value = validator(
            value,
            field_name=field_name,
        )

        if inclusive:
            valid = (
                normalized_minimum
                <= normalized_value
                <= normalized_maximum
            )
        else:
            valid = (
                normalized_minimum
                < normalized_value
                < normalized_maximum
            )

        if not valid:
            interval_type = "between" if inclusive else "strictly between"

            raise ValueError(
                f"{field_name} must be {interval_type} "
                f"{normalized_minimum} and {normalized_maximum}."
            )

        return normalized_value

    @classmethod
    def less_than_or_equal(
        cls,
        left: Real,
        right: Real,
        *,
        left_name: str = "left",
        right_name: str = "right",
    ) -> tuple[float, float]:
        """Validate that one numeric value is less than or equal to another.

        Parameters
        ----------
        left:
            Value expected to be less than or equal to ``right``.
        right:
            Value expected to be greater than or equal to ``left``.
        left_name:
            Human-readable name of the left operand.
        right_name:
            Human-readable name of the right operand.

        Returns
        -------
        tuple[float, float]
            Normalized values in ``(left, right)`` order.

        Raises
        ------
        ValueError
            If either value is invalid or ``left > right``.
        """
        normalized_left = cls.real(
            left,
            field_name=left_name,
        )

        normalized_right = cls.real(
            right,
            field_name=right_name,
        )

        if normalized_left > normalized_right:
            raise ValueError(
                f"{left_name} cannot be greater than {right_name}."
            )

        return normalized_left, normalized_right
