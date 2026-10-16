"""Reusable validation rules for textual domain values.

The validators defined in this module are independent of specific domain
entities. They provide small normalization and validation operations for
required and optional textual properties.
"""

from __future__ import annotations


class TextValidator:
    """Provide reusable validation operations for textual values.

    The class contains stateless operations intended to be composed by
    domain-specific validators.
    """

    @staticmethod
    def required(
        value: str,
        *,
        field_name: str = "value",
    ) -> str:
        """Validate and normalize a required non-empty string.

        Parameters
        ----------
        value:
            Text value to validate.
        field_name:
            Human-readable field name used in validation error messages.

        Returns
        -------
        str
            Validated text with surrounding whitespace removed.

        Raises
        ------
        ValueError
            If the value is not a string or becomes empty after trimming.
        """
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string.")

        normalized = value.strip()

        if not normalized:
            raise ValueError(f"{field_name} cannot be empty.")

        return normalized

    @staticmethod
    def optional(
        value: str | None,
        *,
        field_name: str = "value",
    ) -> str | None:
        """Validate and normalize an optional textual value.

        Parameters
        ----------
        value:
            Text value or ``None``.
        field_name:
            Human-readable field name used in validation error messages.

        Returns
        -------
        str | None
            Normalized text, or ``None`` when no meaningful text is supplied.

        Raises
        ------
        ValueError
            If the supplied value is neither a string nor ``None``.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string or None.")

        normalized = value.strip()

        return normalized or None
