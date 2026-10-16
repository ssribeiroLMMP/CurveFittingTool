"""Validation rules for atomic mathematical symbols.

A mathematical symbol is distinct from a complete mathematical expression.
This module accepts a deliberately restricted notation that can be shared by
``Parameter`` and ``Variable`` while leaving full LaTeX parsing to the Model
Engine.
"""

from __future__ import annotations

import re

from curvefit.domain.validation.text import TextValidator


class SymbolValidator:
    """Validate atomic mathematical symbols used by domain entities.

    Supported symbols consist of a plain identifier or a recognized LaTeX
    Greek-letter command, optionally followed by one simple subscript.
    Mathematical expressions, functions and operators are intentionally
    rejected.
    """

    _LATEX_GREEK_COMMANDS = (
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "varepsilon",
        "zeta",
        "eta",
        "theta",
        "vartheta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "pi",
        "varpi",
        "rho",
        "varrho",
        "sigma",
        "varsigma",
        "tau",
        "upsilon",
        "phi",
        "varphi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Pi",
        "Sigma",
        "Upsilon",
        "Phi",
        "Psi",
        "Omega",
    )

    _PLAIN_IDENTIFIER = r"[^\W\d_][^\W_]*"
    _LATEX_COMMAND = rf"\\(?:{'|'.join(_LATEX_GREEK_COMMANDS)})"
    _BASE = rf"(?:{_PLAIN_IDENTIFIER}|{_LATEX_COMMAND})"
    _SUBSCRIPT_TOKEN = rf"(?:{_PLAIN_IDENTIFIER}|\d+|{_LATEX_COMMAND})"
    _PATTERN = re.compile(
        rf"^{_BASE}(?:_(?:{_SUBSCRIPT_TOKEN}|\{{{_SUBSCRIPT_TOKEN}\}}))?$"
    )

    @classmethod
    def validate(
        cls,
        value: str,
        *,
        field_name: str = "symbol",
    ) -> str:
        """Validate and normalize an atomic mathematical symbol.

        Parameters
        ----------
        value:
            Mathematical symbol to validate.
        field_name:
            Human-readable field name used in validation error messages.

        Returns
        -------
        str
            Normalized mathematical symbol.

        Raises
        ------
        ValueError
            If the value is not text, is blank or represents unsupported
            symbol syntax rather than one atomic mathematical symbol.
        """
        normalized = TextValidator.required(
            value,
            field_name=field_name,
        )

        if cls._PATTERN.fullmatch(normalized) is None:
            raise ValueError(
                f"{field_name} must be a valid atomic mathematical symbol."
            )

        return normalized
