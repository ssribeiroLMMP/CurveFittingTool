"""Validation rules specific to ModelDefinition entities.

This module validates the structural components of a mathematical model
definition, including variables, parameters, symbolic expressions and the
numerical function used to evaluate the model.

Parsing LaTeX, generating SymPy expressions and building numerical functions
remain responsibilities of the Model Engine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sympy import Expr

from curvefit.domain.parameter import Parameter
from curvefit.domain.validation import TextValidator
from curvefit.domain.variable import Variable


class ModelDefinitionValidator:
    """Validate structural properties of a mathematical model definition.

    The validator operates on objects that have already been created by the
    appropriate domain or Model Engine components.

    It does not parse LaTeX, inspect the mathematical form of an equation or
    determine whether a model is linear or nonlinear.
    """

    @staticmethod
    def name(value: str) -> str:
        """Validate and normalize the required model name.

        Parameters
        ----------
        value:
            Human-readable model name.

        Returns
        -------
        str
            Normalized non-empty model name.

        Raises
        ------
        ValueError
            If ``value`` is not a string or becomes empty after trimming.
        """
        return TextValidator.required(
            value,
            field_name="name",
        )

    @staticmethod
    def variables(
        values: Variable | Sequence[Variable],
        *,
        expected_count: int | None = None,
    ) -> tuple[Variable, ...]:
        """Validate and normalize one or more mathematical variables.

        Parameters
        ----------
        values:
            One Variable or an ordered sequence of Variable objects.
        expected_count:
            Required number of variables. ``None`` allows any positive number.

        Returns
        -------
        tuple[Variable, ...]
            Validated variables in canonical tuple representation.

        Raises
        ------
        ValueError
            If the input is empty, contains objects other than Variable,
            contains duplicate symbols or does not contain the expected number
            of variables.
        """
        if isinstance(values, Variable):
            normalized = (values,)

        elif (
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
        ):
            normalized = tuple(values)

        else:
            raise ValueError(
                "variables must be a Variable or a sequence "
                "of Variable objects."
            )

        if not normalized:
            raise ValueError(
                "variable collection cannot be empty."
            )

        if not all(
            isinstance(variable, Variable)
            for variable in normalized
        ):
            raise ValueError(
                "variable collection must contain only Variable objects."
            )

        symbols = tuple(
            variable.symbol
            for variable in normalized
        )

        if len(set(symbols)) != len(symbols):
            raise ValueError(
                "variable collection cannot contain duplicate symbols."
            )

        if (
            expected_count is not None
            and len(normalized) != expected_count
        ):
            raise ValueError(
                "variable collection must contain exactly "
                f"{expected_count} variable(s)."
            )

        return normalized

    @staticmethod
    def parameters(
        values: Parameter | Sequence[Parameter] | None,
    ) -> tuple[Parameter, ...]:
        """Validate and normalize model parameters.

        Parameters
        ----------
        values:
            One Parameter, an ordered sequence of Parameter objects or
            ``None``. ``None`` represents a model without adjustable
            parameters.

        Returns
        -------
        tuple[Parameter, ...]
            Validated parameters in canonical tuple representation. An empty
            tuple is returned when ``values`` is ``None`` or an empty
            sequence.

        Raises
        ------
        ValueError
            If the input contains objects other than Parameter or contains
            duplicate parameter symbols.
        """
        if values is None:
            return ()

        if isinstance(values, Parameter):
            normalized = (values,)

        elif (
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
        ):
            normalized = tuple(values)

        else:
            raise ValueError(
                "parameters must be a Parameter, a sequence of Parameter "
                "objects or None."
            )

        if not all(
            isinstance(parameter, Parameter)
            for parameter in normalized
        ):
            raise ValueError(
                "parameter collection must contain only Parameter objects."
            )

        symbols = tuple(
            parameter.symbol
            for parameter in normalized
        )

        if len(set(symbols)) != len(symbols):
            raise ValueError(
                "parameter collection cannot contain duplicate symbols."
            )

        return normalized

    @staticmethod
    def symbolic_expressions(
        values: Expr | Sequence[Expr],
        *,
        expected_count: int | None = None,
    ) -> tuple[Expr, ...]:
        """Validate and normalize one or more SymPy expressions.

        Parameters
        ----------
        values:
            One SymPy Expr or an ordered sequence of Expr objects.
        expected_count:
            Required number of expressions. ``None`` allows any positive
            number.

        Returns
        -------
        tuple[Expr, ...]
            Validated symbolic expressions preserving their order.

        Raises
        ------
        ValueError
            If the collection is empty, contains objects other than SymPy Expr
            or does not contain the expected number of expressions.

        Notes
        -----
        This method verifies only the structural type of the expressions.
        It does not inspect their free symbols or mathematical consistency.
        """
        if isinstance(values, Expr):
            normalized = (values,)

        elif (
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
        ):
            normalized = tuple(values)

        else:
            raise ValueError(
                "symbolic expressions must be a SymPy Expr or a sequence "
                "of SymPy Expr objects."
            )

        if not normalized:
            raise ValueError(
                "symbolic expression collection cannot be empty."
            )

        if not all(
            isinstance(expression, Expr)
            for expression in normalized
        ):
            raise ValueError(
                "symbolic expression collection must contain only "
                "SymPy Expr objects."
            )

        if (
            expected_count is not None
            and len(normalized) != expected_count
        ):
            raise ValueError(
                "symbolic expression collection must contain exactly "
                f"{expected_count} expression(s)."
            )

        return normalized

    @staticmethod
    def numeric_function(
        value: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Validate the numerical function used to evaluate the model.

        Parameters
        ----------
        value:
            Callable representing the numerical model.

        Returns
        -------
        Callable
            The validated callable.

        Raises
        ------
        ValueError
            If ``value`` is not callable.

        Notes
        -----
        The numerical function conceptually evaluates ``Y_hat = F(X, params)``
        where ``X`` and ``Y_hat`` follow the canonical two-dimensional matrix
        representation used by Dataset.

        This method validates only that the supplied object is callable.
        Input shapes, output shapes and numerical behavior are validated by
        the Model Engine or fitting layer.
        """
        if not callable(value):
            raise ValueError(
                "numeric_function must be callable."
            )

        return value

    @staticmethod
    def matching_expression_count(
        dependent_variables: tuple[Variable, ...],
        symbolic_expressions: tuple[Expr, ...],
    ) -> None:
        """Validate one-to-one correspondence between outputs and expressions.

        Parameters
        ----------
        dependent_variables:
            Validated dependent Variable objects.
        symbolic_expressions:
            Validated SymPy expressions.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of dependent variables and symbolic expressions is
            not the same.
        """
        if len(symbolic_expressions) != len(dependent_variables):
            raise ValueError(
                "each dependent variable must have one symbolic expression."
            )

    @staticmethod
    def distinct_symbols(
        independent_variables: tuple[Variable, ...],
        dependent_variables: tuple[Variable, ...],
        parameters: tuple[Parameter, ...],
    ) -> None:
        """Validate uniqueness of mathematical symbols across the model.

        Parameters
        ----------
        independent_variables:
            Validated independent Variable objects.
        dependent_variables:
            Validated dependent Variable objects.
        parameters:
            Validated Parameter objects.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the same mathematical symbol is assigned to more than one
            variable or parameter.
        """
        symbols = [
            variable.symbol
            for variable in independent_variables
        ]

        symbols.extend(
            variable.symbol
            for variable in dependent_variables
        )

        symbols.extend(
            parameter.symbol
            for parameter in parameters
        )

        if len(set(symbols)) != len(symbols):
            raise ValueError(
                "variables and parameters must use distinct symbols."
            )
