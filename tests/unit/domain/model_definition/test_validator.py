"""Tests for ModelDefinitionValidator structural rules."""

import pytest
from sympy import Expr, symbols

from curvefit.domain.model_definition import ModelDefinitionValidator
from curvefit.domain.parameter import Parameter
from curvefit.domain.variable import Variable


def _numeric_function(*arguments: object) -> tuple[object, ...]:
    """Provide a callable example for validator tests."""
    return arguments


class TestName:
    """Check required model-name validation and normalization."""

    def test_trims_surrounding_whitespace(self) -> None:
        """Return the model name without surrounding whitespace."""
        assert ModelDefinitionValidator.name("  Linear model  ") == "Linear model"

    @pytest.mark.parametrize("value", [None, "", "  ", 42])
    def test_rejects_missing_or_invalid_names(self, value: object) -> None:
        """Reject names that are absent, blank, or not strings."""
        with pytest.raises(ValueError):
            ModelDefinitionValidator.name(value)


class TestVariables:
    """Check variable collection validation and tuple normalization."""

    def test_normalizes_a_single_variable(self) -> None:
        """Wrap one Variable in a tuple."""
        variable = Variable("x")

        assert ModelDefinitionValidator.variables(variable) == (variable,)

    def test_preserves_variable_sequence_order(self) -> None:
        """Retain the order of variables in a supplied sequence."""
        variables = (Variable("x"), Variable("t"))

        assert ModelDefinitionValidator.variables(variables) == variables

    @pytest.mark.parametrize("value", [None, "x", 1])
    def test_rejects_non_variable_inputs(self, value: object) -> None:
        """Reject inputs that are neither a Variable nor a sequence of them."""
        with pytest.raises(ValueError, match="variables must be a Variable"):
            ModelDefinitionValidator.variables(value)

    def test_rejects_empty_variable_sequence(self) -> None:
        """Require at least one variable in a sequence."""
        with pytest.raises(ValueError, match="variable collection cannot be empty"):
            ModelDefinitionValidator.variables(())

    def test_rejects_non_variable_sequence_items(self) -> None:
        """Require every item in the variable sequence to be a Variable."""
        with pytest.raises(ValueError, match="only Variable objects"):
            ModelDefinitionValidator.variables((Variable("x"), object()))

    def test_rejects_duplicate_symbols_in_variable_sequence(self) -> None:
        """Require symbols to be unique within a variable collection."""
        with pytest.raises(ValueError, match="duplicate symbols"):
            ModelDefinitionValidator.variables((Variable("x"), Variable("x")))

    def test_enforces_expected_variable_count(self) -> None:
        """Reject a valid variable collection with the wrong required count."""
        with pytest.raises(ValueError, match="exactly 1 variable"):
            ModelDefinitionValidator.variables(
                (Variable("x"), Variable("t")),
                expected_count=1,
            )


class TestParameters:
    """Check parameter collection validation and tuple normalization."""

    def test_normalizes_absent_parameters_to_empty_tuple(self) -> None:
        """Treat None as a model with no adjustable parameters."""
        assert ModelDefinitionValidator.parameters(None) == ()

    def test_accepts_an_empty_parameter_sequence(self) -> None:
        """Allow an empty sequence to express a parameter-free model."""
        assert ModelDefinitionValidator.parameters(()) == ()

    def test_normalizes_a_single_parameter(self) -> None:
        """Wrap one Parameter in a tuple."""
        parameter = Parameter("a")

        assert ModelDefinitionValidator.parameters(parameter) == (parameter,)

    def test_preserves_parameter_sequence_order(self) -> None:
        """Retain parameter ordering in the normalized tuple."""
        parameters = (Parameter("a"), Parameter("b"))

        assert ModelDefinitionValidator.parameters(parameters) == parameters

    @pytest.mark.parametrize("value", ["a", 1])
    def test_rejects_non_parameter_inputs(self, value: object) -> None:
        """Reject inputs that are neither None nor Parameter collections."""
        with pytest.raises(ValueError, match="parameters must be a Parameter"):
            ModelDefinitionValidator.parameters(value)

    def test_rejects_non_parameter_sequence_items(self) -> None:
        """Require every item in a parameter sequence to be a Parameter."""
        with pytest.raises(ValueError, match="only Parameter objects"):
            ModelDefinitionValidator.parameters((Parameter("a"), object()))

    def test_rejects_duplicate_parameter_symbols(self) -> None:
        """Require each parameter in a model to have a distinct symbol."""
        with pytest.raises(ValueError, match="duplicate symbols"):
            ModelDefinitionValidator.parameters((Parameter("a"), Parameter("a")))


class TestSymbolicExpressions:
    """Check symbolic-expression collection validation and normalization."""

    def test_normalizes_a_single_expression(self) -> None:
        """Wrap one SymPy expression in a tuple."""
        expression = symbols("x") + 1

        assert ModelDefinitionValidator.symbolic_expressions(expression) == (
            expression,
        )

    def test_preserves_expression_sequence_order(self) -> None:
        """Retain the order of expressions in a supplied sequence."""
        x = symbols("x")
        expressions = (x, x**2)

        assert ModelDefinitionValidator.symbolic_expressions(expressions) == expressions

    @pytest.mark.parametrize("value", [None, "x", 1])
    def test_rejects_non_expression_inputs(self, value: object) -> None:
        """Reject values that are neither SymPy expressions nor their sequence."""
        with pytest.raises(ValueError, match="must be a SymPy Expr"):
            ModelDefinitionValidator.symbolic_expressions(value)

    def test_rejects_empty_expression_sequence(self) -> None:
        """Require at least one symbolic expression."""
        with pytest.raises(
            ValueError,
            match="symbolic expression collection cannot be empty",
        ):
            ModelDefinitionValidator.symbolic_expressions(())

    def test_rejects_non_expression_sequence_items(self) -> None:
        """Require every item in an expression sequence to be a SymPy Expr."""
        with pytest.raises(ValueError, match="only SymPy Expr objects"):
            ModelDefinitionValidator.symbolic_expressions(
                (symbols("x"), object())
            )

    def test_enforces_expected_expression_count(self) -> None:
        """Reject a valid expression collection with the wrong required count."""
        x = symbols("x")

        with pytest.raises(ValueError, match="exactly 1 expression"):
            ModelDefinitionValidator.symbolic_expressions(
                (x, x**2),
                expected_count=1,
            )


class TestNumericFunction:
    """Check validation of the callable numerical model representation."""

    def test_accepts_callable_values(self) -> None:
        """Return the supplied callable unchanged."""
        assert (
            ModelDefinitionValidator.numeric_function(_numeric_function)
            is _numeric_function
        )

    @pytest.mark.parametrize("value", [None, "function", 1])
    def test_rejects_non_callable_values(self, value: object) -> None:
        """Reject values that cannot be used as numerical model functions."""
        with pytest.raises(ValueError, match="numeric_function must be callable"):
            ModelDefinitionValidator.numeric_function(value)


class TestMatchingExpressionCount:
    """Check one-to-one correspondence between outputs and expressions."""

    def test_accepts_matching_dependent_and_expression_counts(self) -> None:
        """Allow one expression for each dependent variable."""
        x = symbols("x")

        assert ModelDefinitionValidator.matching_expression_count(
            (Variable("y"),),
            (Expr(x),),
        ) is None

    def test_rejects_mismatched_dependent_and_expression_counts(self) -> None:
        """Reject output definitions without a corresponding expression."""
        x = symbols("x")

        with pytest.raises(
            ValueError,
            match="each dependent variable must have one symbolic expression",
        ):
            ModelDefinitionValidator.matching_expression_count(
                (Variable("y"),),
                (x, x**2),
            )


class TestDistinctSymbols:
    """Check symbol uniqueness across variables and parameters."""

    def test_accepts_distinct_variable_and_parameter_symbols(self) -> None:
        """Allow a model whose variables and parameters use unique symbols."""
        assert ModelDefinitionValidator.distinct_symbols(
            (Variable("x"),),
            (Variable("y"),),
            (Parameter("a"),),
        ) is None

    @pytest.mark.parametrize(
        ("independent", "dependent", "parameters"),
        [
            ((Variable("x"),), (Variable("x"),), ()),
            ((Variable("x"),), (Variable("y"),), (Parameter("x"),)),
            ((Variable("x"),), (Variable("y"),), (Parameter("y"),)),
        ],
    )
    def test_rejects_symbol_collisions_across_components(
        self,
        independent: tuple[Variable, ...],
        dependent: tuple[Variable, ...],
        parameters: tuple[Parameter, ...],
    ) -> None:
        """Reject collisions between either variable and any other component."""
        with pytest.raises(
            ValueError,
            match="variables and parameters must use distinct symbols",
        ):
            ModelDefinitionValidator.distinct_symbols(
                independent,
                dependent,
                parameters,
            )
