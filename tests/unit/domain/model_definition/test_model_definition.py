"""Tests for the ModelDefinition domain entity."""

from uuid import UUID

import pytest
from sympy import symbols

from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.parameter import Parameter
from curvefit.domain.variable import Variable


def _numeric_function(*arguments: object) -> tuple[object, ...]:
    """Provide a stable callable for model-definition construction tests."""
    return arguments


def _make_model_definition() -> ModelDefinition:
    """Create a model definition with one input, one output, and two parameters."""
    x, a, b = symbols("x a b")
    return ModelDefinition(
        name="Linear",
        independent_variables=Variable("x", name="Time"),
        dependent_variables=Variable("y", name="Response"),
        parameters=(Parameter("a"), Parameter("b")),
        symbolic_expressions=a * x + b,
        numeric_function=_numeric_function,
    )


class TestConstruction:
    """Check construction, normalization, and validation of model definitions."""

    def test_normalizes_single_objects_to_tuples(self) -> None:
        """Store individual variables and expressions in the canonical tuple form."""
        x, a, b = symbols("x a b")
        independent = Variable("x")
        dependent = Variable("y")
        parameter = Parameter("a")
        expression = a * x + b

        model = ModelDefinition(
            name="Linear",
            independent_variables=independent,
            dependent_variables=dependent,
            parameters=parameter,
            symbolic_expressions=expression,
            numeric_function=_numeric_function,
        )

        assert model.independent_variables == (independent,)
        assert model.dependent_variables == (dependent,)
        assert model.parameters == (parameter,)
        assert model.symbolic_expressions == (expression,)
        assert model.independent_variables[0] is independent
        assert model.dependent_variables[0] is dependent
        assert model.parameters[0] is parameter
        assert model.numeric_function is _numeric_function

    def test_preserves_parameter_order(self) -> None:
        """Keep parameters in the same order supplied by the caller."""
        model = _make_model_definition()

        assert tuple(parameter.symbol for parameter in model.parameters) == (
            "a",
            "b",
        )

    def test_normalizes_absent_parameters_to_empty_tuple(self) -> None:
        """Represent a model without adjustable parameters using an empty tuple."""
        x = symbols("x")
        model = ModelDefinition(
            name="Identity",
            independent_variables=Variable("x"),
            dependent_variables=Variable("y"),
            parameters=None,
            symbolic_expressions=x,
            numeric_function=_numeric_function,
        )

        assert model.parameters == ()

    def test_generates_a_distinct_uuid_when_model_id_is_omitted(self) -> None:
        """Assign each newly constructed model a distinct UUID by default."""
        first = _make_model_definition()
        second = _make_model_definition()

        assert isinstance(first.model_id, UUID)
        assert first.model_id != second.model_id

    def test_preserves_a_supplied_model_id(self) -> None:
        """Use the UUID explicitly supplied during construction."""
        model_id = UUID("12345678-1234-5678-1234-567812345678")

        model = ModelDefinition(
            name="Linear",
            independent_variables=Variable("x"),
            dependent_variables=Variable("y"),
            parameters=None,
            symbolic_expressions=symbols("x"),
            numeric_function=_numeric_function,
            model_id=model_id,
        )

        assert model.model_id is model_id

    def test_model_id_cannot_be_reassigned(self) -> None:
        """Keep the model identifier immutable through its public property."""
        model = _make_model_definition()

        with pytest.raises(AttributeError):
            model.model_id = UUID("12345678-1234-5678-1234-567812345678")

    def test_rejects_duplicate_symbols_across_model_components(self) -> None:
        """Reject a parameter whose symbol collides with an input variable."""
        x = symbols("x")

        with pytest.raises(
            ValueError,
            match="variables and parameters must use distinct symbols",
        ):
            ModelDefinition(
                name="Invalid",
                independent_variables=Variable("x"),
                dependent_variables=Variable("y"),
                parameters=Parameter("x"),
                symbolic_expressions=x,
                numeric_function=_numeric_function,
            )

    def test_rejects_more_independent_variables_than_supported(self) -> None:
        """Enforce the current one-input ModelDefinition contract."""
        x, z = symbols("x z")

        with pytest.raises(ValueError, match="exactly 1 variable"):
            ModelDefinition(
                name="Invalid",
                independent_variables=(Variable("x"), Variable("z")),
                dependent_variables=Variable("y"),
                parameters=None,
                symbolic_expressions=x + z,
                numeric_function=_numeric_function,
            )

    def test_rejects_mismatched_dependent_variables_and_expressions(self) -> None:
        """Require one symbolic expression for every dependent variable."""
        x = symbols("x")

        with pytest.raises(
            ValueError,
            match="each dependent variable must have one symbolic expression",
        ):
            ModelDefinition(
                name="Invalid",
                independent_variables=Variable("x"),
                dependent_variables=Variable("y"),
                parameters=None,
                symbolic_expressions=(x, x**2),
                numeric_function=_numeric_function,
            )

    def test_rejects_a_non_callable_numeric_function(self) -> None:
        """Require the model's numerical representation to be callable."""
        x = symbols("x")

        with pytest.raises(ValueError, match="numeric_function must be callable"):
            ModelDefinition(
                name="Invalid",
                independent_variables=Variable("x"),
                dependent_variables=Variable("y"),
                parameters=None,
                symbolic_expressions=x,
                numeric_function=object(),
            )


class TestCopy:
    """Check copied model identity and independent configuration objects."""

    def test_copy_has_new_id_and_independent_variables_and_parameters(self) -> None:
        """Copy configurable objects independently while retaining model structure."""
        model = _make_model_definition()

        copied = model.copy()

        assert copied is not model
        assert copied.model_id != model.model_id
        assert copied.independent_variables[0] is not model.independent_variables[0]
        assert copied.dependent_variables[0] is not model.dependent_variables[0]
        assert copied.parameters[0] is not model.parameters[0]
        assert copied.parameters[1] is not model.parameters[1]
        assert copied.symbolic_expressions is model.symbolic_expressions
        assert copied.numeric_function is model.numeric_function

    def test_configuration_changes_to_copy_do_not_affect_original(self) -> None:
        """Keep Variable and Parameter configuration independent after copying."""
        model = _make_model_definition()
        copied = model.copy()

        copied.independent_variables[0].configure(name="Copied input")
        copied.parameters[0].configure(name="Copied slope", initial_value=2.0)

        assert model.independent_variables[0].name == "Time"
        assert model.parameters[0].name is None
        assert model.parameters[0].initial_value == 1.0
        assert copied.independent_variables[0].name == "Copied input"
        assert copied.parameters[0].name == "Copied slope"
        assert copied.parameters[0].initial_value == 2.0


class TestRepresentation:
    """Check the developer-facing representation of a model definition."""

    def test_repr_reports_name_and_component_counts(self) -> None:
        """Include the model name and component counts in the representation."""
        representation = repr(_make_model_definition())

        assert "ModelDefinition(" in representation
        assert "name='Linear'" in representation
        assert "independent_variables=1" in representation
        assert "dependent_variables=1" in representation
        assert "parameters=2" in representation
