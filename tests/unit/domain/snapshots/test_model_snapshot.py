"""Tests for ModelSnapshot."""

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest
from sympy import srepr, symbols

from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.parameter import Parameter
from curvefit.domain.snapshots import (
    ModelSnapshot,
    ParameterSnapshot,
    VariableSnapshot,
)
from curvefit.domain.variable import Variable


def _make_model() -> ModelDefinition:
    x, a, b = symbols("x a b")
    return ModelDefinition(
        name="Linear",
        independent_variables=Variable("x", name="Distance", unit="m"),
        dependent_variables=Variable("y", name="Response", unit="Pa"),
        parameters=(
            Parameter("a", name="Slope", initial_value=2.0),
            Parameter("b", name="Intercept", initial_value=1.0),
        ),
        symbolic_expressions=a * x + b,
        numeric_function=lambda inputs, values: inputs,
    )


def test_captures_model_identity_configuration_and_expression() -> None:
    model = _make_model()
    expected_expression = srepr(model.symbolic_expressions[0])

    snapshot = ModelSnapshot.from_model(model)

    assert snapshot.model_id == model.model_id
    assert snapshot.name == "Linear"
    assert snapshot.independent_variables[0].symbol == "x"
    assert snapshot.dependent_variables[0].symbol == "y"
    assert tuple(item.symbol for item in snapshot.parameters) == ("a", "b")
    assert snapshot.symbolic_expressions == (expected_expression,)


def test_nested_snapshots_do_not_follow_model_configuration_changes() -> None:
    model = _make_model()
    snapshot = ModelSnapshot.from_model(model)

    model.independent_variables[0].configure(name="Edited input")
    model.parameters[0].configure(name="Edited slope", initial_value=7.0)

    assert snapshot.independent_variables[0].name == "Distance"
    assert snapshot.parameters[0].name == "Slope"
    assert snapshot.parameters[0].initial_value == 2.0


def test_model_snapshot_is_immutable() -> None:
    snapshot = ModelSnapshot.from_model(_make_model())

    with pytest.raises(FrozenInstanceError):
        snapshot.name = "Edited"


def test_rejects_duplicate_symbols_across_model_components() -> None:
    with pytest.raises(ValueError, match="symbols must be distinct"):
        ModelSnapshot(
            model_id=uuid4(),
            name="Invalid",
            independent_variables=(VariableSnapshot("x"),),
            dependent_variables=(VariableSnapshot("y"),),
            parameters=(ParameterSnapshot("x"),),
            symbolic_expressions=("Symbol('x')",),
        )


def test_rejects_expression_count_that_does_not_match_outputs() -> None:
    with pytest.raises(ValueError, match="one expression"):
        ModelSnapshot(
            model_id=uuid4(),
            name="Invalid",
            independent_variables=(VariableSnapshot("x"),),
            dependent_variables=(VariableSnapshot("y"),),
            parameters=(),
            symbolic_expressions=("Symbol('x')", "Integer(1)"),
        )


def test_rejects_invalid_source_type() -> None:
    with pytest.raises(ValueError, match="must be a ModelDefinition"):
        ModelSnapshot.from_model(object())
