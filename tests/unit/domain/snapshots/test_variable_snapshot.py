"""Tests for VariableSnapshot."""

from dataclasses import FrozenInstanceError

import pytest

from curvefit.domain.snapshots import VariableSnapshot
from curvefit.domain.variable import Variable


def test_captures_normalized_variable_metadata() -> None:
    variable = Variable("x", name="Distance", unit="m")

    snapshot = VariableSnapshot.from_variable(variable)

    assert snapshot == VariableSnapshot("x", "Distance", "m")


def test_is_immutable_and_independent_from_source_variable() -> None:
    variable = Variable("x", name="Distance", unit="m")
    snapshot = VariableSnapshot.from_variable(variable)

    variable.configure(name="Position", unit="cm")

    assert snapshot.name == "Distance"
    assert snapshot.unit == "m"
    with pytest.raises(FrozenInstanceError):
        snapshot.name = "Edited"


def test_rejects_invalid_source_type() -> None:
    with pytest.raises(ValueError, match="must be a Variable"):
        VariableSnapshot.from_variable(object())
