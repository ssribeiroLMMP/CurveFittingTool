"""Tests for ParameterSnapshot."""

from dataclasses import FrozenInstanceError

import pytest

from curvefit.domain.parameter import Parameter
from curvefit.domain.snapshots import ParameterSnapshot


def test_captures_complete_parameter_configuration() -> None:
    parameter = Parameter(
        "a",
        name="Slope",
        initial_value=2.0,
        min_value=-1.0,
        max_value=5.0,
        vary=False,
        unit="m/s",
    )

    snapshot = ParameterSnapshot.from_parameter(parameter)

    assert snapshot.symbol == "a"
    assert snapshot.name == "Slope"
    assert snapshot.initial_value == 2.0
    assert snapshot.min_value == -1.0
    assert snapshot.max_value == 5.0
    assert snapshot.vary is False
    assert snapshot.unit == "m/s"


def test_is_immutable_and_independent_from_source_parameter() -> None:
    parameter = Parameter("a", name="Slope", initial_value=2.0)
    snapshot = ParameterSnapshot.from_parameter(parameter)

    parameter.configure(name="Edited", initial_value=8.0)

    assert snapshot.name == "Slope"
    assert snapshot.initial_value == 2.0
    with pytest.raises(FrozenInstanceError):
        snapshot.initial_value = 4.0


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="initial_value must be between"):
        ParameterSnapshot(
            symbol="a",
            initial_value=10.0,
            min_value=0.0,
            max_value=5.0,
        )


def test_rejects_invalid_source_type() -> None:
    with pytest.raises(ValueError, match="must be a Parameter"):
        ParameterSnapshot.from_parameter(object())
