"""Tests for ParameterResult."""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from curvefit.domain.parameter_result import ParameterResult


def test_stores_normalized_value_and_optional_uncertainty() -> None:
    result = ParameterResult("a", 2, 0.25)

    assert result.symbol == "a"
    assert result.value == 2.0
    assert result.stderr == 0.25


def test_accepts_missing_uncertainty() -> None:
    assert ParameterResult("a", 2.0).stderr is None


@pytest.mark.parametrize("value", [nan, inf, -inf, True, "2"])
def test_rejects_invalid_fitted_value(value: object) -> None:
    with pytest.raises(ValueError):
        ParameterResult("a", value)


@pytest.mark.parametrize("stderr", [-0.1, inf, nan, True, "0.1"])
def test_rejects_invalid_uncertainty(stderr: object) -> None:
    with pytest.raises(ValueError):
        ParameterResult("a", 2.0, stderr)


def test_is_immutable() -> None:
    result = ParameterResult("a", 2.0)

    with pytest.raises(FrozenInstanceError):
        result.value = 3.0
