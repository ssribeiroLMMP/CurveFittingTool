"""Tests for the FitEngine abstraction."""

import pytest

from curvefit.fitting import FitEngine


def test_fit_engine_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError, match="abstract"):
        FitEngine()


def test_subclass_must_implement_fit_contract() -> None:
    class IncompleteEngine(FitEngine):
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteEngine()
