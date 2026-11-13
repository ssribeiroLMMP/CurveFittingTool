"""Tests for the domain exception hierarchy."""

from curvefit.domain.exceptions import (
    CurveFitError,
    FitError,
    SnapshotValidationError,
)


def test_fit_error_can_be_handled_as_application_error() -> None:
    error = FitError("solver failed")

    assert isinstance(error, CurveFitError)
    assert str(error) == "solver failed"


def test_snapshot_error_can_be_handled_as_application_error() -> None:
    error = SnapshotValidationError("invalid snapshot")

    assert isinstance(error, CurveFitError)
    assert str(error) == "invalid snapshot"
