"""Domain-specific exceptions raised by fitting and snapshot workflows."""


class CurveFitError(Exception):
    """Base exception for controlled curve-fitting application failures."""


class FitError(CurveFitError):
    """Report a numerical fitting failure through the engine abstraction."""


class SnapshotValidationError(CurveFitError):
    """Report invalid persisted or reconstructed snapshot content."""
