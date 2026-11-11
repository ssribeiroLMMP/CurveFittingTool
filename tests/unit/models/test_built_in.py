"""Tests for the default built-in model catalog."""

from curvefit.models import BUILTIN_MODEL_CATALOG


def test_default_catalog_discovers_linear_and_polynomial_models() -> None:
    """Expose the linear, quadratic, and cubic built-in models."""
    metadata = BUILTIN_MODEL_CATALOG.list_models()

    assert tuple(item.key for item in metadata) == (
        "linear",
        "polynomial_2",
        "polynomial_3",
    )


def test_default_catalog_creates_a_fresh_definition() -> None:
    """Create independent model definitions for repeated catalog lookups."""
    first = BUILTIN_MODEL_CATALOG.create_model("polynomial_2")
    second = BUILTIN_MODEL_CATALOG.create_model("polynomial_2")

    assert first.model_id != second.model_id
    assert first.parameters[0] is not second.parameters[0]
