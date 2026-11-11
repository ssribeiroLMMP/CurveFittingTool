"""Tests for ModelService built-in model workflows."""

import pytest

from curvefit.services import ModelService


class TestModelDiscovery:
    """Check built-in model discovery through the application service."""

    def test_lists_available_models(self) -> None:
        """Expose built-in model metadata without returning model instances."""
        service = ModelService()

        metadata = service.list_available_models()

        assert tuple(item.key for item in metadata) == (
            "linear",
            "polynomial_2",
            "polynomial_3",
            "exponential",
            "sine",
            "cosine",
        )
        assert service.list_available_models(family="trigonometric") == (
            metadata[-2],
            metadata[-1],
        )

    def test_gets_metadata_by_model_key(self) -> None:
        """Resolve user-facing model information by its stable key."""
        metadata = ModelService().get_model_metadata("linear")

        assert metadata.name == "Linear"
        assert metadata.family == "linear"


class TestOpeningModels:
    """Check that each opened model is a separate editable definition."""

    def test_opening_same_builtin_returns_independent_models(self) -> None:
        """Keep parameter and variable edits isolated between open models."""
        service = ModelService()

        first = service.open_model("linear")
        second = service.open_model("linear")

        first.parameters[0].configure(name="Edited slope", initial_value=4.0)
        first.independent_variables[0].configure(name="Elapsed time")

        assert first.model_id != second.model_id
        assert first.parameters[0] is not second.parameters[0]
        assert first.independent_variables[0] is not second.independent_variables[0]
        assert second.parameters[0].name == "Slope"
        assert second.parameters[0].initial_value == 1.0
        assert second.independent_variables[0].name == "Independent variable"

    def test_opening_a_model_does_not_change_the_catalog_factory(self) -> None:
        """Keep edits to an opened definition out of future model instances."""
        service = ModelService()
        opened = service.open_model("polynomial_2")

        opened.parameters[0].configure(name="Edited coefficient", initial_value=9.0)
        reopened = service.open_model("polynomial_2")

        assert reopened.parameters[0].name == "Coefficient for x^0"
        assert reopened.parameters[0].initial_value == 0.0
        assert reopened.model_id != opened.model_id

    def test_opening_unknown_model_raises_key_error(self) -> None:
        """Report an unknown model key rather than returning a fallback."""
        with pytest.raises(KeyError, match="unknown built-in model key"):
            ModelService().open_model("missing")

    def test_getting_unknown_metadata_raises_key_error(self) -> None:
        """Report missing metadata through the application service."""
        with pytest.raises(KeyError, match="unknown built-in model key"):
            ModelService().get_model_metadata("missing")

    def test_rejects_an_invalid_injected_catalog(self) -> None:
        """Require injected catalogs to implement the supported catalog API."""
        with pytest.raises(ValueError, match="catalog must be a ModelCatalog"):
            ModelService(catalog=object())
