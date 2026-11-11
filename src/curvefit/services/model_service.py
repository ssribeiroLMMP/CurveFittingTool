"""Application service for discovering and opening built-in models."""

from __future__ import annotations

from curvefit.domain.model_definition import ModelDefinition
from curvefit.models import BUILTIN_MODEL_CATALOG, ModelCatalog, ModelMetadata


class ModelService:
    """Provide built-in model discovery and fresh editable definitions.

    The service keeps the GUI independent from catalog implementation details.
    Each call to ``open_model`` delegates to a model factory and returns a
    separate ``ModelDefinition`` for editing.
    """

    def __init__(self, catalog: ModelCatalog | None = None) -> None:
        """Create a service using the built-in catalog or an injected catalog.

        Raises
        ------
        ValueError
            If ``catalog`` is not a ``ModelCatalog`` instance.
        """
        if catalog is not None and not isinstance(catalog, ModelCatalog):
            raise ValueError("catalog must be a ModelCatalog.")
        self._catalog = (
            BUILTIN_MODEL_CATALOG
            if catalog is None
            else catalog
        )

    def list_available_models(
        self,
        *,
        family: str | None = None,
    ) -> tuple[ModelMetadata, ...]:
        """List built-in model metadata, optionally filtered by family."""
        return self._catalog.list_models(family=family)

    def get_model_metadata(self, model_key: str) -> ModelMetadata:
        """Return discovery metadata for a stable built-in model key."""
        return self._catalog.get_metadata(model_key)

    def open_model(self, model_key: str) -> ModelDefinition:
        """Open a fresh built-in model definition for editing.

        Raises
        ------
        KeyError
            If ``model_key`` does not identify a built-in model.
        """
        return self._catalog.create_model(model_key)
