"""Metadata and discovery API for built-in mathematical models."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from curvefit.domain.model_definition import ModelDefinition

ModelFactory: TypeAlias = Callable[[], ModelDefinition]


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Describe a built-in model without exposing a mutable model instance."""

    _KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")

    key: str
    name: str
    family: str
    description: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or self._KEY_PATTERN.fullmatch(self.key) is None
        ):
            raise ValueError(
                "key must be a lowercase identifier containing only "
                "letters, digits, and underscores."
            )

        object.__setattr__(self, "name", self._required_text(self.name, "name"))
        object.__setattr__(
            self,
            "family",
            self._required_text(self.family, "family"),
        )
        object.__setattr__(
            self,
            "description",
            self._required_text(self.description, "description"),
        )

    @staticmethod
    def _required_text(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        return value.strip()

class ModelCatalog:
    """Discover built-in models and create them through registered factories.

    Factories must return a fresh ``ModelDefinition`` each time they are
    called. The catalog retains only immutable metadata and factory
    references, never a shared mutable model instance.
    """

    def __init__(
        self,
        registrations: Iterable[tuple[ModelMetadata, ModelFactory]] = (),
    ) -> None:
        registered: dict[str, tuple[ModelMetadata, ModelFactory]] = {}

        for metadata, factory in registrations:
            if not isinstance(metadata, ModelMetadata):
                raise ValueError(
                    "each registration must contain ModelMetadata."
                )
            if not callable(factory):
                raise ValueError(
                    "each model registration must contain a callable factory."
                )
            if metadata.key in registered:
                raise ValueError(
                    f"duplicate model catalog key: {metadata.key!r}."
                )
            registered[metadata.key] = (metadata, factory)

        self._registrations = MappingProxyType(registered)

    def list_models(
        self,
        *,
        family: str | None = None,
    ) -> tuple[ModelMetadata, ...]:
        """Return model metadata in registration order.

        Parameters
        ----------
        family:
            Optional exact family name used to filter the results.
        """
        metadata = tuple(
            registration[0]
            for registration in self._registrations.values()
        )
        if family is None:
            return metadata
        return tuple(item for item in metadata if item.family == family)

    def get_metadata(self, key: str) -> ModelMetadata:
        """Return metadata for a stable catalog key.

        Raises
        ------
        KeyError
            If no built-in model is registered for ``key``.
        """
        try:
            return self._registrations[key][0]
        except KeyError:
            raise KeyError(f"unknown built-in model key: {key!r}.") from None

    def create_model(self, key: str) -> ModelDefinition:
        """Create a fresh model definition using the registered factory.

        Raises
        ------
        KeyError
            If no built-in model is registered for ``key``.
        TypeError
            If a registered factory does not return a ModelDefinition.
        """
        try:
            factory = self._registrations[key][1]
        except KeyError:
            raise KeyError(f"unknown built-in model key: {key!r}.") from None

        model = factory()
        if not isinstance(model, ModelDefinition):
            raise TypeError(
                f"factory for model key {key!r} must return a ModelDefinition."
            )
        return model
