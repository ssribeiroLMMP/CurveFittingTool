"""Tests for built-in model metadata and discovery."""

import pytest
from sympy import symbols

from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.variable import Variable
from curvefit.models import ModelCatalog, ModelMetadata


def _make_model(name: str = "Identity") -> ModelDefinition:
    """Create a valid parameter-free model for catalog tests."""
    x = symbols("x")
    return ModelDefinition(
        name=name,
        independent_variables=Variable("x"),
        dependent_variables=Variable("y"),
        parameters=None,
        symbolic_expressions=x,
        numeric_function=lambda *arguments: arguments[0],
    )


def _make_catalog() -> ModelCatalog:
    """Create a catalog with two entries from different families."""
    return ModelCatalog(
        (
            (
                ModelMetadata(
                    key="identity",
                    name="Identity",
                    family="linear",
                    description="A parameter-free identity model.",
                ),
                _make_model,
            ),
            (
                ModelMetadata(
                    key="square",
                    name="Square",
                    family="polynomial",
                    description="A parameter-free quadratic model.",
                ),
                lambda: _make_model("Square"),
            ),
        )
    )


class TestModelMetadata:
    """Check stable keys and user-facing metadata validation."""

    def test_strips_metadata_text(self) -> None:
        """Normalize display metadata while preserving the stable key."""
        metadata = ModelMetadata(
            key="linear",
            name=" Linear ",
            family=" linear ",
            description=" A straight line. ",
        )

        assert metadata.key == "linear"
        assert metadata.name == "Linear"
        assert metadata.family == "linear"
        assert metadata.description == "A straight line."

    @pytest.mark.parametrize(
        ("key", "name", "family", "description"),
        [
            ("Linear", "Linear", "linear", "A straight line."),
            ("two words", "Linear", "linear", "A straight line."),
            ("linear", "", "linear", "A straight line."),
            ("linear", "Linear", " ", "A straight line."),
            ("linear", "Linear", "linear", ""),
        ],
    )
    def test_rejects_invalid_metadata(
        self,
        key: str,
        name: str,
        family: str,
        description: str,
    ) -> None:
        """Require a stable key and non-empty user-facing metadata."""
        with pytest.raises(ValueError):
            ModelMetadata(key, name, family, description)

    def test_metadata_properties_are_read_only(self) -> None:
        """Prevent changes to metadata after it is added to a catalog."""
        metadata = ModelMetadata("linear", "Linear", "linear", "A straight line.")

        with pytest.raises(AttributeError):
            metadata.key = "quadratic"


class TestModelCatalog:
    """Check model discovery and factory-backed lookup behavior."""

    def test_lists_metadata_in_registration_order(self) -> None:
        """Return metadata without constructing model instances."""
        catalog = _make_catalog()

        metadata = catalog.list_models()

        assert tuple(item.key for item in metadata) == ("identity", "square")

    def test_filters_metadata_by_family(self) -> None:
        """Return only entries matching the requested model family."""
        catalog = _make_catalog()

        metadata = catalog.list_models(family="polynomial")

        assert tuple(item.key for item in metadata) == ("square",)
        assert catalog.list_models(family="unknown") == ()

    def test_gets_metadata_by_stable_key(self) -> None:
        """Find a registered model using its stable catalog key."""
        catalog = _make_catalog()

        assert catalog.get_metadata("identity").name == "Identity"

    def test_raises_for_unknown_metadata_key(self) -> None:
        """Report an unknown key instead of returning an empty fallback."""
        with pytest.raises(KeyError, match="unknown built-in model key"):
            _make_catalog().get_metadata("missing")

    def test_creates_a_new_model_for_each_factory_call(self) -> None:
        """Invoke the registered factory for every model request."""
        catalog = _make_catalog()

        first = catalog.create_model("identity")
        second = catalog.create_model("identity")

        assert first is not second
        assert first.model_id != second.model_id

    def test_raises_for_unknown_model_key(self) -> None:
        """Report an unknown key rather than returning a default model."""
        with pytest.raises(KeyError, match="unknown built-in model key"):
            _make_catalog().create_model("missing")

    def test_rejects_duplicate_registration_keys(self) -> None:
        """Prevent ambiguous model lookup when a key is registered twice."""
        metadata = ModelMetadata("linear", "Linear", "linear", "A straight line.")

        with pytest.raises(ValueError, match="duplicate model catalog key"):
            ModelCatalog(((metadata, _make_model), (metadata, _make_model)))

    def test_rejects_invalid_registration_metadata(self) -> None:
        """Require each registration to pair a ModelMetadata with a factory."""
        with pytest.raises(ValueError, match="ModelMetadata"):
            ModelCatalog(((object(), _make_model),))

    def test_rejects_non_callable_factories(self) -> None:
        """Require registered model factories to be callable."""
        metadata = ModelMetadata("linear", "Linear", "linear", "A straight line.")

        with pytest.raises(ValueError, match="callable factory"):
            ModelCatalog(((metadata, object()),))

    def test_rejects_factories_returning_the_wrong_type(self) -> None:
        """Reject a factory result that cannot be used as a model definition."""
        metadata = ModelMetadata("linear", "Linear", "linear", "A straight line.")
        catalog = ModelCatalog(((metadata, lambda: object()),))

        with pytest.raises(TypeError, match="must return a ModelDefinition"):
            catalog.create_model("linear")
