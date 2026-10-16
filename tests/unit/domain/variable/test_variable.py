"""Unit tests for the Variable domain entity."""

import pytest

from curvefit.domain.variable import Variable


class TestConstruction:
    """Tests for Variable construction and default state."""

    def test_creates_variable_with_defaults(self) -> None:
        variable = Variable(symbol="x")

        assert variable.symbol == "x"
        assert variable.name is Variable.DEFAULT_NAME
        assert variable.unit is Variable.DEFAULT_UNIT

    def test_creates_variable_with_metadata(self) -> None:
        variable = Variable(
            symbol="t",
            name="Tempo",
            unit="s",
        )

        assert variable.symbol == "t"
        assert variable.name == "Tempo"
        assert variable.unit == "s"

    def test_normalizes_optional_metadata(self) -> None:
        variable = Variable(
            symbol="T",
            name="  Temperatura  ",
            unit="  °C  ",
        )

        assert variable.name == "Temperatura"
        assert variable.unit == "°C"

    def test_blank_optional_metadata_becomes_none(self) -> None:
        variable = Variable(
            symbol="T",
            name="   ",
            unit="\t",
        )

        assert variable.name is None
        assert variable.unit is None

    def test_rejects_invalid_symbol(self) -> None:
        with pytest.raises(ValueError):
            Variable(symbol="x+y")


class TestIdentity:
    """Tests for immutable Variable mathematical identity."""

    def test_symbol_has_no_public_setter(self) -> None:
        variable = Variable(symbol="x")

        with pytest.raises(AttributeError):
            variable.symbol = "t"

    def test_symbol_is_preserved_after_configuration(self) -> None:
        variable = Variable(symbol="x")

        variable.configure(name="Posição", unit="m")

        assert variable.symbol == "x"


class TestConfigure:
    """Tests for complete Variable metadata reconfiguration."""

    def test_configure_replaces_complete_metadata(self) -> None:
        variable = Variable(
            symbol="x",
            name="Posição",
            unit="m",
        )

        variable.configure(
            name="Comprimento",
            unit="cm",
        )

        assert variable.symbol == "x"
        assert variable.name == "Comprimento"
        assert variable.unit == "cm"

    def test_configure_without_arguments_restores_defaults(self) -> None:
        variable = Variable(
            symbol="t",
            name="Tempo",
            unit="s",
        )

        variable.configure()

        assert variable.symbol == "t"
        assert variable.name is Variable.DEFAULT_NAME
        assert variable.unit is Variable.DEFAULT_UNIT

    def test_omitted_values_use_defaults_not_previous_values(self) -> None:
        variable = Variable(
            symbol="T",
            name="Temperatura",
            unit="°C",
        )

        variable.configure(name="Temperatura absoluta")

        assert variable.name == "Temperatura absoluta"
        assert variable.unit is None

    def test_failed_configuration_preserves_previous_state(self) -> None:
        variable = Variable(
            symbol="t",
            name="Tempo",
            unit="s",
        )

        with pytest.raises(ValueError):
            variable.configure(
                name="Novo tempo",
                unit=123,
            )

        assert variable.name == "Tempo"
        assert variable.unit == "s"


class TestCopy:
    """Tests for independent Variable copies."""

    def test_copy_preserves_identity_and_metadata(self) -> None:
        variable = Variable(
            symbol=r"\theta",
            name="Ângulo",
            unit="rad",
        )

        copied = variable.copy()

        assert copied == variable
        assert copied is not variable

    def test_copy_is_independent_from_original(self) -> None:
        variable = Variable(
            symbol="t",
            name="Tempo",
            unit="s",
        )

        copied = variable.copy()
        copied.configure(name="Tempo normalizado")

        assert variable.name == "Tempo"
        assert variable.unit == "s"
        assert copied.name == "Tempo normalizado"
        assert copied.unit is None


class TestEquality:
    """Tests for Variable equality semantics."""

    def test_identical_variables_are_equal(self) -> None:
        left = Variable(symbol="x", name="Posição", unit="m")
        right = Variable(symbol="x", name="Posição", unit="m")

        assert left == right

    def test_different_symbols_are_not_equal(self) -> None:
        assert Variable(symbol="x") != Variable(symbol="t")

    def test_different_names_are_not_equal(self) -> None:
        left = Variable(symbol="x", name="Posição")
        right = Variable(symbol="x", name="Comprimento")

        assert left != right

    def test_different_units_are_not_equal(self) -> None:
        left = Variable(symbol="x", unit="m")
        right = Variable(symbol="x", unit="cm")

        assert left != right

    def test_comparison_with_other_type_is_false(self) -> None:
        assert Variable(symbol="x") != "x"


class TestRepresentation:
    """Tests for developer-oriented Variable representation."""

    def test_repr_contains_variable_state(self) -> None:
        variable = Variable(
            symbol=r"\theta",
            name="Ângulo",
            unit="rad",
        )

        representation = repr(variable)

        assert "Variable(" in representation
        assert r"\\theta" in representation
        assert "Ângulo" in representation
        assert "rad" in representation
