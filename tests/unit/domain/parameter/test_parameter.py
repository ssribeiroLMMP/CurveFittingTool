"""Unit tests for the Parameter domain entity."""

from math import inf

import pytest

from curvefit.domain.parameter import Parameter


class TestConstruction:
    """Check that construction stores only parameter configuration."""

    def test_creates_parameter_with_defaults(self) -> None:
        """Check that omitted options use their defaults and leave it unbounded."""
        parameter = Parameter(symbol="A")

        assert parameter.symbol == "A"
        assert parameter.name is None
        assert parameter.initial_value == Parameter.DEFAULT_INITIAL_VALUE
        assert parameter.min_value == Parameter.DEFAULT_MIN_VALUE
        assert parameter.max_value == Parameter.DEFAULT_MAX_VALUE
        assert parameter.bounds == (
            Parameter.DEFAULT_MIN_VALUE,
            Parameter.DEFAULT_MAX_VALUE,
        )
        assert parameter.vary is Parameter.DEFAULT_VARY
        assert parameter.unit is None
        assert parameter.has_bounds is False

    def test_does_not_expose_fit_results_or_lifecycle_methods(self) -> None:
        """Check that fit results and fit-state methods are absent from Parameter."""
        parameter = Parameter(symbol="A")

        assert not hasattr(parameter, "value")
        assert not hasattr(parameter, "stderr")
        assert not hasattr(parameter, "is_fitted")
        assert not hasattr(parameter, "set_fit_result")
        assert not hasattr(parameter, "clear_fit_result")

    def test_creates_parameter_with_custom_configuration(self) -> None:
        """Check that construction preserves the supplied parameter options."""
        parameter = Parameter(
            symbol=r"\mu",
            name="Viscosidade",
            initial_value=2.0,
            min_value=0.0,
            max_value=10.0,
            vary=False,
            unit="Pa.s",
        )

        assert parameter.symbol == r"\mu"
        assert parameter.name == "Viscosidade"
        assert parameter.initial_value == 2.0
        assert parameter.min_value == 0.0
        assert parameter.max_value == 10.0
        assert parameter.vary is False
        assert parameter.unit == "Pa.s"

    def test_normalizes_optional_text_fields(self) -> None:
        """Check that surrounding whitespace is removed from the name and unit."""
        parameter = Parameter(
            symbol="k",
            name="  Taxa de decaimento  ",
            unit="  1/s  ",
        )

        assert parameter.name == "Taxa de decaimento"
        assert parameter.unit == "1/s"

    def test_rejects_invalid_symbol(self) -> None:
        """Check that a blank mathematical symbol is rejected."""
        with pytest.raises(
            ValueError,
            match="symbol cannot be empty",
        ):
            Parameter(symbol="   ")

    def test_rejects_initial_value_outside_bounds(self) -> None:
        """Check that construction rejects a starting value outside its bounds."""
        with pytest.raises(
            ValueError,
            match="initial_value must be between",
        ):
            Parameter(
                symbol="A",
                initial_value=11.0,
                min_value=0.0,
                max_value=10.0,
            )


class TestIdentity:
    """Check that the mathematical symbol remains immutable."""

    def test_symbol_has_no_public_setter(self) -> None:
        """Check that assigning a new symbol through the public property fails."""
        parameter = Parameter(symbol="A")

        with pytest.raises(AttributeError):
            parameter.symbol = "B"

    def test_symbol_is_preserved_after_configuration(self) -> None:
        """Check that changing configuration does not change the symbol."""
        parameter = Parameter(symbol="A")

        parameter.configure(
            name="Amplitude",
            initial_value=2.0,
        )

        assert parameter.symbol == "A"


class TestBounds:
    """Check how configured bounds determine the derived bounded state."""

    def test_has_bounds_is_false_for_unbounded_parameter(self) -> None:
        """Check that infinite bounds do not count as constraints."""
        parameter = Parameter(symbol="A")

        assert parameter.has_bounds is False

    @pytest.mark.parametrize(
        ("minimum", "maximum"),
        [
            (0.0, inf),
            (-inf, 10.0),
            (0.0, 10.0),
        ],
    )
    def test_has_bounds_is_true_when_at_least_one_bound_is_finite(
        self,
        minimum,
        maximum,
    ) -> None:
        """Check that any finite lower or upper bound marks the parameter bounded."""
        parameter = Parameter(
            symbol="A",
            min_value=minimum,
            max_value=maximum,
        )

        assert parameter.has_bounds is True


class TestConfigure:
    """Check complete configuration replacement and validation."""

    def test_configure_replaces_complete_configuration(self) -> None:
        """Check that configure replaces every option while preserving identity."""
        parameter = Parameter(
            symbol="A",
            name="Old name",
            initial_value=4.0,
            min_value=0.0,
            max_value=10.0,
            vary=False,
            unit="m",
        )

        parameter.configure(
            name="Amplitude",
            initial_value=2.0,
            min_value=-5.0,
            max_value=5.0,
            vary=True,
            unit="Pa",
        )

        assert parameter.symbol == "A"
        assert parameter.name == "Amplitude"
        assert parameter.initial_value == 2.0
        assert parameter.min_value == -5.0
        assert parameter.max_value == 5.0
        assert parameter.vary is True
        assert parameter.unit == "Pa"

    def test_configure_without_arguments_restores_defaults(self) -> None:
        """Check that calling configure without options restores all defaults."""
        parameter = Parameter(
            symbol="k",
            name="Taxa",
            initial_value=3.0,
            min_value=0.0,
            max_value=5.0,
            vary=False,
            unit="1/s",
        )

        parameter.configure()

        assert parameter.symbol == "k"
        assert parameter.name is Parameter.DEFAULT_NAME
        assert parameter.initial_value == Parameter.DEFAULT_INITIAL_VALUE
        assert parameter.min_value == Parameter.DEFAULT_MIN_VALUE
        assert parameter.max_value == Parameter.DEFAULT_MAX_VALUE
        assert parameter.vary is Parameter.DEFAULT_VARY
        assert parameter.unit is Parameter.DEFAULT_UNIT

    def test_omitted_values_use_defaults_not_previous_values(self) -> None:
        """Check that omitted options reset to defaults rather than retain old values."""
        parameter = Parameter(
            symbol="A",
            name="Amplitude",
            initial_value=4.0,
            min_value=0.0,
            max_value=10.0,
            vary=False,
            unit="m",
        )

        parameter.configure(
            initial_value=2.0,
        )

        assert parameter.name is None
        assert parameter.initial_value == 2.0
        assert parameter.min_value == -inf
        assert parameter.max_value == inf
        assert parameter.vary is True
        assert parameter.unit is None

    def test_failed_configuration_preserves_previous_state(self) -> None:
        """Check that invalid options raise an error without changing configuration."""
        parameter = Parameter(
            symbol="A",
            name="Amplitude",
            initial_value=2.0,
            min_value=0.0,
            max_value=10.0,
            vary=False,
            unit="m",
        )

        with pytest.raises(ValueError):
            parameter.configure(
                initial_value=20.0,
                min_value=0.0,
                max_value=10.0,
            )

        assert parameter.name == "Amplitude"
        assert parameter.initial_value == 2.0
        assert parameter.min_value == 0.0
        assert parameter.max_value == 10.0
        assert parameter.vary is False
        assert parameter.unit == "m"


class TestCopy:
    """Check that copied parameters retain configuration independently."""

    def test_copy_preserves_configuration(self) -> None:
        """Check that a copy has equal configuration but is a distinct object."""
        parameter = Parameter(
            symbol=r"\mu",
            name="Viscosidade",
            initial_value=2.0,
            min_value=0.0,
            max_value=10.0,
            vary=False,
            unit="Pa.s",
        )

        copied = parameter.copy()

        assert copied == parameter
        assert copied is not parameter

    def test_copy_is_independent_from_original(self) -> None:
        """Check that reconfiguring a copy leaves the original unchanged."""
        parameter = Parameter(
            symbol="A",
            name="Amplitude",
        )

        copied = parameter.copy()

        copied.configure(
            name="Outra amplitude",
            initial_value=5.0,
        )

        assert parameter.name == "Amplitude"
        assert parameter.initial_value == 1.0

        assert copied.name == "Outra amplitude"
        assert copied.initial_value == 5.0


class TestEquality:
    """Check that equality compares parameter identity and configuration."""

    def test_identical_parameters_are_equal(self) -> None:
        """Check that separate parameters with matching options compare equal."""
        left = Parameter(
            symbol="A",
            name="Amplitude",
        )

        right = Parameter(
            symbol="A",
            name="Amplitude",
        )

        assert left == right

    def test_different_symbols_are_not_equal(self) -> None:
        """Check that parameters with different mathematical symbols compare unequal."""
        left = Parameter(symbol="A")
        right = Parameter(symbol="B")

        assert left != right

    def test_different_configuration_is_not_equal(self) -> None:
        """Check that different initial values make otherwise matching parameters unequal."""
        left = Parameter(
            symbol="A",
            initial_value=1.0,
        )

        right = Parameter(
            symbol="A",
            initial_value=2.0,
        )

        assert left != right

    def test_comparison_with_other_type_is_false(self) -> None:
        """Check that a Parameter does not compare equal to a string."""
        parameter = Parameter(symbol="A")

        assert parameter != "A"


class TestRepresentation:
    """Check the developer-oriented representation of a Parameter."""

    def test_repr_contains_parameter_configuration(self) -> None:
        """Check that repr includes the symbol and configured options."""
        parameter = Parameter(
            symbol=r"\mu",
            name="Viscosidade",
            initial_value=2.0,
            min_value=0.0,
            max_value=10.0,
            vary=False,
            unit="Pa.s",
        )

        representation = repr(parameter)

        assert "Parameter(" in representation
        assert r"\\mu" in representation
        assert "Viscosidade" in representation
        assert "initial_value=2.0" in representation
        assert "min_value=0.0" in representation
        assert "max_value=10.0" in representation
        assert "vary=False" in representation
        assert "unit='Pa.s'" in representation
