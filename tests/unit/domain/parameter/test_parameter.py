"""Unit tests for the Parameter domain entity."""

from math import inf

import pytest

from curvefit.domain.parameter import Parameter


class TestConstruction:
    """Tests for Parameter construction and default state."""

    def test_creates_parameter_with_defaults(self) -> None:
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
        assert parameter.value is None
        assert parameter.stderr is None
        assert parameter.is_fitted is False
        assert parameter.has_bounds is False

    def test_creates_parameter_with_custom_configuration(self) -> None:
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
        parameter = Parameter(
            symbol="k",
            name="  Taxa de decaimento  ",
            unit="  1/s  ",
        )

        assert parameter.name == "Taxa de decaimento"
        assert parameter.unit == "1/s"

    def test_rejects_invalid_symbol(self) -> None:
        with pytest.raises(
            ValueError,
            match="symbol cannot be empty",
        ):
            Parameter(symbol="   ")

    def test_rejects_initial_value_outside_bounds(self) -> None:
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
    """Tests for the immutable mathematical identity of Parameter."""

    def test_symbol_has_no_public_setter(self) -> None:
        parameter = Parameter(symbol="A")

        with pytest.raises(AttributeError):
            parameter.symbol = "B"

    def test_symbol_is_preserved_after_configuration(self) -> None:
        parameter = Parameter(symbol="A")

        parameter.configure(
            name="Amplitude",
            initial_value=2.0,
        )

        assert parameter.symbol == "A"


class TestBounds:
    """Tests for derived parameter bound state."""

    def test_has_bounds_is_false_for_unbounded_parameter(self) -> None:
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
        parameter = Parameter(
            symbol="A",
            min_value=minimum,
            max_value=maximum,
        )

        assert parameter.has_bounds is True


class TestConfigure:
    """Tests for complete Parameter reconfiguration."""

    def test_configure_replaces_complete_configuration(self) -> None:
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


class TestFitResult:
    """Tests for fitting result lifecycle."""

    def test_set_fit_result_stores_value_and_stderr(self) -> None:
        parameter = Parameter(
            symbol="A",
            min_value=0.0,
            max_value=10.0,
        )

        parameter.set_fit_result(
            value=5.0,
            stderr=0.2,
        )

        assert parameter.value == 5.0
        assert parameter.stderr == 0.2
        assert parameter.is_fitted is True

    def test_set_fit_result_accepts_missing_stderr(self) -> None:
        parameter = Parameter(symbol="A")

        parameter.set_fit_result(value=2.5)

        assert parameter.value == 2.5
        assert parameter.stderr is None
        assert parameter.is_fitted is True

    def test_rejects_fit_value_outside_bounds(self) -> None:
        parameter = Parameter(
            symbol="A",
            min_value=0.0,
            max_value=10.0,
        )

        with pytest.raises(
            ValueError,
            match="value must be between",
        ):
            parameter.set_fit_result(
                value=11.0,
                stderr=0.1,
            )

    def test_failed_fit_result_preserves_previous_result(self) -> None:
        parameter = Parameter(
            symbol="A",
            min_value=0.0,
            max_value=10.0,
        )

        parameter.set_fit_result(
            value=5.0,
            stderr=0.2,
        )

        with pytest.raises(ValueError):
            parameter.set_fit_result(
                value=6.0,
                stderr=-1.0,
            )

        assert parameter.value == 5.0
        assert parameter.stderr == 0.2

    def test_clear_fit_result_preserves_configuration(self) -> None:
        parameter = Parameter(
            symbol="A",
            name="Amplitude",
            initial_value=2.0,
            min_value=0.0,
            max_value=10.0,
            vary=False,
            unit="m",
        )

        parameter.set_fit_result(
            value=5.0,
            stderr=0.2,
        )

        parameter.clear_fit_result()

        assert parameter.symbol == "A"
        assert parameter.name == "Amplitude"
        assert parameter.initial_value == 2.0
        assert parameter.min_value == 0.0
        assert parameter.max_value == 10.0
        assert parameter.vary is False
        assert parameter.unit == "m"

        assert parameter.value is None
        assert parameter.stderr is None
        assert parameter.is_fitted is False

    def test_configure_clears_previous_fit_result(self) -> None:
        parameter = Parameter(symbol="A")

        parameter.set_fit_result(
            value=2.0,
            stderr=0.1,
        )

        parameter.configure(
            initial_value=3.0,
        )

        assert parameter.value is None
        assert parameter.stderr is None
        assert parameter.is_fitted is False


class TestCopy:
    """Tests for independent Parameter copies."""

    def test_copy_preserves_configuration_and_fit_state(self) -> None:
        parameter = Parameter(
            symbol=r"\mu",
            name="Viscosidade",
            initial_value=2.0,
            min_value=0.0,
            max_value=10.0,
            vary=False,
            unit="Pa.s",
        )

        parameter.set_fit_result(
            value=4.0,
            stderr=0.3,
        )

        copied = parameter.copy()

        assert copied == parameter
        assert copied is not parameter

    def test_copy_is_independent_from_original(self) -> None:
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
    """Tests for Parameter equality semantics."""

    def test_identical_parameters_are_equal(self) -> None:
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
        left = Parameter(symbol="A")
        right = Parameter(symbol="B")

        assert left != right

    def test_different_configuration_is_not_equal(self) -> None:
        left = Parameter(
            symbol="A",
            initial_value=1.0,
        )

        right = Parameter(
            symbol="A",
            initial_value=2.0,
        )

        assert left != right

    def test_different_fit_results_are_not_equal(self) -> None:
        left = Parameter(symbol="A")
        right = Parameter(symbol="A")

        left.set_fit_result(2.0)

        assert left != right

    def test_comparison_with_other_type_is_false(self) -> None:
        parameter = Parameter(symbol="A")

        assert parameter != "A"


class TestRepresentation:
    """Tests for developer-oriented Parameter representation."""

    def test_repr_contains_relevant_parameter_state(self) -> None:
        parameter = Parameter(
            symbol=r"\mu",
            name="Viscosidade",
            initial_value=2.0,
            min_value=0.0,
            max_value=10.0,
            vary=False,
            unit="Pa.s",
        )

        parameter.set_fit_result(
            value=4.0,
            stderr=0.3,
        )

        representation = repr(parameter)

        assert "Parameter(" in representation
        assert r"\\mu" in representation
        assert "Viscosidade" in representation
        assert "initial_value=2.0" in representation
        assert "value=4.0" in representation
        assert "stderr=0.3" in representation
