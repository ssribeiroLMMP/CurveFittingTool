"""Tests for ParameterMapper."""

import numpy as np
import pytest

from curvefit.domain.parameter import Parameter
from curvefit.fitting import ParameterMapper


def _make_parameters() -> tuple[Parameter, ...]:
    return (
        Parameter(
            "a",
            initial_value=2.0,
            min_value=0.0,
            max_value=5.0,
        ),
        Parameter(
            "b",
            initial_value=3.0,
            vary=False,
        ),
        Parameter(
            "c",
            initial_value=-1.0,
            min_value=-2.0,
            max_value=2.0,
        ),
    )


def test_maps_varying_and_fixed_parameter_indices() -> None:
    mapper = ParameterMapper(_make_parameters())

    assert mapper.varying_indices == (0, 2)
    assert mapper.fixed_indices == (1,)


def test_returns_solver_initial_values_and_bounds() -> None:
    mapper = ParameterMapper(_make_parameters())

    np.testing.assert_array_equal(mapper.initial_values(), [2.0, -1.0])
    np.testing.assert_array_equal(
        mapper.bounds(),
        [[0.0, -2.0], [5.0, 2.0]],
    )


def test_expands_solver_values_and_preserves_fixed_initial_values() -> None:
    mapper = ParameterMapper(_make_parameters())

    expanded = mapper.expand_values(np.array([4.0, 1.5]))

    np.testing.assert_array_equal(expanded, [4.0, 3.0, 1.5])


def test_captures_configuration_at_construction_time() -> None:
    parameters = _make_parameters()
    mapper = ParameterMapper(parameters)

    parameters[0].configure(initial_value=4.0, min_value=-10.0, max_value=10.0)
    parameters[1].configure(initial_value=8.0, vary=True)

    assert mapper.varying_indices == (0, 2)
    assert mapper.fixed_indices == (1,)
    np.testing.assert_array_equal(mapper.initial_values(), [2.0, -1.0])
    np.testing.assert_array_equal(
        mapper.expand_values(np.array([4.0, 1.5])),
        [4.0, 3.0, 1.5],
    )


def test_solver_configuration_arrays_are_read_only() -> None:
    mapper = ParameterMapper(_make_parameters())

    assert mapper.initial_values().flags.writeable is False
    assert mapper.bounds().flags.writeable is False
    with pytest.raises(ValueError):
        mapper.initial_values()[0] = 5.0
    with pytest.raises(ValueError):
        mapper.bounds()[0, 0] = -5.0


def test_supports_no_varying_parameters() -> None:
    mapper = ParameterMapper(
        (
            Parameter("a", initial_value=2.0, vary=False),
            Parameter("b", initial_value=3.0, vary=False),
        )
    )

    assert mapper.varying_indices == ()
    assert mapper.fixed_indices == (0, 1)
    assert mapper.initial_values().shape == (0,)
    assert mapper.bounds().shape == (2, 0)
    np.testing.assert_array_equal(
        mapper.expand_values(np.array([])),
        [2.0, 3.0],
    )


def test_supports_empty_parameter_collection() -> None:
    mapper = ParameterMapper(())

    assert mapper.varying_indices == ()
    assert mapper.fixed_indices == ()
    assert mapper.bounds().shape == (2, 0)
    assert mapper.expand_values(np.array([])).shape == (0,)


def test_rejects_non_parameter_items() -> None:
    with pytest.raises(ValueError, match="only Parameter objects"):
        ParameterMapper((Parameter("a"), object()))


@pytest.mark.parametrize(
    "values",
    [
        np.array([[1.0, 2.0]]),
        np.array([1.0]),
        np.array([1.0, 2.0, 3.0]),
    ],
)
def test_rejects_invalid_solver_vector_shape_or_size(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="one-dimensional|size must match"):
        ParameterMapper(_make_parameters()).expand_values(values)
