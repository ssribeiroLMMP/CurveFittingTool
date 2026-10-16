"""Validation rules specific to Dataset entities."""

from __future__ import annotations

from curvefit.domain.data_series import DataSeries


class DatasetValidator:
    """Validate Dataset-specific constraints."""

    @staticmethod
    def data_series(
        value: DataSeries,
        *,
        expected_independent_count: int | None = None,
        expected_dependent_count: int | None = None,
    ) -> DataSeries:
        """Validate a DataSeries and optional Dataset dimensionality limits."""
        if not isinstance(value, DataSeries):
            raise ValueError("data_series must be a DataSeries.")

        if (
            expected_independent_count is not None
            and value.independent_variable_count != expected_independent_count
        ):
            raise ValueError(
                "data series has an unsupported number of independent "
                "variables."
            )

        if (
            expected_dependent_count is not None
            and value.dependent_variable_count != expected_dependent_count
        ):
            raise ValueError(
                "data series has an unsupported number of dependent variables."
            )

        return value
