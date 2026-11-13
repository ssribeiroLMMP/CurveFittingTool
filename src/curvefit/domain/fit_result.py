"""Immutable historical record of one model fitting execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

from curvefit.domain.data_series import DataSeries
from curvefit.domain.model_curve import ModelCurve, _copy_data_series
from curvefit.domain.parameter_result import ParameterResult
from curvefit.domain.snapshots import ModelSnapshot
from curvefit.domain.validation import NumericValidator


class FitStatus(str, Enum):
    """Describe the final status of a fitting execution."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FitResult:
    """Preserve all outputs and model configuration from one fitting run."""

    _fit_id: UUID
    _model_snapshot: ModelSnapshot
    _parameter_results: tuple[ParameterResult, ...]
    _predicted_series: DataSeries = field(repr=False)
    _model_curve: ModelCurve
    _residuals: NDArray[np.float64] = field(repr=False)
    _covariance: NDArray[np.float64] | None = field(repr=False)
    _metrics: Mapping[str, float]
    _status: FitStatus
    _message: str
    _metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        model_snapshot: ModelSnapshot,
        parameter_results: Sequence[ParameterResult],
        predicted_series: DataSeries,
        model_curve: ModelCurve,
        residuals: NDArray[np.number],
        covariance: NDArray[np.number] | None,
        metrics: Mapping[str, Real],
        status: FitStatus | str,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
        fit_id: UUID | None = None,
    ) -> None:
        validated_fit_id = self._validated_fit_id(fit_id)
        validated_snapshot = self._validated_snapshot(model_snapshot)
        validated_results = self._validated_parameter_results(
            parameter_results,
            validated_snapshot,
        )
        validated_predictions = self._validated_series(
            predicted_series,
            validated_snapshot,
        )
        validated_curve = self._validated_curve(
            model_curve,
            validated_snapshot,
        )
        validated_residuals = self._validated_residuals(
            residuals,
            validated_predictions,
        )
        validated_covariance = self._validated_covariance(
            covariance,
            validated_snapshot,
        )
        validated_metrics = self._validated_metrics(metrics)
        validated_status = self._validated_status(status)
        validated_message = self._validated_message(message)
        validated_metadata = self._validated_metadata(metadata)

        object.__setattr__(self, "_fit_id", validated_fit_id)
        object.__setattr__(self, "_model_snapshot", validated_snapshot)
        object.__setattr__(self, "_parameter_results", validated_results)
        object.__setattr__(self, "_predicted_series", validated_predictions)
        object.__setattr__(self, "_model_curve", validated_curve)
        object.__setattr__(self, "_residuals", validated_residuals)
        object.__setattr__(self, "_covariance", validated_covariance)
        object.__setattr__(self, "_metrics", validated_metrics)
        object.__setattr__(self, "_status", validated_status)
        object.__setattr__(self, "_message", validated_message)
        object.__setattr__(self, "_metadata", validated_metadata)

    @property
    def fit_id(self) -> UUID:
        """Return the unique identifier of this fitting execution."""
        return self._fit_id

    @property
    def model_snapshot(self) -> ModelSnapshot:
        """Return the immutable model configuration used by the execution."""
        return self._model_snapshot

    @property
    def parameter_results(self) -> tuple[ParameterResult, ...]:
        """Return numerical results in model parameter order."""
        return self._parameter_results

    @property
    def predicted_series(self) -> DataSeries:
        """Return independent predictions at the experimental inputs."""
        return _copy_data_series(self._predicted_series)

    @property
    def model_curve(self) -> ModelCurve:
        """Return the immutable presentation curve."""
        return self._model_curve

    @property
    def residuals(self) -> NDArray[np.float64]:
        """Return the read-only residual matrix."""
        return self._residuals

    @property
    def covariance(self) -> NDArray[np.float64] | None:
        """Return the read-only covariance matrix when available."""
        return self._covariance

    @property
    def metrics(self) -> Mapping[str, float]:
        """Return immutable fitting metrics."""
        return self._metrics

    @property
    def status(self) -> FitStatus:
        """Return the final execution status."""
        return self._status

    @property
    def success(self) -> bool:
        """Return whether the fitting execution completed successfully."""
        return self._status is FitStatus.SUCCESS

    @property
    def message(self) -> str:
        """Return the solver or application status message."""
        return self._message

    @property
    def metadata(self) -> dict[str, Any]:
        """Return an independent copy of execution metadata."""
        return deepcopy(dict(self._metadata))

    @staticmethod
    def _validated_fit_id(value: UUID | None) -> UUID:
        if value is None:
            return uuid4()
        if not isinstance(value, UUID):
            raise ValueError("fit_id must be a UUID or None.")
        return value

    @staticmethod
    def _validated_snapshot(value: ModelSnapshot) -> ModelSnapshot:
        if not isinstance(value, ModelSnapshot):
            raise ValueError("model_snapshot must be a ModelSnapshot object.")
        return value

    @staticmethod
    def _validated_parameter_results(
        values: Sequence[ParameterResult],
        snapshot: ModelSnapshot,
    ) -> tuple[ParameterResult, ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(
                "parameter_results must be a sequence of ParameterResult objects."
            )
        normalized = tuple(values)
        if not all(isinstance(value, ParameterResult) for value in normalized):
            raise ValueError(
                "parameter_results must contain only ParameterResult objects."
            )

        result_symbols = tuple(value.symbol for value in normalized)
        snapshot_symbols = tuple(
            value.symbol
            for value in snapshot.parameters
        )
        if result_symbols != snapshot_symbols:
            raise ValueError(
                "parameter_results must match model snapshot parameter order."
            )
        return normalized

    @staticmethod
    def _validated_series(
        value: DataSeries,
        snapshot: ModelSnapshot,
    ) -> DataSeries:
        if not isinstance(value, DataSeries):
            raise ValueError("predicted_series must be a DataSeries object.")
        FitResult._validate_series_symbols(value, snapshot, "predicted_series")
        return _copy_data_series(value)

    @staticmethod
    def _validated_curve(
        value: ModelCurve,
        snapshot: ModelSnapshot,
    ) -> ModelCurve:
        if not isinstance(value, ModelCurve):
            raise ValueError("model_curve must be a ModelCurve object.")
        FitResult._validate_series_symbols(
            value.data_series,
            snapshot,
            "model_curve",
        )
        return value

    @staticmethod
    def _validate_series_symbols(
        series: DataSeries,
        snapshot: ModelSnapshot,
        field_name: str,
    ) -> None:
        independent_symbols = tuple(
            variable.symbol
            for variable in series.independent_variables
        )
        dependent_symbols = tuple(
            variable.symbol
            for variable in series.dependent_variables
        )
        if independent_symbols != tuple(
            variable.symbol
            for variable in snapshot.independent_variables
        ) or dependent_symbols != tuple(
            variable.symbol
            for variable in snapshot.dependent_variables
        ):
            raise ValueError(
                f"{field_name} variable symbols must match model_snapshot."
            )

    @staticmethod
    def _validated_residuals(
        values: NDArray[np.number],
        predicted_series: DataSeries,
    ) -> NDArray[np.float64]:
        normalized = FitResult._validated_array(values, "residuals")
        if normalized.shape != predicted_series.dependent_data.shape:
            raise ValueError(
                "residuals must match predicted_series dependent-data shape."
            )
        return normalized

    @staticmethod
    def _validated_covariance(
        values: NDArray[np.number] | None,
        snapshot: ModelSnapshot,
    ) -> NDArray[np.float64] | None:
        if values is None:
            return None
        normalized = FitResult._validated_array(values, "covariance")
        varying_count = sum(
            parameter.vary
            for parameter in snapshot.parameters
        )
        if normalized.shape != (varying_count, varying_count):
            raise ValueError(
                "covariance must be square with one row and column for each "
                "varying parameter."
            )
        return normalized

    @staticmethod
    def _validated_array(
        values: NDArray[np.number],
        field_name: str,
    ) -> NDArray[np.float64]:
        if not isinstance(values, np.ndarray):
            raise ValueError(f"{field_name} must be a NumPy array.")
        if values.ndim != 2:
            raise ValueError(f"{field_name} must be two-dimensional.")
        if (
            not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            raise ValueError(f"{field_name} must contain numeric values.")

        normalized = values.astype(np.float64, copy=True)
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"{field_name} must contain only finite values.")
        normalized.setflags(write=False)
        return normalized

    @staticmethod
    def _validated_metrics(values: Mapping[str, Real]) -> Mapping[str, float]:
        if not isinstance(values, Mapping):
            raise ValueError("metrics must be a mapping.")

        normalized: dict[str, float] = {}
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("metric names must be non-empty strings.")
            normalized_name = name.strip()
            if normalized_name in normalized:
                raise ValueError("metric names must be unique after normalization.")
            normalized[normalized_name] = NumericValidator.finite(
                value,
                field_name=f"metric {name!r}",
            )
        return MappingProxyType(normalized)

    @staticmethod
    def _validated_status(value: FitStatus | str) -> FitStatus:
        try:
            return FitStatus(value)
        except (TypeError, ValueError):
            raise ValueError(
                "status must be FitStatus.SUCCESS or FitStatus.FAILURE."
            ) from None

    @staticmethod
    def _validated_message(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("message must be a string.")
        return value.strip()

    @staticmethod
    def _validated_metadata(
        values: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if values is None:
            return MappingProxyType({})
        if not isinstance(values, Mapping):
            raise ValueError("metadata must be a mapping or None.")
        return MappingProxyType(deepcopy(dict(values)))
