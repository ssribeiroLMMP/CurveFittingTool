"""Concrete FitEngine implementation backed by LMFIT."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import lmfit
import numpy as np
from numpy.typing import NDArray

from curvefit.domain.data_series import DataSeries
from curvefit.domain.dataset import Dataset
from curvefit.domain.exceptions import FitError
from curvefit.domain.fit_result import FitResult, FitStatus
from curvefit.domain.model_curve import ModelCurve
from curvefit.domain.model_definition import ModelDefinition
from curvefit.domain.parameter_result import ParameterResult
from curvefit.domain.snapshots import ModelSnapshot
from curvefit.fitting.fit_engine import FitEngine
from curvefit.fitting.parameter_mapper import ParameterMapper
from curvefit.fitting.statistics import (
    calculate_fit_metrics,
    calculate_residuals,
)


class LmfitEngine(FitEngine):
    """Fit explicit regression models through LMFIT least squares."""

    DEFAULT_CURVE_POINT_COUNT = 200
    _METHOD = "least_squares"

    def __init__(
        self,
        *,
        curve_point_count: int = DEFAULT_CURVE_POINT_COUNT,
    ) -> None:
        if (
            isinstance(curve_point_count, bool)
            or not isinstance(curve_point_count, int)
            or curve_point_count < 2
        ):
            raise ValueError("curve_point_count must be an integer of at least 2.")
        self._curve_point_count = curve_point_count

    @property
    def curve_point_count(self) -> int:
        """Return the number of samples used for the presentation curve."""
        return self._curve_point_count

    def fit(
        self,
        dataset: Dataset,
        model: ModelDefinition,
    ) -> FitResult:
        """Fit a model to a dataset and return an immutable execution result."""
        self._validate_inputs(dataset, model)
        snapshot = ModelSnapshot.from_model(model)
        mapper = ParameterMapper(model.parameters)

        try:
            if mapper.varying_indices:
                return self._fit_varying_parameters(
                    dataset,
                    model,
                    snapshot,
                    mapper,
                )
            return self._evaluate_fixed_parameters(
                dataset,
                model,
                snapshot,
                mapper,
            )
        except FitError:
            raise
        except Exception as error:
            raise FitError(f"fitting failed: {error}") from error

    def _fit_varying_parameters(
        self,
        dataset: Dataset,
        model: ModelDefinition,
        snapshot: ModelSnapshot,
        mapper: ParameterMapper,
    ) -> FitResult:
        solver_parameters = self._build_solver_parameters(mapper)
        solver_names = tuple(solver_parameters.keys())

        def objective(parameters: lmfit.Parameters) -> NDArray[np.float64]:
            varying_values = np.asarray(
                [parameters[name].value for name in solver_names],
                dtype=np.float64,
            )
            full_values = mapper.expand_values(varying_values)
            predictions = self._evaluate_model(
                model,
                dataset.independent_data,
                full_values,
                expected_shape=dataset.dependent_data.shape,
            )
            return calculate_residuals(
                dataset.dependent_data,
                predictions,
            ).reshape(-1)

        result = lmfit.minimize(
            objective,
            solver_parameters,
            method=self._METHOD,
            nan_policy="raise",
        )
        if not bool(result.success):
            raise FitError(
                f"LMFIT did not converge: {result.message or 'unknown failure'}."
            )

        varying_values = np.asarray(
            [result.params[name].value for name in solver_names],
            dtype=np.float64,
        )
        full_values = mapper.expand_values(varying_values)
        standard_errors = {
            parameter_index: self._normalized_standard_error(
                result.params[solver_name].stderr
            )
            for parameter_index, solver_name in zip(
                mapper.varying_indices,
                solver_names,
                strict=True,
            )
        }
        covariance = self._normalized_covariance(result.covar)
        metadata = self._solver_metadata(result)

        return self._build_result(
            dataset=dataset,
            model=model,
            snapshot=snapshot,
            full_values=full_values,
            standard_errors=standard_errors,
            covariance=covariance,
            message=str(result.message or "Fit completed."),
            metadata=metadata,
        )

    def _evaluate_fixed_parameters(
        self,
        dataset: Dataset,
        model: ModelDefinition,
        snapshot: ModelSnapshot,
        mapper: ParameterMapper,
    ) -> FitResult:
        full_values = mapper.expand_values(np.empty(0, dtype=np.float64))
        return self._build_result(
            dataset=dataset,
            model=model,
            snapshot=snapshot,
            full_values=full_values,
            standard_errors={},
            covariance=None,
            message="No varying parameters; evaluated configured values.",
            metadata={
                "engine": "lmfit",
                "method": "fixed_parameters",
                "function_evaluations": 1,
            },
        )

    def _build_result(
        self,
        *,
        dataset: Dataset,
        model: ModelDefinition,
        snapshot: ModelSnapshot,
        full_values: NDArray[np.float64],
        standard_errors: Mapping[int, Real | None],
        covariance: NDArray[np.float64] | None,
        message: str,
        metadata: Mapping[str, Any],
    ) -> FitResult:
        predictions = self._evaluate_model(
            model,
            dataset.independent_data,
            full_values,
            expected_shape=dataset.dependent_data.shape,
        )
        predicted_series = self._build_series(
            model,
            dataset.independent_data,
            predictions,
        )
        residuals = calculate_residuals(dataset.dependent_data, predictions)
        metrics = calculate_fit_metrics(dataset.dependent_data, predictions)
        model_curve = self._build_model_curve(model, dataset, full_values)
        parameter_results = tuple(
            ParameterResult(
                symbol=parameter.symbol,
                value=full_values[index],
                stderr=standard_errors.get(index),
            )
            for index, parameter in enumerate(model.parameters)
        )

        return FitResult(
            model_snapshot=snapshot,
            parameter_results=parameter_results,
            predicted_series=predicted_series,
            model_curve=model_curve,
            residuals=residuals,
            covariance=covariance,
            metrics=metrics,
            status=FitStatus.SUCCESS,
            message=message,
            metadata=metadata,
        )

    def _build_model_curve(
        self,
        model: ModelDefinition,
        dataset: Dataset,
        full_values: NDArray[np.float64],
    ) -> ModelCurve:
        independent_values = dataset.independent_data[:, 0]
        curve_inputs = np.linspace(
            float(np.min(independent_values)),
            float(np.max(independent_values)),
            self._curve_point_count,
            dtype=np.float64,
        ).reshape(-1, 1)
        curve_predictions = self._evaluate_model(
            model,
            curve_inputs,
            full_values,
            expected_shape=(self._curve_point_count, 1),
        )
        return ModelCurve(
            self._build_series(
                model,
                curve_inputs,
                curve_predictions,
            )
        )

    @staticmethod
    def _build_series(
        model: ModelDefinition,
        independent_data: NDArray[np.float64],
        dependent_data: NDArray[np.float64],
    ) -> DataSeries:
        return DataSeries(
            independent_variables=tuple(
                variable.copy()
                for variable in model.independent_variables
            ),
            dependent_variables=tuple(
                variable.copy()
                for variable in model.dependent_variables
            ),
            independent_data=independent_data,
            dependent_data=dependent_data,
        )

    @staticmethod
    def _build_solver_parameters(
        mapper: ParameterMapper,
    ) -> lmfit.Parameters:
        parameters = lmfit.Parameters()
        initial_values = mapper.initial_values()
        bounds = mapper.bounds()
        for index, initial_value in enumerate(initial_values):
            parameters.add(
                f"parameter_{index}",
                value=initial_value,
                min=bounds[0, index],
                max=bounds[1, index],
                vary=True,
            )
        return parameters

    @staticmethod
    def _evaluate_model(
        model: ModelDefinition,
        independent_data: NDArray[np.float64],
        parameter_values: NDArray[np.float64],
        *,
        expected_shape: tuple[int, int],
    ) -> NDArray[np.float64]:
        try:
            predictions = np.asarray(
                model.numeric_function(independent_data, parameter_values),
                dtype=np.float64,
            )
        except Exception as error:
            raise FitError(f"model evaluation failed: {error}") from error

        if predictions.shape != expected_shape:
            raise FitError(
                "model evaluation returned shape "
                f"{predictions.shape}, expected {expected_shape}."
            )
        if not np.all(np.isfinite(predictions)):
            raise FitError("model evaluation returned non-finite predictions.")
        return predictions

    @staticmethod
    def _normalized_covariance(
        covariance: NDArray[np.number] | None,
    ) -> NDArray[np.float64] | None:
        if covariance is None:
            return None
        normalized = np.asarray(covariance, dtype=np.float64)
        if normalized.ndim != 2 or not np.all(np.isfinite(normalized)):
            return None
        return normalized

    @staticmethod
    def _normalized_standard_error(value: Real | None) -> float | None:
        if value is None or isinstance(value, bool) or not isinstance(value, Real):
            return None
        normalized = float(value)
        if not np.isfinite(normalized) or normalized < 0.0:
            return None
        return normalized

    @staticmethod
    def _solver_metadata(result: Any) -> dict[str, Any]:
        return {
            "engine": "lmfit",
            "method": str(getattr(result, "method", LmfitEngine._METHOD)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "chi_square": float(getattr(result, "chisqr", np.nan)),
            "reduced_chi_square": float(getattr(result, "redchi", np.nan)),
        }

    @staticmethod
    def _validate_inputs(dataset: Dataset, model: ModelDefinition) -> None:
        if not isinstance(dataset, Dataset):
            raise FitError("dataset must be a Dataset object.")
        if not isinstance(model, ModelDefinition):
            raise FitError("model must be a ModelDefinition object.")

        dataset_independent_symbols = tuple(
            variable.symbol
            for variable in dataset.independent_variables
        )
        model_independent_symbols = tuple(
            variable.symbol
            for variable in model.independent_variables
        )
        dataset_dependent_symbols = tuple(
            variable.symbol
            for variable in dataset.dependent_variables
        )
        model_dependent_symbols = tuple(
            variable.symbol
            for variable in model.dependent_variables
        )
        if (
            dataset_independent_symbols != model_independent_symbols
            or dataset_dependent_symbols != model_dependent_symbols
        ):
            raise FitError("dataset and model variable symbols must match.")
