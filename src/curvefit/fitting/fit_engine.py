"""Abstract fitting engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from curvefit.domain.dataset import Dataset
from curvefit.domain.fit_result import FitResult
from curvefit.domain.model_definition import ModelDefinition


class FitEngine(ABC):
    """Define the contract for numerical fitting engines."""

    @abstractmethod
    def fit(
        self,
        dataset: Dataset,
        model: ModelDefinition,
    ) -> FitResult:
        """Fit a mathematical model to a dataset.

        Parameters
        ----------
        dataset:
            Experimental dataset to be fitted.
        model:
            Mathematical model definition and parameter configuration.

        Returns
        -------
        FitResult
            Result of the numerical fitting process.

        Raises
        ------
        FitError
            If the fitting process cannot be completed successfully.
        """
        raise NotImplementedError
