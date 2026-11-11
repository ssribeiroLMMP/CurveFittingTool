"""Default catalog of application-shipped model definitions."""

from curvefit.models.catalog import ModelCatalog
from curvefit.models.linear import (
    LINEAR_MODEL_METADATA,
    create_linear_model,
)
from curvefit.models.polynomial import (
    create_polynomial_model,
    polynomial_model_metadata,
)

_POLYNOMIAL_DEGREES = (2, 3)

BUILTIN_MODEL_CATALOG = ModelCatalog(
    (
        (LINEAR_MODEL_METADATA, create_linear_model),
        *(
            (
                polynomial_model_metadata(degree),
                lambda degree=degree: create_polynomial_model(degree),
            )
            for degree in _POLYNOMIAL_DEGREES
        ),
    )
)
