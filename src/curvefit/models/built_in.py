"""Default catalog of application-shipped model definitions."""

from curvefit.models.catalog import ModelCatalog
from curvefit.models.exponential import (
    EXPONENTIAL_MODEL_METADATA,
    create_exponential_model,
)
from curvefit.models.linear import (
    LINEAR_MODEL_METADATA,
    create_linear_model,
)
from curvefit.models.polynomial import (
    create_polynomial_model,
    polynomial_model_metadata,
)
from curvefit.models.trigonometric import (
    COSINE_MODEL_METADATA,
    SINE_MODEL_METADATA,
    create_cosine_model,
    create_sine_model,
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
        (EXPONENTIAL_MODEL_METADATA, create_exponential_model),
        (SINE_MODEL_METADATA, create_sine_model),
        (COSINE_MODEL_METADATA, create_cosine_model),
    )
)
