# Predefined Model Catalog Architecture

## Status and scope

This document records the approved architecture for built-in model definitions
and supplements
`Curve_Fitting_Application_Architecture_Baseline_v1.5.pdf`.

The baseline already requires the user to select a built-in model, allows the
user to edit or copy a model, and places model-domain dependencies in
`curvefit.models` (acceptance criteria, PDF page 19; allowed dependencies,
PDF page 18). This addendum specifies how the catalog and editing workflow
preserve those requirements.

The catalog contains application-shipped definitions for explicit regression
models of the form `y = f(x, parameters...)`. It does not implement fitting,
LaTeX parsing, or persistence of user-created models.

## Package responsibilities

`src/curvefit/models/` owns the built-in model catalog and the factories that
create its model definitions.

- Each built-in model has a stable catalog key independent of its display name
  and generated `ModelDefinition.model_id`.
- A catalog lookup creates and returns a fresh `ModelDefinition`; it never
  returns a shared mutable model instance.
- Each returned definition owns fresh `Variable` and `Parameter` objects.
  Immutable symbolic expressions and numerical functions may be shared between
  instances.
- A factory creates the symbolic expression and numerical callable together
  so that they describe the same model.
- Built-in numerical callables accept an `(n, 1)` NumPy input matrix and an
  ordered one-dimensional parameter vector, and return an `(n, 1)` prediction
  matrix.
- The package may depend on the domain model types, SymPy, and NumPy for
  vectorized numerical callables. It must not depend on GUI classes, fitting
  engines, or concrete persistence and data loading implementations.

The catalog exposes model metadata needed to list and select built-in models,
such as the stable key and user-facing name. It does not expose mutable
catalog-owned `ModelDefinition` instances.

## Selection and editing workflow

The GUI requests an available built-in model through an application service.
The service uses the catalog to obtain a fresh definition and provides it to
the editing workflow.

Editing a selected model never changes the catalog entry or another open
model. Name and parameter configuration changes are made on the working
definition or on an independent copy. The `ModelDefinition.copy()` contract
creates a new model ID and independent `Variable` and `Parameter` objects.

Changing the equation is a model-construction operation, not an in-place
mutation of `ModelDefinition`. The Model Engine processes the edited equation
through the validated symbolic and numerical pipeline, then creates a new
definition. GUI callbacks do not parse equations or implement model rules.

## User-created model persistence

Built-in definitions remain application code and are not written to user
storage. Persistence of user-created models is a separate future capability
owned by an application service and a persistence adapter. It must not mutate
or replace catalog entries. The storage format and lifecycle are outside this
catalog feature's scope.

## Initial model families

The initial catalog may provide linear, polynomial, exponential, and
trigonometric model definitions. They remain explicit-regression models and
must follow the same factory and fresh-instance contract. Adding implicit
models or unrelated symbolic capabilities requires a separate architectural
decision.
