# Curve Fitting Application — Codex Development Instructions

This repository implements a desktop scientific curve-fitting application with a Tkinter GUI and a Python object-oriented backend.

## 1. Normative architecture baseline

The primary architectural reference for this repository is:

`docs/Curve_Fitting_Application_Architecture_Baseline_v1.4.pdf`

Before implementing or modifying any module, read the relevant sections of that document and preserve the defined package boundaries, responsibilities, and dependency directions.

The architecture baseline is normative but evolutionary. If a technically justified implementation would materially diverge from the baseline:

1. Do not silently introduce the divergence.
2. Clearly identify what would diverge.
3. Explain the technical reason and expected impact.
4. Recommend whether the code or the architecture document should change.
5. Stop short of treating the divergent design as final until the decision is explicitly made.

## 2. Development strategy

Develop the system incrementally, module by module.

For every production module implemented:

- create or update its corresponding unit tests;
- keep tests close in conceptual scope to the module under test;
- execute the relevant tests before considering the task complete;
- when practical, execute the complete test suite as a regression check;
- report which tests were executed and whether they passed;
- never hide, suppress, or silently ignore failing tests.

Use `pytest` for automated testing unless the architecture baseline is explicitly revised.

## 3. Source and test layout

Production packages live under:

`src/curvefit/`

Unit tests mirror the production package structure under:

`tests/unit/`

Examples:

- `src/curvefit/domain/parameter.py` -> `tests/unit/domain/test_parameter.py`
- `src/curvefit/data/csv_loader.py` -> `tests/unit/data/test_csv_loader.py`
- `src/curvefit/models/latex_parser.py` -> `tests/unit/models/test_latex_parser.py`
- `src/curvefit/fitting/lmfit_engine.py` -> `tests/unit/fitting/test_lmfit_engine.py`

Cross-module behavior belongs under:

`tests/integration/`

Reusable non-sensitive test datasets or fixtures belong under:

`tests/fixtures/`

Do not place real, confidential, customer, proprietary, or personal datasets in the repository.

## 4. Dependency direction

The intended dependency direction is generally:

`gui -> services -> domain/models/data/fitting -> external scientific libraries`

The domain layer must remain independent from GUI and concrete fitting frameworks.

### Domain package restrictions

Modules under `src/curvefit/domain/` must not depend on:

- Tkinter
- Matplotlib
- Pandas
- SciPy
- LMFIT
- GUI classes
- concrete file loaders

NumPy should only be introduced into a domain object where the architecture explicitly benefits from array semantics and the dependency has been accepted for that object.

### GUI restrictions

Tkinter callbacks must not contain scientific fitting logic, LaTeX parsing logic, or data-processing business rules. The GUI should delegate those responsibilities to controllers/services.

## 5. Model parsing and arbitrary equations

User-entered mathematical models are supplied as LaTeX and must follow the architecture pipeline:

`LaTeX -> validated symbolic representation -> parameter identification -> numerical callable -> fitting engine`

Do not use Python `eval()` on user input.

Treat the LaTeX parser as an isolated adapter so that parser technology can be replaced without affecting the rest of the system.

The initial model scope is explicit regression of the form:

`y = f(x, parameters...)`

Do not silently add implicit-model fitting or unrelated symbolic capabilities without an architectural decision.

## 6. Fitting engine

The fitting engine must be accessed through an abstraction rather than directly from GUI code.

The initial concrete implementation is expected to use LMFIT/SciPy according to the baseline, but domain objects must not depend on LMFIT-specific parameter classes.

Initial guesses, bounds, fixed/varying parameters, uncertainties, residuals, and fit statistics must remain explicit concepts in the application model.

## 7. Security and sensitive data

Never commit or generate secrets inside version-controlled files.

Do not commit:

- API keys
- passwords
- credentials
- access tokens
- private keys or certificates
- `.env` files containing real values
- local IDE secrets
- user datasets containing confidential or personal information
- generated exports that may contain sensitive data

Use environment variables for secrets when external services are eventually introduced.

If an example environment file is needed, create `.env.example` using placeholders only.

## 8. Generated files

Do not treat generated artifacts as source code unless explicitly requested.

Generated test reports, coverage reports, caches, temporary exports, local databases, logs, plots, compiled Python files, virtual environments, and build outputs must remain unversioned according to `.gitignore`.

## 9. Coding expectations

Prefer:

- small, cohesive classes;
- explicit interfaces;
- type hints;
- dataclasses where appropriate for domain data structures;
- deterministic behavior in tests;
- clear exceptions instead of silent failure;
- dependency injection where it improves testability;
- pure functions for transformations where practical.

Avoid:

- global mutable state;
- hidden dependencies;
- circular imports;
- scientific logic embedded in event handlers;
- oversized modules combining unrelated responsibilities;
- premature coupling to a particular GUI widget implementation.

## 10. Testing expectations

Every new public behavior should have a corresponding test.

At minimum, unit tests should cover:

- normal behavior;
- invalid inputs;
- boundary conditions;
- expected exceptions;
- state reset/update behavior where relevant.

Numerical tests must use appropriate tolerances rather than exact floating-point equality when necessary.

Use synthetic, deterministic, non-sensitive data in tests.

## 11. Completion criteria for a module

A module is not considered complete until:

1. Its responsibility matches the architecture baseline.
2. Public interfaces are clear and typed where practical.
3. Unit tests exist for its behavior.
4. Relevant tests pass.
5. No architectural boundary has been violated.
6. Any discovered architectural ambiguity or divergence has been reported.
7. No sensitive or generated local data has been added to Git.

## 12. Initial implementation order

Unless a later architectural decision changes the sequence, prefer this order:

1. `domain/parameter.py`
2. `domain/dataset.py`
3. `domain/fit_result.py`
4. `domain/model.py`
5. `domain/exceptions.py`
6. data ingestion modules
7. model registry and custom/LaTeX model parsing
8. fitting engine
9. application services
10. plotting
11. Tkinter GUI
12. end-to-end integration

When implementing any item, create its corresponding tests in the mirrored `tests/unit/` directory at the same time.
