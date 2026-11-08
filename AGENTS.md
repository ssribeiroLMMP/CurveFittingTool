# Curve Fitting Application — Codex Development Instructions

This repository implements a desktop scientific curve-fitting application with a Tkinter GUI and a Python object-oriented backend.

## 1. Normative architecture baseline

The primary architectural reference for this repository is:

`docs/Curve_Fitting_Application_Architecture_Baseline_v1.5.pdf`

Version 1.5 supersedes the earlier baseline, including the former Parameter fit-state and Dataset construction contracts. Historical documents under `docs/old/` are not normative.

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

- `src/curvefit/domain/parameter/parameter.py` -> `tests/unit/domain/parameter/test_parameter.py`
- `src/curvefit/domain/data_series/data_series.py` -> `tests/unit/domain/data_series/test_data_series.py`
- `src/curvefit/domain/dataset/dataset.py` -> `tests/unit/domain/dataset/test_dataset.py`
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

NumPy and SymPy are permitted in domain objects only where explicitly provided for by the v1.5 contracts. DataSeries owns canonical 2D float64 arrays; Dataset composes DataSeries and adds experimental semantics and metadata. Pandas must remain in the data layer.

ModelDefinition stores symbolic_expressions, a numerical callable Y = F(X, theta), and an immutable model_id. LaTeX belongs to the input pipeline rather than mandatory ModelDefinition state. copy() creates a new model_id. ModelDefinition must not depend on fitting.ParameterMapper.

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

- Parameter contains only input configuration; it must not store fitted value or stderr, or expose fit-state reset methods. ParameterValidator does not validate fit_value or stderr.
- FitEngine is an ABC with the contract fit(dataset: Dataset, model: ModelDefinition) -> FitResult. Parameters belong to ModelDefinition and are not passed separately.
- ParameterMapper belongs to fitting and captures the current parameter configuration at the start of each fit. It must not write results into Parameter.
- ParameterResult stores numerical parameter results. Immutable ModelSnapshot, ParameterSnapshot, and VariableSnapshot preserve the configuration used in each execution without live mutable references.
- FitResult is an immutable historical execution with its own fit_id, model_snapshot, parameter_results, predicted_series, model_curve, residuals, covariance, metrics, status, and metadata. It has no partial clear operation.
- ModelCurve composes DataSeries for presentation; predicted_series contains predictions at experimental inputs. Multiple FitResult objects may coexist for one Dataset.

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

1. domain validators, `domain/variable/`, and `domain/parameter/`
2. `domain/data_series/` and `domain/dataset/`
3. `domain/model_definition/`
4. `domain/snapshots/`, `domain/parameter_result.py`, `domain/model_curve.py`, `domain/fit_result.py`, and `domain/exceptions.py`
5. Model Engine: LaTeX parsing, ModelSymbolAnalyzer, parameter identification, NumericFunctionBuilder, and model registry
6. fitting: FitEngine, ParameterMapper, LmfitEngine, statistics, and synthetic-data tests
7. data ingestion modules
8. application services
9. plotting and structural Tkinter GUI
10. end-to-end integration and comparison of multiple FitResult objects
11. snapshot persistence, integrity validation, and polishing

When implementing any item, create its corresponding tests in the mirrored `tests/unit/` directory at the same time.

## 13. Git commit subjects

Write commit subjects in English. Prefix each subject with exactly one of these tags, in brackets:

- `[add]` — add a feature, component, code, or comment.
- `[update]` — change or extend an existing feature or component.
- `[fix]` — fix a bug or address an issue or review suggestion.
- `[comment]` — add or revise comments or docstrings in existing code.
- `[clean]` — remove unused code, comments, or files.

Use the matching subject format:

- `[add] <feature, component, or concise description of new code>`
- `[update] <feature or component>: <concise description of the change>`
- `[fix] <feature or component>: <issue, bug, or review suggestion>`
- `[comment] <feature or component>: <concise description of the comment change>`
- `[clean] <feature or component>`

Keep subjects short and specific. Choose the tag according to the change's primary purpose.

## 14. Git branch names

Start each branch name with exactly one of these prefixes, followed by a concise name using the `/` separator:

- `feature/` — develop a new feature.
- `update/` — change or extend an existing feature or specification.
- `fix/` — fix a bug or address a review suggestion.
- `doc/` — document code or update README files.
- `clean/` — remove unused code, comments, or files.
- `test/` — add or update automated unit or integration tests.
- `temp/` — perform temporary studies or scripts that are usually not merged.

Use a specific, concise branch name after the prefix, for example `test/parameter-contract`.
