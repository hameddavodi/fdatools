# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.

## [Unreleased]

### Added
- `fdatools.interop`, bridges to the wider Python stack:
  - NumPy: `from_grid` / `to_grid` in either the fdatools `(n_points, n_curves)`
    or the scikit-learn `(n_curves, n_points)` layout.
  - SciPy: `to_scipy` / `from_scipy` convert spline curves to and from
    `scipy.interpolate.BSpline`, exactly (same knots and coefficients).
  - scikit-fda: `to_skfda` / `from_skfda`, exact for B-spline, polygonal,
    Fourier, monomial and constant bases, through a grid otherwise.
  - Panels: `from_panel` / `to_panel` for `(entity, time)` MultiIndex data, the
    layout of `linearmodels` and `statsmodels`; unbalanced panels are smoothed
    per entity.
  - `functional_design` / `FunctionalDesign`: the design matrix of a
    scalar-on-function model for any linear estimator (statsmodels OLS, GLM,
    quantile regression; linearmodels PanelOLS, IV). `beta()` maps the
    parameters back to `beta(t)`, and `band()` builds a pointwise band from the
    estimator's covariance (classical, robust or clustered).
- `econ` extra (pandas, statsmodels, linearmodels, scikit-fda) and a new
  tutorial, "Econometrics and interop".
- fdatools is on PyPI: `uv add fdatools`. Releases are published by
  `.github/workflows/release.yml` with PyPI trusted publishing (no API token);
  it uploads the files attached to the GitHub release.

### Fixed
- The test suite no longer fails without PyTorch. The `tests/torch/` folder
  was imported as a fake `torch` namespace package, which broke
  array-api-compat's type checks (`module 'torch' has no attribute 'Tensor'`).
  It is renamed to `tests/torch_backend/`.

### Changed
- Install instructions use uv only (`uv add fdatools`, extras as
  `uv add "fdatools[plot]"`), also in the hints of missing-extra errors.
- The CI dependency audit uses the OSV database and skips the project itself,
  so it does not depend on pypi.org being up.

## [1.1.0] - 2026-09-28

First release under the name **fdatools** (the first one on PyPI).

### Changed
- The project is renamed from **fabel** to **fdatools**: `pip install fdatools`,
  `import fdatools as fdt`. The GitHub repository moves to
  `hameddavodi/fdatools` and the documentation to
  <https://hameddavodi.github.io/fdatools>. The environment variables
  `FABEL_DATA_DIR` / `FABEL_RUN_NETWORK_TESTS` become `FDATOOLS_DATA_DIR` /
  `FDATOOLS_RUN_NETWORK_TESTS`, and the dataset cache moves to
  `~/.cache/fdatools`. The 1.0.0 entry below describes the release published
  under the old name.
- `rpy2` moved from the `dev` extra to a new `golden` extra (only
  `tools/make_golden.py` uses it), so the development install no longer needs R.

### Fixed
- `PACE(sigma2=...)` no longer raises a `RuntimeWarning` about a non-positive
  measurement-error estimate: with `sigma2` given, the estimate is kept in
  `cov_estimate_` but not used, so it is not worth a warning.
- `FPCA`, `FCCA` and `PACE` reject a NumPy bool for `n` on every NumPy version
  (NumPy 2.2 only warned in `operator.index`).

### Added
- `notebooks/tour.ipynb`: a full tour that uses every public module on the
  bundled data sets, records 131 checks, and compares key results live with R
  `fda` (through `Rscript`). Built from `notebooks/tour.py` by
  `tools/build_tour_notebook.py`.

## [1.0.0] - 2026-09-27

First public release: a clean-room Python rewrite of R `fda` 6.3.0 with
golden-file parity (`rtol = 1e-8`, `1e-5` for iterative fits).

### Added
- Basis systems (`fabel.basis`): `BSpline`, `Fourier`, `Monomial`, `Exponential`,
  `Power`, `Constant`, `Polygonal`, with evaluation and derivatives, roughness
  penalties for any `LDO`, cached Gram matrices and exact basis products.
- Core objects (`fabel.core`): `FData` (callable curves, exact derivatives,
  arithmetic, `mean` / `std` / `center` / `cov`, indexing, `@` inner product),
  `BiFData`, `LDO` (including the harmonic accelerator) and `inprod`.
- Smoothing (`fabel.smoothing`): `smooth()` with GCV or degrees-of-freedom
  selection of λ, positive / monotone / morph constraints, observation weights
  and irregular per-curve designs; `SmoothResult`; the scikit-learn `Smoother`;
  `gcv_curve`, `lambda_to_df`, `df_to_lambda`.
- Decomposition (`fabel.decomposition`): `FPCA` (with roughness-penalised
  harmonics and varimax rotation) and `FCCA`, both scikit-learn estimators.
- Regression (`fabel.regression`): `fregress()` for scalar and functional
  responses with scalar and functional covariates (model type read from the
  arguments, or an R-style formula string with treatment-coded factors), with
  `predict()`, `stderr()` and `cv()`; the scikit-learn `FRegress` estimator.
- Registration (`fabel.registration`): continuous registration `register()`
  (Newton with the exact Hessian, optional periodic shift), landmark registration
  (`landmarks=` or `landmark_register()`), the amplitude/phase decomposition
  `RegistrationResult.decompose()` and the scikit-learn `Registrator`.
- Dynamics (`fabel.dynamics`): principal differential analysis `PDA` for single
  equations and coupled systems, with an ODE `solve()` and `plot_overlay()`, and
  the `phase_plane()` plot.
- Statistics (`fabel.stats`): `cov`, `cor`, functional depth (MBD, BD2, FM),
  the functional `boxplot`, and the permutation tests `t_test` and `f_test`.
  `f_test` takes either raw inputs `(y, x, basis=, lam=, penalty=)` like R
  `Fperm.fd`, or a fitted `fregress` model: `f_test(model, n_perm=, q=, t=,
  random_state=)`.
- scikit-learn: `check_estimator` passes for `Smoother`, `FPCA`, `FRegress` and
  `Registrator` with no exemptions. `FRegress` validates `y` as scikit-learn does
  (finite, column vectors flattened with a warning, at least 2 samples).
  `Registrator` has `n_iter_`, and its default basis for an n-column coefficient
  matrix is `BSpline(n_basis=n, order=min(4, n))`, the same rule as `FPCA`.
- PyTorch layers (`fabel.nn`, optional `fabel[torch]` extra): `BasisLayer`,
  `SmoothingLayer` (learnable λ) and `FDataDataset`. `import fabel` does not
  import PyTorch; `fabel.nn` loads on first use.
- Top-level exports: `smooth`, `Smoother`, `SmoothResult`, `FPCA`, `FCCA`,
  `fregress`, `FRegress`, `register`, `landmark_register`, `Registrator`, `PDA`,
  `phase_plane`, and the `stats` module.
- Datasets (`fabel.datasets`): 14 loaders for the FDA book datasets; `growth`,
  `gait` and `pinch` ship in the package, the others download once with SHA-256
  verification and a `FABEL_DATA_DIR`-overridable cache.
- I/O (`fabel.io`): `from_pandas`, `to_pandas`, `to_xarray` and `read_rds` for R
  `fd` / `bifd` / `basisfd` objects.
- Array API backend: NumPy and PyTorch inputs, with gradients flowing through
  evaluation, products and inner products.
- Golden-file parity suite against R `fda` 6.3.0, and `tools/parity_report.py`,
  which measures every parity check and writes `PARITY_REPORT.md`.
- Documentation site (MkDocs Material + mkdocstrings): quickstart, R migration
  table, API reference for every module, and six tutorials (smoothing, FPCA,
  registration, regression, dynamics, machine learning). Tests run every
  tutorial block and check that every public symbol is rendered.
- `notebooks/book_figures.ipynb`: figures of Ramsay, Hooker & Graves (2009),
  assembled from `notebooks/book/ch*.py` by `tools/build_book_notebook.py`
  (`make book`) and run in CI with `nbmake`.
- Packaging: typed (`py.typed`) wheel and sdist for Python 3.10 to 3.13,
  BSD-3-Clause license, `CITATION.cff`.
- `register()` accepts an `FData` with PyTorch coefficients and returns PyTorch
  results; gradients flow from the registered curves to the input coefficients
  (autodiff Newton path; the optimal warps are held fixed).
- Build: the Makefile uses the `.venv` Python and has new `sync` and `gate5`
  targets.
- Sparse / longitudinal FPCA (`fabel.sparse`, PACE): `sparse_mean`
  (R `smooth.sparse.mean`), `sparse_cov` / `SparseCov` (R `covPACE`, with the
  measurement-error variance `sigma2`), and the `PACE` estimator (R `pcaPACE`)
  with conditional-expectation (BLUP) scores in `transform` (replaces R's
  defective `scoresPACE`) and `inverse_transform`.
- Density and intensity estimation (`fabel.density`): `fit_density` (R
  `density.fd`, no longer shipped in fda 6.3.0) and `fit_intensity` (R
  `intensity.fd`), damped Newton with the exact Hessian and exact integrals;
  `DensityResult`, `IntensityResult`.
- Generalized profiling for ODE parameters (`fabel.profiling`, replaces the R
  CSTR family): `ODEModel` (analytic, finite-difference or torch-autodiff
  derivatives), `ProfiledODE`, `profile_ode`, `ProfileResult`, built-in
  `cstr_model` / `cstr_inputs` and `fitzhugh_nagumo_model`, and `simpson_rule`
  (R `quadset`). Unobserved states are allowed.
- Regression: `linmod()` / `LinmodResult` for a functional response on a
  functional covariate with a bivariate coefficient beta(s, t) (R `linmod`),
  with weights and a `predict()` method. `fregress` now computes in the input's
  array namespace: torch tensors in give torch results, with gradients to the
  response, covariates and weights.
- Registration: `register()` accepts multivariate curves (one warp per curve,
  new `var_weights=` keyword; `var_weights=[1, 0, ...]` reproduces R, which uses
  the first variable only); landmark registration and `decompose()` accept
  multivariate curves; new `RegistrationResult.apply(fd)` (R `register.newfd`).
- Dynamics: `PDA` forcing functions (`forcing_basis=`, `forcing_lam=`,
  `fit(X, forcing=u)`, `forcing_weights_`; R `pda.fd` `awtlist` / `ufdlist`) and
  `PDA.stability()` returning `PDAStability` (R `eigen.pda`, with the true
  equilibrium limits).
- Smoothing: monotone, positive and morph `SmoothResult`s evaluate exact
  derivatives of any order (Faà di Bruno / complete Bell polynomials; R
  `eval.monfd`, `eval.posfd`, `predict.monfd`).
- Statistics: pointwise `confidence_band()` / `ConfidenceBand` for a smooth
  (via `y2c_map`) or an `fregress` result (via `stderr`), and the plots
  `plot_beta` (R `plotbeta`), `cycleplot` (R `cycleplot.fd`) and `plot_scores`
  (R `plotscores`).
- Top-level exports added: `PACE`, `SparseCov`, `sparse_mean`, `sparse_cov`,
  `fit_density`, `fit_intensity`, `DensityResult`, `IntensityResult`,
  `ODEModel`, `ProfiledODE`, `profile_ode`, `linmod`, `LinmodResult`,
  `PDAStability`, and the `sparse`, `density` and `profiling` modules.

### Fixed
- `register(lam=0, criterion='eigen')` no longer raises `LinAlgError`: it warns
  (`RuntimeWarning`) when a curve has no finite optimum, and raises `ValueError`
  on NaN or infinite input.
- `FPCA` and `FCCA` accept any integer-like `n` (`SupportsIndex`).
- `FData.std` samples the pointwise standard deviation on
  `max(201, 10 * n_basis + 1)` points, matching R `sd.fd` to 1.8e-15.

### Documentation
- Dataset docstrings state the unit of every value and time field.
- Observation weights: behaviour compared with R `fRegress(wt=)` and
  `Fperm.fd` in the `fregress` / `f_test` notes and the R migration page.

### Known differences from R `fda`
- Where R `fda` 6.3.0 is demonstrably less accurate (for example `deriv.fd`,
  `times.fd`, B-spline penalties with no interior knots, `smooth.pos` stopping
  one iteration short), Fabel returns the exact value. Each case is a strict
  expected failure in the test suite with its measured error; see
  `PARITY_REPORT.md`.
