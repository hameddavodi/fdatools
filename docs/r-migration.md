# Migrating from R `fda`

fdatools replaces the 515 functions of R's `fda` 6.3.0 with about 40 Python
symbols. The ideas stay the same: a curve is a set of basis coefficients, a
roughness penalty is a linear differential operator, and every estimator works
on those coefficients. What changes is the surface:

- **Objects are callable.** `fd(t)` evaluates a curve. There is no `eval.*` family.
- **Operators are math.** `fd1 + fd2`, `fd * 2`, `fd1 @ fd2` (inner product), `basis1 * basis2`.
- **One entry point per task.** One `smooth()`, one `register()`, one `.plot()`.
- **Immutable objects.** Every method returns a new object, so chaining is safe.
- **scikit-learn estimators.** `Smoother`, `FPCA`, `FCCA`, `FRegress` and
  `Registrator` implement `fit` / `transform` / `predict`.

In the tables below, `fb` is `import fdatools as fdt`.

## Functional data objects (`fd` → `FData`)

| R `fda` | fdatools |
|---|---|
| `fd(coef, basisobj)` | `fdt.FData(coefs, basis)` |
| `eval.fd` | `fd(t, deriv=0)` |
| `eval.monfd`, `eval.posfd`, `predict.monfd` (any derivative) | `result(t, deriv=n)` on a monotone or positive `SmoothResult` (exact, any order) |
| `eval.fd(t, fdobj, Lfdobj)` | `fd(t, operator)` with an `LDO` |
| `deriv.fd` | `fd.derivative(n=1)` |
| `mean.fd`, `sd.fd`, `stddev.fd`, `center.fd` | `fd.mean()`, `fd.std()`, `fd.center()` |
| `arithmetic.fd`, `sum.fd` (`+`, `-`, `*`, `^`) | `fd + fd`, `fd * 2`, `fd ** 2` |
| `inprod`, `inprod.bspline` | `fd1 @ fd2`, `fdt.inprod(a, b, lfd1=, lfd2=)` |
| `var.fd` | `fd.cov()` |
| `plot.fd`, `lines.fd`, `plotfit.fd` | `fd.plot()`, `fd.plot_fit(y, t)` |
| `[.fd`, `subscript.fd` | `fd[i]`, `fd[i:j]` |
| `fdlabels`, `norder`, `nbasis` | `len(fd)`, `fd.n_curves`, `fd.domain`, `fd.basis` |
| (none) | `fd.to_numpy(t)`, `fd.to_torch(t)` |
| `bifd`, `eval.bifd` | `fdt.BiFData(coefs, s_basis, t_basis)`, `bifd(s, t)` |

## Basis systems (`create.*.basis` → `Basis` subclasses)

| R `fda` | fdatools |
|---|---|
| `create.bspline.basis` | `fdt.BSpline(domain=, n_basis=, order=, breaks=)` |
| `create.fourier.basis` | `fdt.Fourier(domain=, n_basis=, period=)` |
| `create.monomial.basis` | `fdt.Monomial(domain=, n_basis=)` |
| `create.exponential.basis` | `fdt.Exponential(domain=, rates=)` |
| `create.power.basis` | `fdt.Power(domain=, exponents=)` |
| `create.constant.basis` | `fdt.Constant(domain=)` |
| `create.polygonal.basis` | `fdt.Polygonal(argvals)` |
| `eval.basis`, `getbasismatrix`, `bsplineS`, `fourier`, `expon`, `monomial`, `polyg`, `powerbasis` | `basis(t, deriv=0)` |
| `getbasispenalty`, `eval.penalty`, `bsplinepen`, `fourierpen`, `polygpen`, `powerpen`, `polynompen` | `basis.penalty(op=2)` |
| `basisfd.product` | `basis1 * basis2` |
| internal `inprod` calls on a basis | `basis.gram()` (cached) |

## Linear differential operators (`Lfd` → `LDO`)

| R `fda` | fdatools |
|---|---|
| `int2Lfd(m)` | `fdt.LDO(m)` |
| `vec2Lfd`, `Lfd` | `fdt.LDO(weights=[w0, w1, ...])` |
| harmonic acceleration `vec2Lfd(c(0, (2*pi/T)^2, 0), rng)` | `fdt.LDO.harmonic(period=T)` |

## Smoothing (`smooth.*` → `smooth()`)

| R `fda` | fdatools |
|---|---|
| `smooth.basis`, `smooth.basisPar`, `Data2fd` | `fdt.smooth(y, t, basis=, lam=, penalty=)` |
| `smooth.monotone` | `fdt.smooth(y, t, constraint="monotone")` |
| `smooth.pos` | `fdt.smooth(y, t, constraint="positive")` |
| `smooth.morph` | `fdt.smooth(y, t, constraint="morph")` |
| `lambda2gcv`, GCV search over λ | `fdt.smooth(y, t, lam="gcv")`, `gcv_curve(...)` |
| `df2lambda` | `fdt.smooth(y, t, lam="df=12")`, `df_to_lambda(...)` |
| `lambda2df` | `lambda_to_df(...)`, `result.df` |
| `smooth.basis.sparse` (irregular sampling per curve) | `fdt.smooth([y1, y2, ...], [t1, t2, ...])` |
| `smooth.bibasis` | `fdt.smooth()` with a 2-D `y` |
| the returned list (`fd`, `df`, `gcv`, `SSE`, `penmat`, `y2cMap`) | `SmoothResult`: `.fd`, `.df`, `.gcv`, `.sse`, `.penalty_matrix`, `.y2c_map` |
| (none) | `fdt.Smoother()`: scikit-learn transformer |

## Decomposition (`pca.fd`, `cca.fd` → `FPCA`, `FCCA`)

| R `fda` | fdatools |
|---|---|
| `pca.fd(fdobj, nharm, harmfdPar)` | `fdt.FPCA(n=4, lam="gcv").fit(fd)` |
| `pca.fd(...)$harmonics` / `$scores` / `$values` / `$varprop` | `pca.harmonics`, `pca.scores`, `pca.values`, `pca.varprop` |
| scores of new curves | `pca.transform(fd)` |
| `plot.pca.fd` | `pca.harmonics.plot()` |
| `varmx.pca.fd` | `pca.rotate("varimax")` |
| `cca.fd` | `fdt.FCCA(n=3).fit(fd1, fd2)` |

## Regression (`fRegress` family → `fregress()` / `FRegress`)

| R `fda` | fdatools |
|---|---|
| `fRegress` (scalar or functional response) | `fdt.fregress(y, x_list)` |
| `fRegress.formula` | `fdt.fregress("temp ~ region + precip", data)` |
| `predict.fRegress` | `model.predict(x_new)` |
| `fRegress.stderr` | `model.stderr()` |
| `fRegress.CV` | `model.cv()` |
| `Fperm.fd` | `fdt.stats.f_test(model, n_perm=1000)` |
| (none) | `fdt.FRegress`: scikit-learn estimator |
| `linmod` (with `bifdPar` for the surface penalties) | `fdt.linmod(y, x, s_basis=, t_basis=, lam_alpha=, lam_s=, lam_t=)` → `LinmodResult` (`.alpha`, `.beta`, `.fitted`, `.predict`) |
| pointwise limits from `fRegress.stderr` | `fdt.stats.confidence_band(model)` (one band per term) |
| `plotbeta` | `fdt.stats.plot_beta(model)` |

## Registration (`register.fd`, `landmarkreg` → `register()`)

| R `fda` | fdatools |
|---|---|
| `register.fd` (continuous registration to the mean) | `fdt.register(fd)` |
| `landmarkreg` | `fdt.register(fd, landmarks=lm)`, `fdt.landmark_register(fd, landmarks)` |
| `$regfd`, `$warpfd` | `result.registered`, `result.warp` |
| `AmpPhaseDecomp` | `result.decompose()` → amplitude MSE, phase MSE, R² |
| `register.fd` on multivariate curves | `fdt.register(fd, var_weights=...)` (one warp per curve; `var_weights=[1, 0, ...]` reproduces R, which uses the first variable only) |
| `landmarkreg` on multivariate curves | `fdt.landmark_register(fd, landmarks)` (R rejects multivariate curves) |
| `register.newfd` | `result.apply(new_fd)` |
| (none) | `fdt.Registrator`: scikit-learn transformer |

## Dynamics (`pda.fd` → `PDA`)

| R `fda` | fdatools |
|---|---|
| `pda.fd` | `fdt.PDA(order=2).fit(fd)` |
| `pda.overlay` | `pda.plot_overlay()` |
| `phaseplanePlot` | `fdt.phase_plane(fd)` |
| `pda.fd(..., awtlist, ufdlist)` (forcing functions) | `fdt.PDA(forcing_basis=, forcing_lam=).fit(fd, forcing=u)`, `pda.forcing_weights_` |
| `eigen.pda` | `pda.stability()` → `PDAStability` (`.eigenvalues`, `.limits`, `.plot()`) |

## Statistics (`fdatools.stats`)

| R `fda` | fdatools |
|---|---|
| `var.fd` | `fdt.stats.cov(fd)` |
| `cor.fd` | `fdt.stats.cor(fd1, fd2)` |
| `fdepth` | `fdt.stats.depth(fd)` |
| `fbplot`, `boxplot.fd` | `fdt.stats.boxplot(fd)` |
| `Fperm.fd` | `fdt.stats.f_test(model, n_perm=)` |
| `tperm.fd` | `fdt.stats.t_test(fd1, fd2, n_perm=)` |
| pointwise variance from `smooth.basis()$y2cMap` | `fdt.stats.confidence_band(smooth_result, t, sigma_e=)` |
| `cycleplot.fd` | `fdt.stats.cycleplot(fd)` |
| `plotscores` | `fdt.stats.plot_scores(pca, (0, 1))` |
| `zerofind` | (private; used by `plot_beta`) |

## Sparse longitudinal data (PACE → `fdatools.sparse`)

| R `fda` | fdatools |
|---|---|
| `smooth.sparse.mean` | `fdt.sparse_mean(y, t, basis, lam=)` |
| `covPACE` | `fdt.sparse_cov(y, t, mean=, basis=, lam=)` → `SparseCov` (`.cov`, `.sigma2`, `.variance`) |
| `pcaPACE` | `fdt.PACE(n=3, ...).fit(y, t=t)`: `.harmonics`, `.values`, `.varprop` |
| `scoresPACE` | `pace.transform(y, t)` (conditional expectation; R's `scoresPACE` is defective in 6.3.0) |

## Density and intensity (`fdatools.density`)

| R `fda` | fdatools |
|---|---|
| `density.fd` (no longer shipped in fda 6.3.0) | `fdt.fit_density(x, basis=, lam=, penalty=)` → `DensityResult` |
| `intensity.fd` | `fdt.fit_intensity(times, basis=, lam=, penalty=)` → `IntensityResult` |

## ODE parameters by profiling (CSTR family → `fdatools.profiling`)

| R `fda` | fdatools |
|---|---|
| `CSTR2` | `fdt.profiling.cstr_model(condition, estimate=...)` or any `fdt.ODEModel` |
| `CSTR2in` | `fdt.profiling.cstr_inputs(t, condition)` (the 4 step scenarios) |
| `CSTRfitLS` | `fdt.ProfiledODE(...).residuals(coefs, theta)` |
| `CSTRfn` | `problem.fit_states(theta)` |
| `CSTRres`, `CSTRsse` (+ `nls` / `optim`) | `fdt.profile_ode(model, t, y, basis, lam=, theta0=)` / `problem.fit(theta0)` |
| `quadset` | `fdt.profiling.simpson_rule(breaks, n_quad)` |
| `lsoda(y, times, CSTR2, parms)` | `model.simulate(...)` |

## Datasets (`data(...)` → `fdatools.datasets`)

| R `fda` | fdatools |
|---|---|
| `CanadianWeather` | `fdt.datasets.load_canadian_weather()` |
| `growth` | `fdt.datasets.load_growth()` |
| `gait` | `fdt.datasets.load_gait()` |
| `handwrit`, `handwritTime` | `fdt.datasets.load_handwriting()` |
| `pinch`, `pinchtime`, `pinchraw` | `fdt.datasets.load_pinch()` |
| `melanoma` | `fdt.datasets.load_melanoma()` |
| `refinery` | `fdt.datasets.load_refinery()` |
| `seabird` | `fdt.datasets.load_seabird()` |
| `ReginaPrecip` | `fdt.datasets.load_regina_precip()` |
| `MontrealTemp` | `fdt.datasets.load_montreal_temp()` |
| `daily` | `fdt.datasets.load_daily()` |
| `infantGrowth` | `fdt.datasets.load_infant_growth()` |
| `nondurables` | `fdt.datasets.load_nondurables()` |
| `lip`, `lipmarks`, `liptime` | `fdt.datasets.load_lip()` |

`growth`, `gait` and `pinch` ship inside the package. The others are
downloaded once from the project's GitHub release, checked against a SHA-256
checksum, and cached in `~/.cache/fdatools` (override with `FDATOOLS_DATA_DIR`).

## Ecosystem I/O, econometrics interop and deep learning

| R `fda` | fdatools |
|---|---|
| `readRDS` on a saved `fd`, `bifd` or `basisfd` | `fdt.read_rds("weather.rds")` |
| (none) | `fdt.from_pandas(df, id_col, t_col, y_col)` (long format) |
| (none) | `fd.to_pandas(t)`, `fd.to_xarray(t)` |
| `eval.fd` on a grid, then `t()` | `interop.to_grid(fd, t, curve_axis=0)` (NumPy, one row per curve) |
| `Data2fd` / `smooth.basis` on a matrix | `interop.from_grid(X, t, curve_axis=0)` |
| (none) | `interop.to_scipy(fd)`, `interop.from_scipy(spl)`: exact `scipy.interpolate.BSpline` |
| (none) | `interop.to_skfda(fd)`, `interop.from_skfda(obj)`: scikit-fda `FDataBasis` / `FDataGrid` |
| (none) | `interop.from_panel(df, "y")`, `interop.to_panel(fd, t)`: `(entity, time)` panels |
| `Z <- inprod(xfd, betabasis)` then `lm(y ~ Z)` | `d = interop.functional_design({"x": xfd})`, fit with statsmodels / linearmodels, then `d.beta(params)` |
| (none) | `d.band(params, cov, simultaneous=True)`, `d.predict(params, x_new)` |
| (none) | `fdt.nn.BasisLayer(basis)`: differentiable `fd(t)` for PyTorch |
| (none) | `fdt.nn.FDataDataset(fd, labels)`: `DataLoader`-ready dataset |

## Not carried over

- R plotting quirks (`axisIntervals`, `matplot`): fdatools uses matplotlib defaults.
- `readHMD` (Human Mortality Database scraper).
- Internal helpers (`wtcheck`, `symsolve`, `polintmat`, `lnsrch`, ...): private
  in fdatools or replaced by SciPy.

## Behaviour differences from R

| Topic | R `fda` 6.3.0 | fdatools |
|---|---|---|
| Weights in `fRegress` | `fRegress(y, xfdlist, betalist, wt = w)` fits by weighted least squares, scalar and functional response alike. The argument is `wt`: `wtvec = w` falls into `...` and is ignored without a warning, so R returns the unweighted fit. | `fdt.fregress(y, x, weights=w)` fits by weighted least squares and agrees with R's `wt = w` to the accuracy of R's integration. Drop `weights` when porting a script that passed `wtvec` to `fRegress`. |
| Weights in `linmod` | `linmod(..., wtvec = w)` errors in 6.3.0. | `fdt.linmod(y, x, weights=w)` fits by weighted least squares. |
| `eigen.pda` limits | `limvals` has a wrong sign (order 1), wrong entries (order 2) or loses forcing (systems). | `PDAStability.limits` is the true equilibrium −A(t)⁻¹f(t). |
| Weights in `Fperm.fd` | `Fperm.fd(..., wt = w)` accepts weights but ignores them: `Fobs` and the null distribution are the same as with no weights. | `fdt.stats.f_test(model)` refits a weighted model with its weights under every permutation, so the statistic changes with the weights. Test a model fitted without `weights` to reproduce R. The raw form `f_test(y, x)` takes no weights. |

With a roughness penalty, only the relative size of the weights and `lam`
matters: multiplying every weight by `c` has the same effect as dividing `lam`
by `c`. With unit weights both functions give exactly the unweighted result.

## Where fdatools and R disagree on purpose

fdatools matches R to `rtol = 1e-8` (`1e-5` for iterative fits) on every golden
case except a documented set where R is the less accurate side, for example
`deriv.fd` re-projecting a spline derivative (1.3% error, fdatools is exact) or
`times.fd` using a basis too smooth to hold the product (12.8% error). Each of
these is a strict expected failure in the test suite with its measured reason;
the full list is in `PARITY_REPORT.md` in the repository.
