# fdatools — API Specification v0.1

**Goal:** Python rewrite of R `fda` (Ramsay et al., v6.3.0).
**Promise:** 515 R functions → ~40 public symbols. Same math. Zero boilerplate.

---

## 1. Design Principles

1. **One obvious way.** One `smooth()`, one `.plot()`, one `register()`.
2. **Objects are callable.** `fd(t)` evaluates. No `eval.*` functions.
3. **Operators are math.** `fd1 + fd2`, `fd1 @ fd2` (inner product), `basis1 * basis2`.
4. **Smart defaults.** Basis size, λ (GCV), knots — all auto unless overridden.
5. **Backend-agnostic.** NumPy / PyTorch / JAX via Array API. Tensors in → tensors out. Gradients flow.
6. **Immutable objects.** Every method returns a new object → safe chaining.
7. **sklearn-compatible.** Estimators implement `fit / transform / predict`.
8. **Internals are private.** Parity is tested on public API + book figures only.

---

## 2. Package Layout

```
fdatools/
├── core.py          # FData, BiFData, LDO
├── basis.py         # Basis + 7 subclasses
├── smoothing.py     # smooth(), Smoother
├── decomposition.py # FPCA, FCCA
├── regression.py    # fregress(), FRegress
├── registration.py  # register(), landmark_register()
├── dynamics.py      # PDA, PDAStability, phase_plane()
├── stats.py         # cov, cor, depth, boxplot, f_test, t_test, confidence_band, plot_beta, cycleplot, plot_scores
├── sparse.py        # PACE, sparse_mean, sparse_cov, SparseCov (sparse / longitudinal FPCA)
├── density.py       # fit_density, fit_intensity, DensityResult, IntensityResult
├── profiling.py     # ODEModel, ProfiledODE, profile_ode, cstr_model, ... (ODE parameters)
├── datasets.py      # load_*() — all 15+ book datasets
├── nn.py            # PyTorch: BasisLayer, FDataDataset
├── _backend.py      # array-api dispatch (private)
└── _linalg.py       # banded/sparse solvers, caching (private)
```

---

## 3. Core Objects

### 3.1 `FData` — replaces `fd` + 20 R functions

```python
fd = fdt.FData(coefs, basis)          # explicit construction
y  = fd(t)                           # eval        ← eval.fd
y1 = fd(t, deriv=1)                  # eval deriv  ← eval.fd(Lfdobj=1)
```

| Python | Replaces (R) |
|---|---|
| `fd(t, deriv=0)` | `eval.fd`, `eval.monfd`, `eval.posfd` |
| `fd.derivative(n=1)` | `deriv.fd` |
| `fd.mean()` / `fd.std()` / `fd.center()` | `mean.fd`, `sd.fd`, `stddev.fd`, `center.fd` |
| `fd + fd`, `fd * 2`, `fd ** 2` | `arithmetic.fd`, `sum.fd` |
| `fd1 @ fd2` | `inprod`, `inprod.bspline` |
| `fd.cov()` | `var.fd` |
| `fd.plot()` / `fd.plot_fit(y, t)` | `plot.fd`, `plotfit.fd`, `lines.fd` |
| `fd[i]`, `fd[i:j]` | `subscript.fd`, `[.fd` |
| `fd.to_numpy(t)` / `fd.to_torch(t)` | — (new) |
| `len(fd)`, `fd.n_curves`, `fd.domain` | `fdlabels`, `norder`, etc. |

### 3.2 `Basis` — replaces 30+ R functions

```python
b = fdt.BSpline(domain=(0, 365), n_basis=65)   # ← create.bspline.basis
b = fdt.Fourier(domain=(0, 365), n_basis=65)   # ← create.fourier.basis
```

Subclasses: `BSpline, Fourier, Monomial, Exponential, Power, Constant, Polygonal`

| Python | Replaces (R) |
|---|---|
| `b(t, deriv=0)` | `eval.basis`, `getbasismatrix`, `bsplineS`, `fourier`, `expon`, `monomial`, `polyg`, `powerbasis` |
| `b.penalty(op=2)` | `getbasispenalty`, `eval.penalty`, `bsplinepen`, `fourierpen`, `polygpen`, `powerpen`, `polynompen` |
| `b1 * b2` | `basisfd.product` |
| `b.gram()` (cached) | internal `inprod` calls |

### 3.3 `LDO` — linear differential operator, replaces `Lfd`

```python
L = fdt.LDO(2)                      # ← int2Lfd(2)
L = fdt.LDO(weights=[w0, w1])       # ← vec2Lfd / Lfd
harm = fdt.LDO.harmonic(period=365) # ← the book's harmonic accelerator
```

---

## 4. High-Level API (the one-liners)

### 4.1 `smooth()` — replaces 12 R functions

```python
fd = fdt.smooth(y, t)                              # auto basis, auto λ (GCV)
fd = fdt.smooth(y, t, basis=b, lam=1e2)            # full control
fd = fdt.smooth(y, t, constraint="monotone")       # ← smooth.monotone
fd = fdt.smooth(y, t, constraint="positive")       # ← smooth.pos
fd = fdt.smooth(y, t, lam="gcv", penalty=fdt.LDO(2))
```

| Argument | Replaces (R) |
|---|---|
| default call | `smooth.basis`, `smooth.basisPar`, `Data2fd` |
| `constraint=` | `smooth.monotone`, `smooth.pos`, `smooth.morph` |
| `lam="gcv"` \| `"df=12"` | `df2lambda`, `lambda2df`, `lambda2gcv` |
| sparse/irregular `t` per curve | `smooth.basis.sparse` |
| 2-D `y` | `smooth.bibasis` |

Returns `SmoothResult`: `.fd`, `.df`, `.gcv`, `.sse`, `.penalty_matrix` — everything R scatters across list elements.

`SmoothResult.__call__(t, deriv)` accepts any `deriv >= 0` for `constraint` in {positive, monotone, morph}. The derivatives are exact, by Faà di Bruno's formula / complete Bell polynomials (this replaces `eval.posfd`, `eval.monfd` and `predict.monfd` for any `Lfdobj`).

### 4.2 Decomposition

```python
pca = fdt.FPCA(n=4, lam="gcv").fit(fd)     # ← pca.fd
pca.harmonics.plot()                       # ← plot.pca.fd
scores = pca.transform(fd)                 # sklearn-style
pca.rotate("varimax")                      # ← varmx
cca = fdt.FCCA(n=3).fit(fd1, fd2)          # ← cca.fd
```

### 4.3 Regression — replaces `fRegress` family (7 functions)

```python
m = fdt.fregress(y_fd, x_list)             # scalar/functional mix auto-detected
m = fdt.fregress("temp ~ region + precip", data)   # ← fRegress.formula
m.predict(x_new)                           # ← predict.fRegress
m.stderr()                                 # ← fRegress.stderr
m.cv()                                     # ← fRegress.CV
fdt.stats.f_test(m, n_perm=1000)            # ← Fperm.fd
```

`fregress` and `linmod` follow §1.5: torch tensors in give torch tensors out, and gradients flow to response, covariates and weights.

**Functional response on a functional covariate with a surface coefficient** (`linmod`):

```python
m = fdt.linmod(y_fd, x_fd, s_basis=bs, t_basis=bt,          # ← linmod
              lam_alpha=1e2, lam_s=1e4, lam_t=1e4)
m.alpha      # FData, intercept alpha(t)                   ← beta0estfd
m.beta       # BiFData, surface beta(s, t)                 ← beta1estbifd
m.fitted     # FData in the response basis                 ← yhatfdobj
m.predict(x_new)                                           # (new: R has no predict for linmod)
```

| Python | Replaces (R) |
|---|---|
| `linmod(y, x, alpha_basis=, s_basis=, t_basis=, lam_alpha=, lam_s=, lam_t=, penalty_alpha=, penalty_s=, penalty_t=, weights=)` | `linmod(xfdobj, yfdobj, list(fdPar(...), bifdPar(bifd, Lfds, Lfdt, lambdas, lambdat)))` |
| `LinmodResult` | the list `linmod` returns |

Model: y_i(t) = alpha(t) + ∫ x_i(s) beta(s,t) ds + e_i(t). s and t may lie on different intervals. Default bases: alpha and t take the response basis, s takes the covariate basis. Default penalties are D², default lambdas are 0. `weights` gives weighted least squares (R's `wtvec` errors in fda 6.3.0).

### 4.4 Registration — replaces 5 functions

```python
res = fdt.register(fd)                      # ← register.fd (to mean)
res = fdt.register(fd, landmarks=lm)        # ← landmarkreg
res.registered, res.warp                   # warped curves + warping fns
res.decompose()                            # ← AmpPhaseDecomp → (amp_mse, phase_mse, R²)
res.apply(new_fd)                          # ← register.newfd
```

**Multivariate curves.** `fd` may be multivariate (coefs `(n_basis, n_curves, n_vars)`). Each curve gets one warp h_i, shared by all its variables. The continuous criterion is Σ_v w_v F(x0_v, x_v ∘ h) + λ cᵀRc, with the new keyword `var_weights` (shape `(n_vars,)`, non-negative, finite, at least one positive, default all ones; ignored for landmark registration). R fda 6.3.0 `register.fd` fits multivariate warps to the first variable only; `var_weights=[1, 0, ...]` reproduces it. Landmark registration and `decompose()` accept multivariate curves (R's `landmarkreg` and `AmpPhaseDecomp` do not); `decompose()` uses squared Euclidean norms over the variables. `Registrator` (sklearn) stays univariate.

`RegistrationResult.apply(fd) -> FData` warps new curves (univariate or multivariate, any basis, e.g. derivatives) with the stored warps and shifts: curve i becomes x_i(h_i(t) + δ_i), wrapped periodically when any shift is non-zero, then least-squares fitted in `fd`'s basis on a grid of max(201, 10K+1) points. Torch coefficients give a differentiable tensor result.

### 4.5 Dynamics

```python
pda = fdt.PDA(order=2).fit(fd)              # ← pda.fd
pda.plot_overlay()                          # ← pda.overlay
fdt.phase_plane(fd)                          # ← phaseplanePlot
pda = fdt.PDA(order=1, forcing_basis=b).fit(fd, forcing=u)  # ← pda.fd awtlist / ufdlist
st = pda.stability()                        # ← eigen.pda → PDAStability(t, eigenvalues, limits)
st.plot()
```

`PDA(order=2, *, weight_basis=None, lam=0.0, penalty=2, n_grid=501, forcing_basis=None, forcing_lam=0.0)` fits D^m x = −Σ_j b_j D^j x + Σ_k a_k u_k. `forcing` is an FData or a list of FData for one equation; for a system it is a list with one entry per equation (each None, an FData or a list of FData). Each u has 1 variable and either 1 curve (shared by all curves) or n_curves curves. The fitted forcing weights are in `forcing_weights_` (a tuple a_k for one equation, tuple[i][k] for a system; empty without forcing). `transform`, `fit_transform` and `solve(t, initial, *, forcing=None)` take the same structure; `solve` without forcing integrates the homogeneous equation.

`PDA.stability(t=None, *, n_points=501, forcing=None) -> PDAStability`: a frozen dataclass `(t, eigenvalues, limits)` with `.plot(ax=None, **kw)`. `eigenvalues` has shape `(n_t, n_vars*order)`, complex: the eigenvalues of the companion matrix A(t), sorted by decreasing modulus. `limits` holds the equilibrium states −A(t)⁻¹ f(t): zero when unforced, NaN where A is singular.

### 4.6 Sparse / longitudinal FPCA (PACE) — replaces 4 R functions

```python
from fdatools.sparse import PACE, sparse_mean, sparse_cov
mu  = sparse_mean(y, t, basis, lam=0.0)          # ← smooth.sparse.mean
est = sparse_cov(y, t, mean=mu, basis=b, lam=1)  # ← covPACE  (.cov BiFData, .mean, .sigma2, .variance)
pace = PACE(n=3, basis=b, mean_basis=mb, harmonic_basis=hb,
            lam_mean=0.0, lam_cov=0.0, lam=0.0).fit(y, t=t)   # ← pcaPACE
pace.harmonics, pace.values, pace.varprop, pace.sigma2_
scores = pace.transform(y_new, t_new)            # ← scoresPACE (conditional expectation / BLUP)
curves = pace.inverse_transform(scores, grid)    # mean + sum_k score_k * harmonic_k on a grid
```

Input is the irregular per-curve form of `smooth()`: `y` and `t` are sequences with one 1-D array per curve. With `t=None`, `y` may instead be R's list form: one `(n_i, 2)` array of (time, value) rows per curve. The covariance surface is fitted to the within-curve cross-products of different points only (the diagonal is left out). The measurement-error variance `sigma2` comes from the smoothed diagonal (Yao, Müller & Wang 2005). Harmonics follow R's `pcaPACE` sign rule. Scores are the PACE conditional expectation, not R's defective `scoresPACE` output. `PACE` follows the estimator API (params, clone, pickle). Its input is ragged, so the generic `check_estimator` checks do not apply.

| Python | Replaces (R) |
|---|---|
| `sparse_mean` | `smooth.sparse.mean` |
| `sparse_cov` / `SparseCov` | `covPACE` |
| `PACE` (fit / values / harmonics / varprop) | `pcaPACE` |
| `PACE.transform` / `PACE.scores` | `scoresPACE` |

### 4.7 Density and intensity estimation — replaces `density.fd`, `intensity.fd`

```python
res = fdt.fit_density(x, basis=b, lam=1e-2, penalty=2)   # ← density.fd (dropped from fda 6.3.0)
res(t)                 # density p(t) = C exp W(t)
res.log_density(t)     # log p(t)
res.fd, res.normaliser # W (FData) and C = 1/∫exp W
ev = fdt.fit_intensity(times, basis=b, lam=10, penalty=1)  # ← intensity.fd
ev(t)                  # intensity mu(t) = exp W(t);  ev.expected_count = ∫ mu
```

| Python | Replaces (R) |
|---|---|
| `fit_density(x, basis=, domain=, lam=, penalty=, start=, tol=, max_iter=)` → `DensityResult` | `density.fd` |
| `fit_intensity(x, basis=, domain=, lam=, penalty=, start=, tol=, max_iter=)` → `IntensityResult` | `intensity.fd` |

Criteria: density `-Σ W(x_i) + n log ∫e^W + λ cᵀRc`; intensity `-Σ W(x_i) + ∫e^W + λ cᵀRc` (the same as R's `Flist$f`). Damped Newton with the exact Hessian. Exact Gauss-Legendre integrals. Stops once a full Newton step moves no coefficient by more than `tol·(1+max|c|)` (default `tol=1e-10`, `max_iter=100`, `RuntimeWarning` if not converged). Parity is iterative: rtol 1e-5. Defaults: `basis` is cubic B-splines with `min(max(4, n//10), 20)` functions. The domain is `(min x, max x)` for a density and `(0, max x)` for an intensity. `lam=0`, `penalty=2`. If W is only defined up to a constant (the basis spans constants and L annihilates them), the returned W has `∫W = 0`.

### 4.8 Profiling (ODE parameter estimation) — replaces the CSTR family

```python
from fdatools.profiling import ODEModel, cstr_model, fitzhugh_nagumo_model
model = ODEModel(rhs, n_states=2, n_params=3, jac_x=..., jac_theta=...)  # f(x, t, theta), (n,d) in/out
model = ODEModel.from_torch(torch_rhs, 2, 3)          # exact autodiff derivatives (fdatools[torch])
fit = fdt.profile_ode(model, t, y, basis, lam=1e3, theta0=[...])   # y: (n,d), NaN = unobserved
fit.theta, fit.cov, fit.stderr, fit.states, fit(t), fit.theta_path, fit.inner.df
problem = fdt.ProfiledODE(model, t, y, bases, lam, state_weights=[1/var_C, 1/var_T])
problem.residuals(coefs, theta)   # ← CSTRfitLS (residuals + Jacobians)
problem.fit_states(theta)         # ← CSTRfn (inner state fit)
problem.fit(theta0)               # ← CSTRres + nls / CSTRsse + optim (outer fit)
```

| Python | Replaces (R) |
|---|---|
| `ODEModel`, `cstr_model(condition, estimate=...)` | `CSTR2` |
| `cstr_inputs(t, condition)` | `CSTR2in` (step scenarios) |
| `ProfiledODE.residuals` | `CSTRfitLS` |
| `ProfiledODE.fit_states` | `CSTRfn` |
| `profile_ode` / `ProfiledODE.fit` | `CSTRres`, `CSTRsse` (+ `nls`/`optim`) |
| `simpson_rule(breaks, n_quad)` | `quadset` |
| `ODEModel.simulate` | `lsoda(y, times, CSTR2, parms)` |

Inner criterion: J(c|θ) = Σ_i w_i [ Σ_j (y_ij − x_i(t_ij))² + λ_i ∫ (Dx_i − f_i(x,t,θ))² ], minimised by damped Gauss-Newton. The integral uses composite Simpson on the B-spline breaks. Outer criterion: the profiled data SSE, minimised by Gauss-Newton with dc/dθ from the implicit function theorem (exact inner Hessian). Covariance = σ² (AᵀA)⁻¹. NumPy results; torch right-hand sides via `ODEModel.from_torch`. Top-level exports: `profile_ode`, `ODEModel`, `ProfiledODE` (the rest stay in `fdatools.profiling`).

### 4.9 Statistics: confidence bands and plots

```python
band  = fdt.stats.confidence_band(smooth_result, t, sigma_e=None, level=0.95, deriv=0)  # ← smooth.basis y2cMap variance (book 5.5)
bands = fdt.stats.confidence_band(fregress_result, t, sigma_e=None, y2c_map=None)      # ← fRegress.stderr pointwise limits (book 9.4)
band.lower, band.upper, band.stderr, band.plot()
fdt.stats.plot_beta(fregress_result)        # ← plotbeta
fdt.stats.cycleplot(fd_bivariate)           # ← cycleplot.fd
fdt.stats.plot_scores(fpca, (0, 1))         # ← plotscores
```

| Python | Replaces (R) |
|---|---|
| `stats.confidence_band` / `stats.ConfidenceBand` | pointwise SE via `smooth.basis()$y2cMap`, `fRegress.stderr` |
| `stats.plot_beta` | `plotbeta` |
| `stats.cycleplot` | `cycleplot.fd` |
| `stats.plot_scores` | `plotscores` |
| (private `fdatools._plot.zerofind`) | `zerofind` |

Smooth: Var x(t) = φ(t)ᵀ S Σ Sᵀ φ(t), with S the `y2c_map`. `sigma_e` is a number, a length-n_obs vector or an (n_obs, n_obs) matrix; the default is SSE/(N(n_obs−df)) I. Constrained or irregular smooths raise ValueError. Regression: Var β_j(t) = θ_j(t)ᵀ V_jj θ_j(t), with V from `FRegressResult.stderr`; one band per term. Bands are pointwise, using the normal quantile z = Φ⁻¹((1+level)/2). Torch in gives torch out.

---

## 5. ML Integration

### 5.1 scikit-learn (native, no wrappers)

`Smoother`, `FPCA`, `FCCA`, `Registrator`, `FRegress` all implement the estimator API:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("smooth", fdt.Smoother()),
    ("fpca",  fdt.FPCA(n=5)),
    ("clf",   LogisticRegression()),
])
GridSearchCV(pipe, {"fpca__n": [3, 5, 8]}).fit(X, y)
```

### 5.2 PyTorch (`fdatools.nn`)

```python
layer = fdt.nn.BasisLayer(fdt.BSpline(domain, 32))  # differentiable fd(t)
ds    = fdt.nn.FDataDataset(fd, labels)            # → DataLoader ready
```

- All core ops accept `torch.Tensor` → return `torch.Tensor`
- Gradients flow through smoothing, eval, inner products (autodiff registration!)
- `.to("cuda")` supported via Array API dispatch

### 5.3 Ecosystem I/O

```python
fdt.from_pandas(df, id_col, t_col, y_col)   # long-format → FData
fd.to_xarray() / fd.to_pandas(t)
fdt.read_rds("weather.rds")                  # ingest R fda objects directly
```

### 5.4 Interoperability (`fdatools.interop`)

Added after 1.1.0 on user request; extends the frozen surface with one module.

```python
from fdatools import interop
interop.from_grid(X, t, curve_axis=0) / interop.to_grid(fd, curve_axis=0)  # NumPy (either layout)
interop.to_scipy(fd) / interop.from_scipy(spl)                            # scipy BSpline, exact
interop.to_skfda(fd) / interop.from_skfda(obj)                            # scikit-fda, exact for shared bases
interop.from_panel(df, "y") / interop.to_panel(fd, t)                     # (entity, time) panels: linearmodels
d = interop.functional_design({"x": fd}, {"w": w})                        # design for any linear estimator
d.beta(res.params); d.band(res.params, res.cov_params())                  # back to beta(t) + band
```

---

## 6. Performance Spec

| Bottleneck in R | fdatools fix |
|---|---|
| Dense penalty matrices | scipy.sparse banded (B-spline Gram is banded) |
| Repeated basis re-evaluation | LRU cache keyed on (knots, order, t-hash) |
| Loops over curves | einsum-batched, one BLAS call |
| Cholesky per λ in GCV search | factorize once, rank-1 λ updates |
| Single-core | optional Numba `parallel=True`; torch → GPU |

**Targets:** ≥10× R on smoothing 1k curves; ≥50× on GCV grid search.

---

## 7. Parity & Testing

- **Golden files:** Docker + rpy2 runs R fda v6.3.0 on fixed inputs → JSON.
- **pytest:** every public symbol vs golden, `rtol=1e-8` (1e-5 for iterative: monotone, register).
- **Acceptance:** reproduce all 76 figures from the 2009 book.
- **CI gate:** parity job must pass to merge.
- **License note:** clean-room from books + docs. No R source translation → BSD-3 safe.

---

## 8. Datasets (`fdatools.datasets`)

`load_canadian_weather()`, `load_growth()`, `load_gait()`, `load_handwriting()`,
`load_pinch()`, `load_melanoma()`, `load_refinery()`, `load_seabird()`,
`load_cstr()`, `load_chinese_script()`, `load_regina_precip()` — lazy-download, cached, returns `FData` + metadata.

---

## 9. Team Split & Phases

| Phase | Owner | Deliverable | Depends on |
|---|---|---|---|
| 0 (wk 1) | All | Freeze this spec | — |
| 1 | Eng 1 | `basis` + `core` + `_backend` | — |
| 1 | Eng 2 | Parity harness + golden files | — |
| 2 | Eng 3 | `smoothing` + `LDO` + `stats` | Phase 1 |
| 2 | Eng 4 | `datasets` + I/O | Phase 1 |
| 3 | Eng 3 | `decomposition` + `regression` | Phase 2 |
| 3 | Eng 4 | `nn` + sklearn compliance tests | Phase 2 |
| 4 | Eng 1+4 | `registration` + `dynamics` (hardest) | Phase 3 |
| 5 | All | Docs, 76-figure notebook, PyPI | Phase 4 |

---

## 10. Explicitly NOT in v1

- R plotting quirks (`axisIntervals`, `matplot`) — replaced by matplotlib/plotly defaults
- `readHMD` (Human Mortality DB scraper) — out of scope
- Deprecated/internal R helpers (`wtcheck`, `symsolve`, `polintmat`, `lnsrch`...) — private or scipy
