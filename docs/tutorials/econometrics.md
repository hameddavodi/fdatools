# Econometrics and interop: NumPy, SciPy, scikit-fda, statsmodels, linearmodels

`fdatools.interop` connects fdatools to the rest of the Python stack:

- **NumPy**: sampled curves in either layout, `(n_points, n_curves)` or
  `(n_curves, n_points)`.
- **SciPy and scikit-fda**: exact conversion, same knots and coefficients.
- **Panels**: `(entity, time)` MultiIndex data, the layout `linearmodels` and
  `statsmodels` use.
- **Any linear estimator**: a design matrix for scalar-on-function
  regression. Fit it with statsmodels or linearmodels, then map the
  parameters back to a coefficient function `beta(t)` with a confidence band.

The extras: `uv add "fdatools[econ]"` (pandas, statsmodels, linearmodels,
scikit-fda).

## 1. NumPy in, NumPy out

```python
import numpy as np
import fdatools as fdt
from fdatools import interop

growth = fdt.datasets.load_growth()
age = growth.age
X = np.hstack([growth.hgtm, growth.hgtf]).T            # (93, 31): one child per row
basis = fdt.BSpline(domain=(1.0, 18.0), n_basis=20, order=6)

height = interop.from_grid(X, age, basis=basis, lam=1e-2, curve_axis=0)
t, velocity = interop.to_grid(height, n_points=171, deriv=1, curve_axis=0)
velocity.shape                                         # (93, 171), ready for sklearn
```

`curve_axis=0` is the scikit-learn layout, one row per curve. Leave it at `1`
for the fdatools layout, one column per curve.

## 2. SciPy and scikit-fda, exactly

A B-spline curve becomes one vector-valued `scipy.interpolate.BSpline` with
the same knots and coefficients. SciPy's `derivative()` and `integrate()` then
give fdatools' answers.

```python
spl = interop.to_scipy(height)
bool(np.allclose(spl(age), height(age)))
```

```python
gain = spl.integrate(1.0, 18.0) / 17.0                 # mean height over ages 1-18
back = interop.from_scipy(spl)
bool(np.allclose(back.coefs, height.coefs))
```

scikit-fda shares the B-spline, Fourier and monomial conventions, so those
bases convert without loss. Other bases go through a grid.

```python
# requires: skfda
sk = interop.to_skfda(height)                          # skfda.FDataBasis
type(sk).__name__, bool(np.allclose(interop.from_skfda(sk).coefs, height.coefs))
```

## 3. Scalar-on-function regression with statsmodels

The model is `y_i = a + g w_i + ∫ x_i(t) beta(t) dt + e_i`. Write
`beta(t) = Σ_k b_k θ_k(t)`. Then the integral becomes `z_i' b` with
`z_ik = ∫ x_i θ_k`, so any linear estimator can fit `b`.
`functional_design` builds the matrix `[1, w, z]`.

To check it, regress height at 18 on height at age 1 and the velocity curve.
The answer is known: `h(18) = h(1) + ∫ v(t) dt`. So `g = 1` and
`beta(t) = 1`.

```python
# requires: statsmodels
import statsmodels.api as sm

h18 = height(np.array([18.0]))[0]
h1 = height(np.array([1.0]))[0]
design = interop.functional_design(
    {"velocity": height.derivative()},
    {"h1": h1},
    beta_basis=fdt.BSpline(domain=(1.0, 18.0), n_basis=6),
)
ols = sm.OLS(h18, design.to_frame()).fit(cov_type="HC3")   # robust covariance
round(ols.params["h1"], 6)
```

```python
# requires: statsmodels
t, est, lower, upper = design.band(ols.params, ols.cov_params(), n_points=18)
float(np.abs(est - 1.0).max()) < 1e-4                  # beta(t) = 1, as it should
```

Fit the same design with a GLM, quantile regression, WLS or a clustered
covariance in the same way. `beta()` and `band()` only need `params` and the
covariance. Here a logistic regression tells girls from boys using the
acceleration curves:

```python
# requires: statsmodels
girl = np.r_[np.zeros(39), np.ones(54)]
logit_design = interop.functional_design(
    {"accel": height.derivative(2)}, n_basis=5
)
logit = sm.Logit(girl, logit_design.to_frame()).fit(disp=0, maxiter=200)
beta_accel = logit_design.beta(logit.params)           # FData, one curve
beta_accel.basis.n_basis
```

### Principal component regression, prediction and simultaneous bands

With a fixed basis, an unpenalised fit can be noisy. `method="fpca"` expands
`beta(t)` in the first principal components of the covariate instead. This is
functional principal component regression, and it regularises by truncation.
`predict` builds the same design for new curves (same functions, same
centring), which gives out-of-sample forecasts. A simultaneous (sup-t) band
covers the whole curve at once, and `df=` uses Student-t critical values for
small samples.

```python
# requires: statsmodels
train, test = height[:70], height[70:]
pcr = interop.functional_design({"height": train}, method="fpca", n_components=3)
fit = sm.OLS(h18[:70], pcr.to_frame()).fit()
forecast = pcr.predict(fit.params, {"height": test})      # 23 out-of-sample predictions
rmse = float(np.sqrt(np.mean((forecast - h18[70:]) ** 2)))
t, est, lo, hi = pcr.band(fit.params, fit.cov_params(), df=fit.df_resid)
_, _, slo, shi = pcr.band(fit.params, fit.cov_params(), df=fit.df_resid, simultaneous=True)
bool(np.all(shi - slo >= hi - lo)), rmse < 1.0
```

With ordinary least squares the coefficient function equals
`fdt.fregress(..., lam=0)` with the same bases. Use `fregress` when you want a
roughness penalty, and the design route when you want another estimator.

## 4. Panels: linearmodels and pandas

`to_panel` writes curves as an `(entity, time)` DataFrame, the layout
`linearmodels.PanelOLS` expects. `from_panel` goes the other way. It smooths
each entity's series into a curve, and unbalanced panels are fine.

```python
# requires: linearmodels
from linearmodels import PanelOLS

rng = np.random.default_rng(7)
x_curves = fdt.FData(rng.standard_normal((8, 25)), fdt.BSpline((0.0, 1.0), 8))
years = np.linspace(0.0, 1.0, 12)
panel = interop.to_panel(x_curves, years, names=("firm", "time"), value="x")
firm_effect = np.repeat(rng.standard_normal(25), years.size)
panel["y"] = 1.5 * panel["x"] + firm_effect + 0.1 * rng.standard_normal(len(panel))
fe = PanelOLS(panel["y"], panel[["x"]], entity_effects=True).fit(cov_type="clustered")
round(float(fe.params["x"]), 1)
```

```python
# requires: pandas
smoothed = interop.from_panel(panel, "y", basis=fdt.BSpline((0.0, 1.0), 6), lam=1e-4)
smoothed.fd.n_curves, smoothed.entities[:3]
```

## R equivalent

R's `fda` leaves these conversions to you. Scalar-on-function regression with
a robust covariance usually means building `Z` by hand:

```r
library(fda); library(sandwich)
Z   <- inprod(velfd, create.bspline.basis(c(1, 18), 6))
fit <- lm(h18 ~ h1 + Z)
V   <- vcovHC(fit, type = "HC3")
```
