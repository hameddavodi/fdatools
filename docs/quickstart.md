# Quickstart

This page walks through the core objects of fdatools with the Berkeley growth
data, which ships inside the package (no download needed). Every code block
runs as written, top to bottom.

```python
import numpy as np
import fdatools as fdt

growth = fdt.datasets.load_growth()
growth.age.shape, growth.hgtf.shape     # 31 ages, 54 girls
```

## 1. A basis

A *basis* is a fixed set of simple functions. Every curve is written as a
weighted sum of them. fdatools has seven basis types: `BSpline`, `Fourier`,
`Monomial`, `Exponential`, `Power`, `Constant` and `Polygonal`.

```python
basis = fdt.BSpline(domain=(1.0, 18.0), n_basis=12)   # cubic by default (order 4)
phi = basis(growth.age)                              # (31, 12): each column is one function
d2 = basis(growth.age, deriv=2)                      # second derivatives, exact
R = basis.penalty(2)                                 # roughness penalty: integral of (D^2 phi)(D^2 phi)'
W = basis.gram()                                     # integral of phi phi'
```

## 2. Curves: `FData`

An `FData` holds the coefficients of many curves in one basis. Call it like a
function to evaluate it.

```python
coefs, *_ = np.linalg.lstsq(phi, growth.hgtf, rcond=None)
girls = fdt.FData(coefs, basis)

girls.n_curves                                   # 54
heights = girls(np.array([2.0, 10.0, 18.0]))     # (3, 54)
speed = girls.derivative()                       # exact: a lower-order spline, no projection
speed(np.array([12.0])).shape                    # (1, 54)
```

Arithmetic and statistics return new `FData` objects:

```python
mean_curve = girls.mean()
deviations = girls.center()                       # girls - mean, curve by curve
spread = girls.std()
first_two = girls[0:2]
doubled = first_two * 2.0
norm_sq = fdt.inprod(girls[0], girls[0])           # L2 inner product, same as girls[0] @ girls[0]
covariance = girls.cov()                          # a BiFData surface v(s, t)
covariance(np.array([5.0]), np.array([5.0, 10.0])).shape
```

## 3. Differential operators: `LDO`

A linear differential operator (`LDO`) is a weighted sum of derivatives. It is
used as a roughness penalty and can be evaluated on a curve.

```python
accel = fdt.LDO(2)                               # D^2, curvature
harmonic = fdt.LDO.harmonic(period=365.0)        # (2 pi / 365)^2 D + D^3, for yearly data
girls(np.array([10.0]), accel).shape            # apply the operator, then evaluate
basis.penalty(accel).shape                      # (12, 12)
```

## 4. Smoothing with automatic λ

`smooth()` fits curves by penalised least squares. With `lam="gcv"` it picks
the smoothing parameter λ that minimises generalised cross-validation (GCV).

```python
from fdatools.smoothing import smooth

fine = fdt.BSpline(domain=(1.0, 18.0), n_basis=20)
fit = smooth(growth.hgtf, growth.age, basis=fine, lam="gcv", penalty=3)
fit.lam, fit.df                                 # chosen lambda and equivalent degrees of freedom
fit.fd.n_curves                                 # 54 smooth curves
fit.gcv.shape                                   # one GCV score per curve

fixed_df = smooth(growth.hgtf, growth.age, basis=fine, lam="df=10")
round(fixed_df.df, 6)                           # 10.0
```

Shape constraints are one argument away: `constraint="monotone"` fits curves
that never decrease (like height), `constraint="positive"` fits curves that
stay above zero.

## 5. Functional PCA

`FPCA` is a scikit-learn estimator. It finds the main modes of variation
(the *harmonics*) and the score of every curve on each mode.

```python
from fdatools.decomposition import FPCA

pca = FPCA(n=3).fit(fit.fd)
pca.varprop                                     # share of variance per harmonic
pca.scores.shape                                # (54, 3)
pca.harmonics.n_curves                          # 3 harmonic curves
```

## 6. pandas and xarray

With the `pandas` extra installed, curves move in and out of data frames.

```python
# requires: pandas
frame = girls[0:3].to_pandas(np.linspace(1.0, 18.0, 5))
frame.shape
```

## 7. NumPy, SciPy and econometrics libraries

`fdatools.interop` moves curves to other libraries. `to_grid` gives a plain
array with one row per curve, the layout scikit-learn expects. `to_scipy`
gives an exact `scipy.interpolate.BSpline` with the same knots and
coefficients. `functional_design` builds a design matrix for statsmodels or
linearmodels.

```python
from fdatools import interop

t, X = interop.to_grid(fit.fd, n_points=50, curve_axis=0)
X.shape                                         # (54, 50)
spl = interop.to_scipy(fit.fd)                  # scipy BSpline, exact
design = interop.functional_design({"height": fit.fd}, method="fpca", n_components=3)
design.columns                                  # ('const', 'height[pc1]', ...)
```

## Next

- The [tutorials](tutorials/index.md) work through seven full analyses; the
  [econometrics and interop](tutorials/econometrics.md) one covers this module.
- [Migrating from R `fda`](r-migration.md) maps each R function to fdatools.
- The [API reference](api/index.md) documents every argument.
