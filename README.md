# fdatools

[![CI](https://github.com/hameddavodi/fdatools/actions/workflows/ci.yml/badge.svg)](https://github.com/hameddavodi/fdatools/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fdatools.svg)](https://pypi.org/project/fdatools/)
[![Python versions](https://img.shields.io/pypi/pyversions/fdatools.svg)](https://pypi.org/project/fdatools/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/hameddavodi/fdatools/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-hameddavodi.github.io%2Ffdatools-blue.svg)](https://hameddavodi.github.io/fdatools)

**Functional data analysis for Python.** fdatools is a clean-room Python rewrite of
R's [`fda`](https://cran.r-project.org/package=fda) package (Ramsay, Hooker &
Graves): basis expansions, penalised smoothing, functional PCA and CCA,
functional regression, curve registration and principal differential analysis.

- **Same numbers as R.** Every public function is tested against golden output
  from R `fda` 6.3.0 at `rtol = 1e-8` (`1e-5` for iterative fits). See the
  [parity report](https://github.com/hameddavodi/fdatools/blob/main/PARITY_REPORT.md).
- **About 40 symbols instead of 515 functions.** Curves are callable (`fd(t)`),
  arithmetic is plain Python (`fd1 + fd2`, `fd1 @ fd2`), and there is one
  `smooth()`, one `register()`, one `.plot()`.
- **Fits the Python stack.** scikit-learn estimators, NumPy or PyTorch arrays
  through the Array API (gradients flow), pandas and xarray I/O.
- **Works with econometrics tools.** `fdatools.interop` converts curves to and
  from NumPy, SciPy and scikit-fda (exactly, where the bases match), and to and
  from `(entity, time)` panels for linearmodels. Its design matrices let
  statsmodels or linearmodels fit scalar-on-function models (plain basis or
  functional PCA regression). You get back `beta(t)`, robust pointwise or
  simultaneous confidence bands, and predictions for new curves.

## Install

```bash
uv add fdatools
```

With [uv](https://docs.astral.sh/uv/). For an extra, put it in brackets:
`uv add "fdatools[plot]"`.

Optional extras: `fdatools[plot]` (matplotlib), `fdatools[pandas]` (pandas + xarray),
`fdatools[io]` (read R `.rds` files), `fdatools[torch]` (PyTorch),
`fdatools[econ]` (statsmodels, linearmodels, scikit-fda). Python 3.10 to 3.13.

## Quickstart

```python
import numpy as np
import fdatools as fdt

growth = fdt.datasets.load_growth()                   # Berkeley growth study, ships with fdatools
basis = fdt.BSpline(domain=(1.0, 18.0), n_basis=12)   # cubic B-splines on ages 1-18
coefs, *_ = np.linalg.lstsq(basis(growth.age), growth.hgtf, rcond=None)
girls = fdt.FData(coefs, basis)                       # 54 height curves in one object
speed = girls.derivative()                           # growth speed in cm/year, exact
print(speed.mean()(np.array([5.0, 12.0])))           # curves are callable: evaluate anywhere
print(fdt.inprod(girls[0], girls[0]))                 # L2 inner product of one curve
```

### With statsmodels

Regress a scalar on a curve with any estimator, then read the coefficient
function back:

```python
# requires: statsmodels
import statsmodels.api as sm
from fdatools import interop

height = fdt.smooth(growth.hgtf, growth.age, basis=basis).fd
adult = height(np.array([18.0]))[0]                   # height at 18
design = interop.functional_design({"speed": height.derivative()}, n_basis=5)
fit = sm.OLS(adult, design.to_frame()).fit(cov_type="HC3")   # robust covariance
beta = design.beta(fit.params)                        # beta(t) as an FData
t, est, lower, upper = design.band(fit.params, fit.cov_params(), simultaneous=True)
```

Next steps: penalised smoothing with automatic λ (`smooth`), functional PCA
(`FPCA`) and more in the [quickstart guide](https://hameddavodi.github.io/fdatools/quickstart/).
Coming from R? The [migration table](https://hameddavodi.github.io/fdatools/r-migration/)
maps every `fda` function to its fdatools equivalent. Working with NumPy,
SciPy, scikit-fda, statsmodels or linearmodels? See the
[econometrics and interop tutorial](https://hameddavodi.github.io/fdatools/tutorials/econometrics/).

## Documentation

Full documentation, API reference and tutorials: <https://hameddavodi.github.io/fdatools>.

## Citing

If you use fdatools in published work, please cite it together with Ramsay &
Silverman (2005); see [`CITATION.cff`](https://github.com/hameddavodi/fdatools/blob/main/CITATION.cff).

## License

BSD 3-Clause. fdatools is a clean-room implementation written from the published
mathematics (Ramsay & Silverman 2005; Ramsay, Hooker & Graves 2009) and the
public behaviour of R `fda`; it contains no R source code.
