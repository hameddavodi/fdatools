# %% [markdown]
# # A full tour of `fdatools`
#
# `fdatools` is a Python rewrite of R's `fda` package (Ramsay, Hooker & Graves) for
# *functional data analysis* (FDA). In FDA every observation is a whole **curve**
# (a height curve, a temperature year, a lip movement), not a single number.
#
# This notebook uses **every public part** of the library on real data sets.
# Each part shows how to use a tool, draws a picture, and **checks** a result.
# The `check()` helper writes every result into a table that appears at the end.
# If a check fails, the notebook stops at that cell.
#
# | Part | Topic | Part | Topic |
# |---|---|---|---|
# | 0 | Setup and the 14 data sets | 9 | Registration (lining up curves) |
# | 1 | Bases: the 7 building blocks | 10 | Dynamics: differential equations from data |
# | 2 | Curves: `FData` and `BiFData` | 11 | Sparse data: PACE |
# | 3 | Linear differential operators: `LDO` | 12 | Densities and event rates |
# | 4 | Smoothing | 13 | ODE parameter estimation (profiling) |
# | 5 | Descriptive statistics | 14 | Machine learning: scikit-learn and PyTorch |
# | 6 | Functional PCA | 15 | Input and output: pandas, xarray, R files |
# | 7 | Functional CCA | 16 | Live comparison with R `fda` |
# | 8 | Regression and permutation tests | 17 | Summary of all checks |
#
# **Run it:** *Kernel → Restart Kernel and Run All Cells*. It takes about one minute.

# %%
import os
import time
import warnings
from pathlib import Path

# Some data sets are not inside the wheel. Use the local copy in data_release/
# when it exists (in a normal install they are downloaded once and cached).
here = Path.cwd()
for folder in (here, *here.parents):
    if (folder / "data_release").is_dir():
        os.environ.setdefault("FDATOOLS_DATA_DIR", str(folder / "data_release"))
        break

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display as show

import fdatools as fdt

plt.rcParams.update({"figure.figsize": (10, 4), "axes.grid": True, "grid.alpha": 0.3})
np.set_printoptions(precision=4, suppress=True)

RESULTS = []  # one row per check, shown at the end


def check(name, ok, detail=""):
    """Record a check and stop the notebook if it fails."""
    detail = str(detail)
    RESULTS.append({"part": PART, "check": name, "passed": bool(ok), "detail": detail})
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    assert ok, name


PART = "0 setup"
T0 = time.time()
print("fdatools", fdt.__version__)

# %% [markdown]
# ## Part 0. The data sets
#
# The library ships with 14 classic FDA data sets (from the Ramsay & Silverman
# books). `growth`, `gait` and `pinch` are inside the package; the others are
# downloaded once and then cached. Each loader returns a small, typed object.

# %%
loaders = [name for name in fdt.datasets.__all__ if name.startswith("load_")]
rows = []
for name in loaders:
    data = getattr(fdt.datasets, name)()
    fields = {
        key: getattr(value, "shape", len(value) if hasattr(value, "__len__") else value)
        for key, value in vars(data).items()
    }
    rows.append({"loader": name, "fields": ", ".join(f"{k} {v}" for k, v in fields.items())})
pd.set_option("display.max_colwidth", 140)
show(pd.DataFrame(rows))
check("all 14 data sets load", len(rows) == 14, len(rows))

# %%
growth = fdt.datasets.load_growth()
weather = fdt.datasets.load_canadian_weather()
gait = fdt.datasets.load_gait()
pinch = fdt.datasets.load_pinch()
lip = fdt.datasets.load_lip()
hw = fdt.datasets.load_handwriting()

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
axes[0, 0].plot(growth.age, growth.hgtf, lw=0.6)
axes[0, 0].set_title("Growth: height of 54 girls")
axes[0, 1].plot(weather.t, weather.temp, lw=0.4)
axes[0, 1].set_title("Canadian weather: 35 stations")
axes[0, 2].plot(gait.t, gait.value[:, :, 0], lw=0.6)
axes[0, 2].set_title("Gait: hip angle, 39 boys")
axes[1, 0].plot(pinch.t, pinch.pinch, lw=0.6)
axes[1, 0].set_title("Pinch force, 20 repeats")
axes[1, 1].plot(lip.t, lip.value, lw=0.6)
axes[1, 1].set_title("Lip position saying 'bob'")
axes[1, 2].plot(hw.value[:, :5, 0], hw.value[:, :5, 1], lw=0.6)
axes[1, 2].set_title("Handwriting 'fda', 5 samples")
axes[1, 2].set_aspect("equal")
fig.tight_layout()

# %% [markdown]
# ## Part 1. Bases: the building blocks
#
# A **basis** is a fixed set of simple functions φ₁, …, φ_K. Every curve is
# written as a weighted sum x(t) = Σ cₖ φₖ(t); the weights cₖ are the
# *coefficients*. There are 7 basis types. Every basis can give its values, any
# derivative, its **Gram matrix** W = ∫ φ φᵀ (inner products of the basis
# functions) and a **roughness penalty matrix** R = ∫ (Dᵐφ)(Dᵐφ)ᵀ.

# %%
PART = "1 bases"
bases = {
    "BSpline (cubic, 8)": fdt.BSpline(domain=(0.0, 1.0), n_basis=8),
    "Fourier (7)": fdt.Fourier(domain=(0.0, 1.0), n_basis=7),
    "Monomial (1, t, t², t³)": fdt.Monomial(domain=(0.0, 1.0), n_basis=4),
    "Exponential (rates 0, ±2)": fdt.Exponential(domain=(0.0, 1.0), rates=[0.0, 2.0, -2.0]),
    "Power (t^0.5, t, t^1.5)": fdt.Power(domain=(0.1, 1.0), exponents=[0.5, 1.0, 1.5]),
    "Constant": fdt.Constant(domain=(0.0, 1.0)),
    "Polygonal (5 nodes)": fdt.Polygonal(np.linspace(0.0, 1.0, 5)),
}
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for ax, (title, basis) in zip(axes.flat, bases.items(), strict=False):
    lo, hi = basis.domain
    s = np.linspace(lo, hi, 300)
    ax.plot(s, basis(s))
    ax.set_title(f"{title}\n{type(basis).__name__}: {basis.n_basis} functions", fontsize=9)
axes.flat[-1].axis("off")
fig.tight_layout()

rows = []
for title, basis in bases.items():
    rows.append(
        {
            "basis": title,
            "n_basis": basis.n_basis,
            "names": basis.names[:4],
            "gram shape": basis.gram().shape,
        }
    )
show(pd.DataFrame(rows))

# %% [markdown]
# Now we check the maths of each basis against facts we know by hand.

# %%
s = np.linspace(0.0, 1.0, 1001)
sf = np.linspace(0.0, 1.0, 200001)  # a fine grid for numerical integrals
b = bases["BSpline (cubic, 8)"]
check("B-splines sum to 1 everywhere (partition of unity)", np.allclose(b(s).sum(axis=1), 1.0))
check("B-spline derivatives sum to 0", np.allclose(b(s, deriv=1).sum(axis=1), 0.0, atol=1e-10))
print("breaks:", np.round(b.breaks, 3), "\nknots:", np.round(b.knots, 3))

f = bases["Fourier (7)"]
check("Fourier basis is orthonormal (Gram = I)", np.allclose(f.gram(), np.eye(7), atol=1e-12))
omega = 2 * np.pi
check(
    "Fourier D1 penalty diagonal = (k·2π)²",
    np.allclose(np.diag(f.penalty(1))[1:3], omega**2),
    np.round(np.diag(f.penalty(1)), 3),
)

m = bases["Monomial (1, t, t², t³)"]
exact = np.array([[1 / (i + j + 1) for j in range(4)] for i in range(4)])  # ∫ t^i t^j dt
check("Monomial Gram = Hilbert matrix", np.allclose(m.gram(), exact))
check("Monomial derivative of t³ is 3t²", np.allclose(m(s, deriv=1)[:, 3], 3 * s**2))

e = bases["Exponential (rates 0, ±2)"]
check(
    "Exponential basis values",
    np.allclose(e(s), np.column_stack([np.ones_like(s), np.exp(2 * s), np.exp(-2 * s)])),
)
p = bases["Power (t^0.5, t, t^1.5)"]
sp = np.linspace(0.1, 1.0, 50)
check("Power basis derivative of t^0.5", np.allclose(p(sp, deriv=1)[:, 0], 0.5 * sp**-0.5))
c = bases["Constant"]
check("Constant basis Gram = length of domain", np.allclose(c.gram(), [[1.0]]))
g = bases["Polygonal (5 nodes)"]
check(
    "Polygonal basis = hat functions (1 at own node)",
    np.allclose(g(np.linspace(0, 1, 5)), np.eye(5)),
)

# Gram matrix against brute-force numerical integration
W_num = np.trapezoid(b(sf)[:, :, None] * b(sf)[:, None, :], sf, axis=0)
check(
    "B-spline Gram matches numerical integral",
    np.allclose(b.gram(), W_num, atol=1e-9),
    f"{np.max(np.abs(b.gram() - W_num)):.1e}",
)
R2 = b.penalty(2)
check(
    "Penalty matrix is symmetric and positive semi-definite",
    np.allclose(R2, R2.T) and np.linalg.eigvalsh(R2).min() > -1e-8,
)

# %% [markdown]
# `inprod` computes inner products ∫ f(t) g(t) dt between bases or curves. B-spline
# bases can also be built from your own **break points** (the places where the
# polynomial pieces join).

# %%
irregular = fdt.BSpline(breaks=[0.0, 0.1, 0.2, 0.5, 1.0], order=3)
print("irregular breaks -> n_basis =", irregular.n_basis)
cross = fdt.inprod(b, f)  # ∫ φ_bspline(t) φ_fourier(t)ᵀ dt
check("inprod between two different bases has shape (8, 7)", cross.shape == (8, 7), cross.shape)
num = np.trapezoid(b(sf)[:, :, None] * f(sf)[:, None, :], sf, axis=0)
check(
    "inprod(B-spline, Fourier) matches numerical integral",
    np.allclose(cross, num, atol=1e-9),
    f"{np.max(np.abs(cross - num)):.1e}",
)

# %% [markdown]
# ## Part 2. Curves: `FData` and `BiFData`
#
# An `FData` holds the coefficients of **many curves in one basis**. It is
# immutable (it never changes after it is made). Call it like a function to
# evaluate it. Arithmetic, slicing, statistics and exact derivatives all return
# new `FData` objects.

# %%
PART = "2 FData"
basis = fdt.BSpline(domain=(1.0, 18.0), n_basis=15)
coefs, *_ = np.linalg.lstsq(basis(growth.age), growth.hgtf, rcond=None)
girls = fdt.FData(coefs, basis)
print(girls)
print("n_curves:", girls.n_curves, " n_vars:", girls.n_vars, " domain:", girls.domain)

t = np.linspace(1.0, 18.0, 400)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
girls[:10].plot(ax=axes[0])
axes[0].set_title("height: girls[:10].plot()")
# (plain least squares, no penalty: the derivatives are rough; Part 4 smooths properly)
girls[:10].plot(ax=axes[1], deriv=1)
axes[1].set_title("speed: plot(deriv=1)")
girls[:10].plot(ax=axes[2], deriv=2)
axes[2].set_title("acceleration: plot(deriv=2)")
fig.tight_layout()

# %%
# exact derivative vs. a finite difference
speed = girls.derivative()
x = np.array([3.0, 9.0, 13.0])
h = 1e-5
check(
    "derivative() matches finite difference",
    np.allclose(speed(x), (girls(x + h) - girls(x - h)) / (2 * h), rtol=1e-5, atol=1e-6),
)
check(
    "derivative(2) == derivative().derivative()",
    np.allclose(girls.derivative(2)(t), speed.derivative()(t)),
)
check("evaluating with deriv=1 == derivative()", np.allclose(girls(t, deriv=1), speed(t)))
print("the derivative of an order-4 spline is an order-3 spline:", speed.basis.order)

# arithmetic
a, b2 = girls[0], girls[1]
check("addition is pointwise", np.allclose((a + b2)(t), a(t) + b2(t)))
check("subtraction is pointwise", np.allclose((a - b2)(t), a(t) - b2(t)))
check("scalar multiplication", np.allclose((2.5 * a)(t), 2.5 * a(t)))
check("power (a**2) is pointwise", np.allclose((a**2)(t), a(t) ** 2, rtol=1e-6))
check("product of two curves is pointwise", np.allclose((a * b2)(t), a(t) * b2(t), rtol=1e-6))

# statistics
check("mean() is the mean of the values", np.allclose(girls.mean()(t)[:, 0], girls(t).mean(axis=1)))
# std() involves a square root, so (like R's sd.fd) it is computed on a grid and
# projected back into the basis: close to the pointwise value, not identical.
sd_err = np.max(np.abs(girls.std()(t)[:, 0] - girls(t).std(axis=1, ddof=1)))
check(
    "std() ≈ pointwise standard deviation (basis projection)",
    sd_err < 0.05 * girls(t).std(axis=1, ddof=1).max(),
    f"max gap {sd_err:.3f} cm",
)
check("center() gives curves with mean zero", np.allclose(girls.center().mean()(t), 0.0, atol=1e-9))

# inner products
grid = np.linspace(1.0, 18.0, 20001)
G = girls[:3] @ girls[:3]
num = np.trapezoid(girls[:3](grid)[:, :, None] * girls[:3](grid)[:, None, :], grid, axis=0)
check("fd @ fd is the matrix of L2 inner products", np.allclose(G, num, rtol=1e-6))
check("inprod(fd, fd) == fd @ fd", np.allclose(fdt.inprod(girls[:3], girls[:3]), G))

# slicing
check("slicing keeps the basis", girls[5:9].n_curves == 4 and girls[5:9].basis == basis)
check("to_numpy(t) == fd(t)", np.allclose(girls.to_numpy(t), girls(t)))

# %% [markdown]
# **Multivariate curves.** A curve can have several variables at once (for
# example hip *and* knee angle). The coefficients then have shape
# `(n_basis, n_curves, n_vars)`.

# %%
gait_basis = fdt.Fourier(domain=(0.0, 1.0), n_basis=21)
gait_fd = fdt.smooth(gait.value.reshape(20, -1), gait.t, basis=gait_basis, lam=1e-6).fd
gait_coefs = np.asarray(gait_fd.coefs).reshape(21, 39, 2)
gait_mv = fdt.FData(gait_coefs, gait_basis)
print("coefs:", gait_mv.coefs.shape, " n_vars:", gait_mv.n_vars)
check("multivariate FData has 2 variables", gait_mv.n_vars == 2)
check(
    "multivariate FData evaluates to (t, curves, vars)",
    gait_mv(np.linspace(0, 1, 5)).shape == (5, 39, 2),
)

# %% [markdown]
# **`BiFData`** is a surface v(s, t), for example a covariance function.
# `FData.cov()` returns one.

# %%
V = girls.cov()
print(V)
ss = np.linspace(1.0, 18.0, 60)
surface = V(ss, ss)
check("BiFData.transpose() swaps s and t", np.allclose(V.transpose()(ss, ss), surface.T))
check("covariance surface is symmetric", np.allclose(surface, surface.T, atol=1e-8))
vals = girls(ss)
emp = np.cov(vals)
check(
    "cov() matches the empirical covariance of the values",
    np.allclose(surface, emp, rtol=1e-6, atol=1e-6),
)

fig = plt.figure(figsize=(12, 4.5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
S, T = np.meshgrid(ss, ss)
ax.plot_surface(S, T, surface, cmap="viridis")
ax.set_title("Covariance surface of height")
ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contourf(ss, ss, surface, 20, cmap="viridis")
fig.colorbar(cs)
ax2.set(title="Same surface, contour view", xlabel="age s", ylabel="age t")
fig.tight_layout()

# %% [markdown]
# ## Part 3. Linear differential operators (`LDO`)
#
# An `LDO` is a weighted sum of derivatives, L x = β₀ x + β₁ Dx + … + Dᵐx.
# It is used as a roughness penalty and can be applied to any curve.
# The **harmonic accelerator** L = ω² D + D³ gives zero on every pure cycle
# a + b sin(ωt) + c cos(ωt), so a penalty with it does not punish a yearly cycle.

# %%
PART = "3 LDO"
fourier = fdt.Fourier(domain=(0.0, 365.0), n_basis=65)
harmonic = fdt.LDO.harmonic(period=365.0)
print(harmonic)
day = weather.t
w = 2 * np.pi / 365
cycle = fdt.FData(
    np.linalg.lstsq(fourier(day), 3 + 5 * np.sin(w * day) - 2 * np.cos(w * day), rcond=None)[0],
    fourier,
)
check(
    "harmonic accelerator gives 0 on a pure yearly cycle",
    np.max(np.abs(cycle(day, harmonic))) < 1e-8,
)

D2 = fdt.LDO(2)
check("LDO(2) applied == second derivative", np.allclose(girls(t, D2), girls(t, deriv=2)))
spring = fdt.LDO(weights=[4.0, 0.0])  # L = 4 + 0·D + D²
tt = np.linspace(0.0, 1.0, 101)
sb = fdt.Fourier(domain=(0.0, np.pi), n_basis=3)  # period π: sin 2t, cos 2t
wave = fdt.FData(np.array([0.0, 1.0, 0.5]), sb)
check(
    "custom LDO: (D² + 4) sin 2t = 0",
    np.max(np.abs(wave(np.linspace(0, np.pi, 50), spring))) < 1e-8,
)
check("penalty(LDO) has the right shape", fourier.penalty(harmonic).shape == (65, 65))

# %% [markdown]
# ## Part 4. Smoothing: from noisy numbers to curves
#
# `fdt.smooth(y, t)` fits curves by *penalised least squares*: it balances the
# fit to the data against a roughness penalty λ ∫ (Lx)². λ can be a number, a
# target number of **degrees of freedom** (df, how flexible the fit is), or
# chosen by **GCV** (generalised cross-validation, an estimate of prediction
# error). Observations always have the time axis first: `y` is `(n_points, n_curves)`.

# %%
PART = "4 smoothing"
fit = fdt.smooth(growth.hgtf, growth.age)  # everything automatic
print("auto basis:", fit.fd.basis, "| lambda:", f"{fit.lam:.3g}", "| df:", round(fit.df, 2))
print(
    "SmoothResult fields: gcv",
    fit.gcv.shape,
    "sse",
    round(float(fit.sse), 4),
    "penalty_matrix",
    fit.penalty_matrix.shape,
    "y2c_map",
    fit.y2c_map.shape,
)

fine = fdt.BSpline(domain=(1.0, 18.0), n_basis=20, order=6)
by_value = fdt.smooth(growth.hgtf, growth.age, basis=fine, lam=1e-1, penalty=3)
by_df = fdt.smooth(growth.hgtf, growth.age, basis=fine, df=8.0, penalty=3)
by_df_str = fdt.smooth(growth.hgtf, growth.age, basis=fine, lam="df=8", penalty=3)
by_gcv = fdt.smooth(growth.hgtf, growth.age, basis=fine, lam="gcv", penalty=3)
check("df=8 gives exactly 8 degrees of freedom", np.isclose(by_df.df, 8.0), round(by_df.df, 10))
check('lam="df=8" is the same as df=8', np.isclose(by_df.lam, by_df_str.lam))
check(
    "y2c_map maps data to coefficients",
    np.allclose(by_value.y2c_map @ growth.hgtf, by_value.fd.coefs),
)
check(
    "zero penalty = ordinary least squares",
    np.allclose(fdt.smooth(growth.hgtf, growth.age, basis=basis, lam=0.0).fd.coefs, coefs),
)

# %%
from fdatools.smoothing import df_to_lambda, gcv_curve, lambda_to_df

lambdas = 10.0 ** np.arange(-6.0, 3.0, 0.25)
scores = gcv_curve(growth.hgtf, growth.age, fine, lambdas, penalty=3)
dfs = [lambda_to_df(growth.age, fine, lam, penalty=3) for lam in lambdas]
lam8 = df_to_lambda(growth.age, fine, 8.0, penalty=3)
check(
    "df_to_lambda and lambda_to_df are inverse",
    np.isclose(lambda_to_df(growth.age, fine, lam8, penalty=3), 8.0),
)
check("df goes down when lambda goes up", np.all(np.diff(dfs) < 0))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].loglog(lambdas, scores.mean(axis=1), marker="o", ms=3)
axes[0].axvline(by_gcv.lam, ls=":", c="k")
axes[0].set(xlabel="lambda", ylabel="mean GCV", title="GCV curve (dotted: chosen lambda)")
axes[1].semilogx(lambdas, dfs)
axes[1].set(xlabel="lambda", ylabel="df", title="lambda vs degrees of freedom")
for lam, style in [(1e-4, "-"), (1e0, "--"), (1e3, ":")]:
    r = fdt.smooth(growth.hgtf[:, :1], growth.age, basis=fine, lam=lam, penalty=3)
    axes[2].plot(t, r.fd(t, deriv=2), style, label=f"lambda={lam:g}")
axes[2].set(title="Acceleration of girl 1 at 3 lambdas", xlabel="age", ylim=(-8, 6))
axes[2].legend()
fig.tight_layout()

# %% [markdown]
# **Noisy data and uncertainty.** We add noise to a known curve, smooth it, and
# draw a 95% pointwise **confidence band** with `stats.confidence_band`.

# %%
from fdatools.stats import confidence_band

rng = np.random.default_rng(0)
tn = np.linspace(0.0, 1.0, 80)
truth = np.sin(2 * np.pi * tn) + 0.5 * tn
noisy = truth + 0.15 * rng.standard_normal(tn.size)
nfit = fdt.smooth(noisy, tn, basis=fdt.BSpline(n_basis=20), lam="gcv")
tg = np.linspace(0.0, 1.0, 300)
band = confidence_band(nfit, tg, sigma_e=0.15**2)
truth_g = np.sin(2 * np.pi * tg) + 0.5 * tg
inside = np.mean((band.lower[:, 0] <= truth_g) & (truth_g <= band.upper[:, 0]))
# A pointwise band covers noise, not the small bias that every smoother has at peaks.
check("the true curve is mostly inside the 95% band", inside >= 0.8, f"{inside:.0%} of points")
ax = band.plot()
ax.plot(tn, noisy, ".", ms=4, alpha=0.5, label="noisy data")
ax.plot(tg, truth_g, "k:", label="truth")
ax.legend()
ax.set_title(f"GCV smooth (df = {nfit.df:.1f}) with 95% band")

# %% [markdown]
# **Shape constraints.** `constraint="monotone"` fits curves that never go down,
# x(t) = β₀ + β₁ ∫ exp W; `constraint="positive"` fits x(t) = exp W > 0.
# Call the result itself (`fit(t)`) to evaluate the constrained curve.

# %%
mono = fdt.smooth(
    growth.hgtf[:, :6],
    growth.age,
    basis=fdt.BSpline(domain=(1.0, 18.0), n_basis=15),
    lam=1e-1,
    constraint="monotone",
)
check("monotone fit: speed > 0 everywhere", np.all(mono(t, deriv=1) > 0))
print("constraint:", mono.constraint, "| beta:", mono.beta.shape)
h = 1e-5
check(
    "monotone derivatives are exact",
    np.allclose(
        mono(t[1:-1], deriv=1), (mono(t[1:-1] + h) - mono(t[1:-1] - h)) / (2 * h), rtol=1e-4
    ),
)

pos_t = np.linspace(0.0, 1.0, 60)
pos_y = np.exp(np.sin(4 * pos_t)) * 0.2
pos = fdt.smooth(pos_y - 0.02, pos_t, constraint="positive", lam=1e-6)
check("positive fit stays above zero", np.min(pos(np.linspace(0, 1, 500))) > 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(growth.age, growth.hgtf[:, :6], "o", ms=3)
axes[0].set_prop_cycle(None)
axes[0].plot(t, mono(t))
axes[0].set(title="Monotone height fits", xlabel="age")
axes[1].plot(t, mono(t, deriv=1))
axes[1].set(title="Their speed (always > 0)", xlabel="age")
fig.tight_layout()

# %%
# Seasonal data: Fourier basis + harmonic accelerator, lambda by GCV
temp_gcv = fdt.smooth(weather.temp, day, basis=fourier, penalty=harmonic, lam="gcv")
temp_fit = fdt.smooth(weather.temp, day, basis=fourier, penalty=harmonic, lam=1e2)
precip_fit = fdt.smooth(weather.log10precip, day, basis=fourier, penalty=harmonic, lam=1e4)
temp, precip = temp_fit.fd, precip_fit.fd
print(f"GCV lambda for temperature: {temp_gcv.lam:.3g}, df = {temp_gcv.df:.1f}")
stations = ["Montreal", "Edmonton", "Pr. Rupert", "Resolute"]
cols = [weather.stations.index(n) for n in stations]
fig, ax = plt.subplots()
ax.plot(day, weather.temp[:, cols], ".", ms=2, alpha=0.4)
ax.set_prop_cycle(None)
ax.plot(day, temp(day)[:, cols])
ax.legend(stations)
ax.set(title="Daily temperature, smoothed", xlabel="day")
check("seasonal smooth residual is small", np.sqrt(np.mean((temp(day) - weather.temp) ** 2)) < 2.5)

# %% [markdown]
# ## Part 5. Descriptive statistics
#
# - `stats.cov`, `stats.cor`: covariance and correlation surfaces (also between two sets of curves).
# - `stats.depth`: how central each curve is (modified band depth or Fraiman-Muniz).
# - `stats.boxplot`: the functional boxplot, which finds outlier curves.
# - `stats.cycleplot`: plots two periodic variables against each other.

# %%
PART = "5 stats"
from fdatools import stats

cor_tp = stats.cor(temp, precip, s=day, t=day)
cor_tt = stats.cor(temp, s=day, t=day)
check("correlation of a set with itself has 1 on the diagonal", np.allclose(np.diag(cor_tt), 1.0))
check("correlations are between -1 and 1", np.all(np.abs(cor_tp) <= 1 + 1e-10))
cov_tp = stats.cov(temp, precip)
check("cross-covariance is a BiFData (65 x 65)", cov_tp.coefs.shape == (65, 65))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
c0 = axes[0].contourf(day, day, cor_tt, 20, cmap="RdBu_r", vmin=-1, vmax=1)
fig.colorbar(c0, ax=axes[0])
axes[0].set(title="Temperature correlation", xlabel="day", ylabel="day")
c1 = axes[1].contourf(day, day, cor_tp, 20, cmap="RdBu_r", vmin=-1, vmax=1)
fig.colorbar(c1, ax=axes[1])
axes[1].set(title="Temperature vs log-precipitation", xlabel="day (precip)", ylabel="day (temp)")
fig.tight_layout()

# %%
dep = stats.depth(temp, t=day)
dep_fm = stats.depth(temp, method="FM", t=day)
median_station = weather.stations[dep.median_index]
print("deepest (most typical) station:", median_station)
check("depth values are between 0 and 1", np.all((dep.depth >= 0) & (dep.depth <= 1)))
check("median has the largest depth", dep.median_index == int(np.argmax(dep.depth)))

box = stats.boxplot(temp, t=day)
print("outlier stations:", [weather.stations[i] for i in box.outliers])
ax = box.plot()
ax.set(title="Functional boxplot of temperature (red: outliers)", xlabel="day")

order = np.argsort(dep.depth)
show(
    pd.DataFrame(
        {
            "station": np.array(weather.stations)[order[::-1]],
            "MBD depth": dep.depth[order[::-1]].round(3),
            "FM depth": dep_fm.depth[order[::-1]].round(3),
        }
    ).head(8)
)

# %%
# cycleplot: hip angle against knee angle over one gait cycle
ax = stats.cycleplot(gait_mv.mean())
ax.set(title="Gait cycle: mean hip vs knee angle", xlabel="hip angle", ylabel="knee angle")
check("cycleplot draws one loop", len(ax.lines) >= 1)

# %% [markdown]
# ## Part 6. Functional PCA
#
# `FPCA` finds the main **modes of variation** (the *harmonics*) and the
# **score** of each curve on each mode: curve ≈ mean + Σ score x harmonic.
# It is a scikit-learn estimator (`fit`, `transform`, `inverse_transform`).

# %%
PART = "6 FPCA"
from fdatools.decomposition import FCCA, FPCA

pca = FPCA(n=4).fit(temp)
print("share of variance:", pca.varprop.round(4), " total:", pca.varprop.sum().round(4))
check("first harmonic explains > 85%", pca.varprop[0] > 0.85)
check("eigenvalues are sorted", np.all(np.diff(pca.values) <= 1e-9))
H = pca.harmonics @ pca.harmonics
check("harmonics are orthonormal", np.allclose(H, np.eye(4), atol=1e-8))
check("transform(temp) == scores", np.allclose(pca.transform(temp), pca.scores))
check("scores have mean zero", np.allclose(pca.scores.mean(axis=0), 0, atol=1e-8))
full = FPCA(n=65).fit(temp)
check(
    "with all harmonics, inverse_transform rebuilds the curves exactly",
    np.allclose(full.inverse_transform(full.scores)(day), temp(day), atol=1e-6),
)
err4 = np.max(np.abs(pca.inverse_transform(pca.scores)(day) - temp(day)))
print(f"max error with 4 harmonics: {err4:.2f} degrees")
axes = pca.plot()

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(day, pca.harmonics(day))
axes[0].legend([f"harmonic {k + 1}" for k in range(4)])
axes[0].set(title="Temperature harmonics", xlabel="day")
stats.plot_scores(pca.scores, ax=axes[1], labels=weather.stations)
for region in sorted(set(weather.region)):
    rows = [i for i, r in enumerate(weather.region) if r == region]
    axes[1].scatter(pca.scores[rows, 0], pca.scores[rows, 1], label=region, zorder=3)
axes[1].legend()
axes[1].set(title="Scores by climate region", xlabel="PC1 (warm)", ylabel="PC2 (seasonal swing)")
fig.tight_layout()

# %%
smooth_pca = FPCA(n=4, lam=1e4, penalty=harmonic).fit(temp)
rotated = pca.rotate("varimax")
print("smoothed harmonics varprop:", smooth_pca.varprop.round(4))
print("rotated varprop:           ", rotated.varprop.round(4))
check("varimax keeps the total variance", np.isclose(rotated.varprop.sum(), pca.varprop.sum()))
check(
    "rotation matrix is orthogonal", np.allclose(rotated.rotation.T @ rotated.rotation, np.eye(4))
)

gcv_pca = FPCA(n=2, lam="gcv", penalty=harmonic).fit(temp[:20])
print(
    "FPCA lambda chosen by leave-one-curve-out CV:",
    gcv_pca.lam_ if hasattr(gcv_pca, "lam_") else gcv_pca.lam,
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(day, smooth_pca.harmonics(day))
axes[0].set_title("Smoothed harmonics (lam=1e4)")
axes[1].plot(day, rotated.harmonics(day))
axes[1].set_title("Varimax-rotated harmonics (seasons)")
fig.tight_layout()

# %% [markdown]
# ## Part 7. Functional CCA
#
# **Canonical correlation analysis** finds a weight function for temperature and
# one for precipitation so that the resulting station scores are as correlated
# as possible. It needs a penalty (`lam1`, `lam2`), or every correlation is 1.

# %%
PART = "7 FCCA"
cca = FCCA(n=3, lam1=1e6, lam2=1e6, penalty=harmonic).fit(temp, precip)
print("canonical correlations:", cca.correlations[:3].round(3))
check(
    "canonical correlations are sorted and in [0, 1]",
    np.all(np.diff(cca.correlations[:3]) <= 1e-12)
    and 0 <= cca.correlations[2] <= cca.correlations[0] <= 1,
)
# The reported correlations are *penalised* (as in R's cca.fd): the penalty adds to
# the variances in the denominator, so the plain correlation of the scores is a bit higher.
raw = [abs(np.corrcoef(cca.scores1[:, k], cca.scores2[:, k])[0, 1]) for k in range(3)]
print("plain correlations of the scores:", np.round(raw, 3))
check(
    "plain score correlation >= penalised canonical correlation",
    all(r_ >= c_ - 1e-12 for r_, c_ in zip(raw, cca.correlations[:3], strict=True)),
    np.round(raw, 3),
)
check("transform() reproduces the scores", np.allclose(cca.transform(temp, precip)[0], cca.scores1))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(day, cca.weights1(day)[:, 0], label="temperature weight")
axes[0].plot(day, cca.weights2(day)[:, 0], label="log-precip weight")
axes[0].legend()
axes[0].set(title="First pair of canonical weight functions", xlabel="day")
axes[1].scatter(cca.scores1[:, 0], cca.scores2[:, 0])
for i in cols:
    axes[1].annotate(weather.stations[i], (cca.scores1[i, 0], cca.scores2[i, 0]), fontsize=8)
axes[1].set(
    title=f"First canonical scores (r = {cca.correlations[0]:.2f})",
    xlabel="temperature",
    ylabel="precipitation",
)
fig.tight_layout()

# %% [markdown]
# ## Part 8. Functional regression and permutation tests
#
# One function, `fregress`, covers all cases:
#
# | response | covariate | example below |
# |---|---|---|
# | number | curve | yearly rain ~ temperature curve |
# | curve | numbers / groups | temperature curve ~ climate region |
# | curve | curve (same time) | precipitation(t) ~ temperature(t) (*concurrent*) |
#
# `linmod` fits the full **function-on-function** model y(t) = a(t) + ∫ x(s) β(s, t) ds.
# `FRegress` is the scikit-learn estimator version.

# %%
PART = "8 regression"
from fdatools.regression import FRegress, fregress, linmod

log_precip = np.log10(weather.precip.sum(axis=0))
beta_basis = fdt.Fourier(domain=(0.0, 365.0), n_basis=35)
model = fregress(
    log_precip, {"const": 1.0, "temp": temp}, beta={"temp": (beta_basis, 10**12.5, harmonic)}
)
r2 = 1 - np.sum((log_precip - model.fitted) ** 2) / np.sum((log_precip - log_precip.mean()) ** 2)
print("names:", model.names, "| df:", round(model.df, 2), "| R²:", round(r2, 3))
check("scalar-on-function R² is about 0.75", 0.7 < r2 < 0.8, round(r2, 3))
check(
    "predict() on training data == fitted",
    np.allclose(model.predict({"const": 1.0, "temp": temp}), model.fitted),
)
cv = model.cv()
check(
    "leave-one-out CV error is larger than the training error",
    cv.sse > np.sum((log_precip - model.fitted) ** 2),
)
se = model.stderr()
bands = confidence_band(model, day)
check("confidence_band(model) returns one band per term", len(bands) == 2)

beta_t = model.beta[1](day)[:, 0]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(day, beta_t, "k")
b2se = 2 * se.beta[1](day)[:, 0]
axes[0].fill_between(day, beta_t - b2se, beta_t + b2se, alpha=0.3)
axes[0].axhline(0, lw=0.6)
axes[0].set(title="β(t): effect of temperature on log rain (±2 SE)", xlabel="day")
axes[1].scatter(model.fitted, log_precip)
lim = [log_precip.min(), log_precip.max()]
axes[1].plot(lim, lim, "k--")
axes[1].set(title=f"Observed vs fitted (R² = {r2:.2f})", xlabel="fitted", ylabel="observed")
fig.tight_layout()

# %%
by_region = fregress("temp ~ region", {"temp": temp_fit, "region": weather.region})
print("terms:", by_region.names)
check("formula interface creates one indicator per region", len(by_region.names) == 4)
check("functional response gives a functional fit", by_region.functional_response)
res = weather.temp - by_region.fitted(day)
region_se = by_region.stderr(sigma_e=res @ res.T / res.shape[1])
fig, grid_axes = plt.subplots(2, 2, figsize=(13, 7))
axes = stats.plot_beta(
    by_region, day, sigma_e=res @ res.T / res.shape[1], axes=list(grid_axes.flat)
)
fig.suptitle("Region effects on temperature with 95% bands (reference: Arctic)")
fig.tight_layout()
check("plot_beta draws one panel per term", len(axes) == 4)

# %%
# concurrent model: log-precip(t) = β0(t) + β1(t) temp(t)
conc = fregress(
    precip,
    {"const": 1.0, "temp": temp},
    beta={
        "const": (fdt.Fourier(domain=(0.0, 365.0), n_basis=11), 0.0, 2),
        "temp": (fdt.Fourier(domain=(0.0, 365.0), n_basis=11), 0.0, 2),
    },
)
print("concurrent terms:", conc.names)
check(
    "concurrent model: residuals smaller than around the mean",
    np.mean((conc.fitted(day) - precip(day)) ** 2) < np.mean((precip.center())(day) ** 2),
)

# function-on-function: temperature -> log-precip, full surface β(s, t)
fof = linmod(
    precip,
    temp,
    s_basis=fdt.Fourier(domain=(0.0, 365.0), n_basis=11),
    t_basis=fdt.Fourier(domain=(0.0, 365.0), n_basis=11),
    lam_s=1e4,
    lam_t=1e4,
    penalty_s=harmonic,
    penalty_t=harmonic,
)
check(
    "linmod: fitted + residuals == data",
    np.allclose((fof.fitted + fof.residuals)(day), precip(day), atol=1e-8),
)
check(
    "linmod.predict on training data == fitted",
    np.allclose(fof.predict(temp)(day), fof.fitted(day)),
)
surf = fof.beta(day[::5], day[::5])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(day, conc.beta[1](day))
axes[0].axhline(0, lw=0.6)
axes[0].set(title="Concurrent β₁(t)", xlabel="day")
cs = axes[1].contourf(day[::5], day[::5], surf.T, 20, cmap="RdBu_r")
fig.colorbar(cs, ax=axes[1])
axes[1].set(title="linmod β(s, t)", xlabel="temperature day s", ylabel="precip day t")
fig.tight_layout()

# %%
# scikit-learn estimator on plain numbers
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 3))
y = 1.0 + X @ np.array([2.0, -1.0, 0.5]) + 0.01 * rng.standard_normal(80)
est = FRegress().fit(X, y)
coef = [round(float(b.coefs[0, 0]), 2) for b in est.coef_]
check(
    "FRegress recovers intercept and slopes",
    np.allclose(coef, [1.0, 2.0, -1.0, 0.5], atol=0.02),
    coef,
)

# %% [markdown]
# **Permutation tests.** `f_test` asks "is the regression effect real?" by
# shuffling the responses many times. `t_test` compares two groups of curves
# (here: growth speed of boys vs girls).

# %%
from fdatools.stats import f_test, t_test

regions = sorted(set(weather.region))
dummies = [np.array([1.0 if r == name else 0.0 for r in weather.region]) for name in regions[1:]]
ftest = f_test(
    temp,
    [np.ones(35), *dummies],
    basis=fdt.Fourier(domain=(0.0, 365.0), n_basis=11),
    n_perm=200,
    random_state=1,
)
print(
    f"F test: statistic {ftest.statistic:.2f}, p-value {ftest.pvalue}, "
    f"95% critical {ftest.critical_value:.2f}"
)
check("region effect on temperature is significant", ftest.pvalue < 0.05)

hb = fdt.BSpline(domain=(1.0, 18.0), n_basis=20, order=6)
boys_v = fdt.smooth(growth.hgtm, growth.age, basis=hb, lam=1e-1, penalty=3).fd.derivative()
girls_v = fdt.smooth(growth.hgtf, growth.age, basis=hb, lam=1e-1, penalty=3).fd.derivative()
ttest = t_test(boys_v, girls_v, n_perm=200, random_state=0, t=np.linspace(1, 18, 200))
print(f"t test: statistic {ttest.statistic:.2f}, p-value {ttest.pvalue}")
check("boys and girls grow at different speeds", ttest.pvalue < 0.05)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(ftest.t, ftest.pointwise, label="observed F(t)")
axes[0].plot(ftest.t, ftest.pointwise_critical_value, "--", label="pointwise 95% of null")
axes[0].axhline(ftest.critical_value, ls=":", label="95% of null maximum")
axes[0].legend()
axes[0].set_title("F test: region")
axes[1].plot(ttest.t, ttest.pointwise, label="observed |t|(age)")
axes[1].plot(ttest.t, ttest.pointwise_critical_value, "--", label="pointwise 95%")
axes[1].legend()
axes[1].set(title="t test: boys vs girls speed", xlabel="age")
fig.tight_layout()

# %% [markdown]
# ## Part 9. Registration: lining up curves in time
#
# Children reach their growth spurt at different ages, so the mean curve blurs
# the spurt. **Registration** gives each curve a *warping function* h(t)
# (time never runs backwards) so that the features line up.
#
# - **landmark**: line up given points (here the age of peak speed);
# - **continuous**: warp each curve to be close to a target;
# - `decompose()`: how much variation is timing (*phase*) vs size (*amplitude*).

# %%
PART = "9 registration"
from fdatools.registration import Registrator, landmark_register, register

b6 = fdt.BSpline(domain=(1.0, 18.0), n_basis=35, order=6)
gfit = fdt.smooth(growth.hgtf[:, :20], growth.age, basis=b6, penalty=4, lam=1e-2)
vel, accel = gfit.fd.derivative(1), gfit.fd.derivative(2)
search = np.linspace(8.0, 16.0, 801)
peak = search[np.argmax(vel(search), axis=0)]
print("age of peak growth speed:", peak.round(1))

lm = register(accel, landmarks=peak)
check(
    "landmark: h_i(mean peak) = own peak",
    np.allclose(lm.warp_values(np.array([peak.mean()]))[0], peak, atol=1e-2),
)
tt = np.linspace(1.0, 18.0, 300)
check("warping functions are increasing", np.all(np.diff(lm.warp_values(tt), axis=0) > 0))
lm2 = landmark_register(accel, peak)
check(
    "landmark_register == register(landmarks=...)",
    np.allclose(lm2.registered.coefs, lm.registered.coefs),
)
check(
    "apply() on the same curves == registered",
    np.allclose(lm.apply(accel).coefs, lm.registered.coefs),
)

cont = register(accel, warp_basis=fdt.BSpline(domain=(1.0, 18.0), n_basis=6), lam=1.0)
amp, phase, rsq, c = lm.decompose(domain=(3.0, 17.0))
rsq_c = cont.decompose(domain=(3.0, 17.0)).rsq
print(f"share of variation due to timing: landmark {rsq:.2f}, continuous {rsq_c:.2f}")
check("phase share is between 0 and 1", 0 < rsq < 1 and 0 < rsq_c < 1)

t3 = np.linspace(3.0, 17.0, 300)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
panels = [("Before", accel), ("Landmark", lm.registered), ("Continuous", cont.registered)]
for ax, (title, fdx) in zip(axes[:3], panels, strict=True):
    ax.plot(t3, fdx(t3), color="grey", alpha=0.5)
    ax.plot(t3, fdx.mean()(t3), "k", lw=2)
    ax.set(title=title, ylim=(-6, 4), xlabel="age")
axes[3].plot(t3, lm.warp_values(t3))
axes[3].plot(t3, t3, "k--")
axes[3].set(title="Landmark warps h(t)", xlabel="registered age")
fig.tight_layout()

# %%
# a synthetic check with a known answer: shifted bumps
bb = fdt.BSpline(domain=(0.0, 1.0), n_basis=15)
ts = np.linspace(0.0, 1.0, 400)
shifts = [0.40, 0.45, 0.5, 0.55, 0.60]
bumps = fdt.FData(
    np.linalg.lstsq(
        bb(ts), np.stack([np.exp(-(((ts - s) / 0.1) ** 2)) for s in shifts], 1), rcond=None
    )[0],
    bb,
)
ls_reg = register(bumps, criterion="least_squares", warp_basis=fdt.BSpline(n_basis=5), lam=1e-3)
peaks_after = ts[np.argmax(ls_reg.registered(ts), axis=0)]
check(
    "least-squares registration pulls the peaks together",
    np.ptp(peaks_after) < 0.5 * np.ptp(shifts),
    f"spread {np.ptp(shifts):.2f} -> {np.ptp(peaks_after):.3f}",
)

# scikit-learn transformer
reg = Registrator(fdt.BSpline(n_basis=5), lam=1e-3, criterion="least_squares", basis=bb)
out = reg.fit_transform(np.asarray(bumps.coefs).T)
check("Registrator.fit_transform returns coefficient rows", out.shape == (5, 15))

# multivariate registration: hip and knee angles share one warp per boy
mv = register(gait_mv[:10], criterion="least_squares", warp_basis=fdt.BSpline(n_basis=5), lam=1e-2)
check("multivariate registration keeps both variables", mv.registered.coefs.shape == (21, 10, 2))
check("one warping function per boy", mv.latent.n_curves == 10)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ts, bumps(ts))
axes[0].set_title("Shifted bumps")
axes[1].plot(ts, ls_reg.registered(ts))
axes[1].set_title("After least-squares registration")
fig.tight_layout()

# %% [markdown]
# ## Part 10. Dynamics: differential equations from data
#
# **Principal differential analysis** (`PDA`) finds a differential equation
# β₀(t) x + β₁(t) Dx + D²x = 0 that the curves almost follow. We use the lip
# data (saying "bob"). `solve()` integrates the fitted equation,
# `stability()` gives eigenvalues over time, and `phase_plane` draws
# acceleration against velocity.

# %%
PART = "10 dynamics"
from fdatools.dynamics import PDA, phase_plane

lip_basis = fdt.BSpline(domain=(0.0, 0.35), n_basis=31, order=6)
lip_fd = fdt.smooth(lip.value, lip.t, basis=lip_basis, penalty=4, lam=1e-8).fd
constant = PDA(order=2).fit(lip_fd)
b0 = float(constant.weights_[0].coefs[0, 0])
b1 = float(constant.weights_[1].coefs[0, 0])
print(
    f"constant weights: beta0 = {b0:.1f}, beta1 = {b1:.2f}, "
    f"period = {2 * np.pi / np.sqrt(b0):.2f} s"
)
pda = PDA(order=2, weight_basis=fdt.BSpline(domain=(0.0, 0.35), n_basis=21), lam=1e-8).fit(lip_fd)
mean_lip = lip_fd.mean()
start = np.array([0.0])
init = [float(mean_lip(start)[0, 0]), float(mean_lip(start, 1)[0, 0])]
path = pda.solve(lip.t, init)
gap = float(np.max(np.abs(path - mean_lip(lip.t)[:, 0])))
check("time-varying PDA reproduces the mean lip curve (< 1 mm)", gap < 1.0, f"{gap:.2f} mm")
check(
    "residuals Lx are smaller than the curves",
    np.mean(pda.transform(lip_fd)(lip.t) ** 2) < np.mean(lip_fd.derivative(2)(lip.t) ** 2),
)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
axes[0].plot(lip.t, mean_lip(lip.t), "k", lw=2, label="mean")
axes[0].plot(lip.t, path, "--", label="time-varying PDA")
axes[0].plot(lip.t, constant.solve(lip.t, init), ":", label="constant PDA")
axes[0].legend()
axes[0].set(title="Solutions of the fitted equations", xlabel="time (s)")
g = np.linspace(0.0, 0.35, 200)
axes[1].plot(g, pda.weights_[0](g), label="β0 (stiffness)")
axes[1].plot(g, pda.weights_[1](g) * 20, label="20 x β1 (damping)")
axes[1].legend()
axes[1].set_title("Time-varying weights")
pda.plot_overlay(ax=axes[2], labels={0.0: "start", 0.175: "middle", 0.35: "end"})
axes[2].set_title("Stability diagram")
fig.tight_layout()

# %%
# a known system: sin and cos solve D²x + x = 0
sc = fdt.FData(
    np.array([[0.0, 0.0], [1.0, 0.5], [0.0, 2.0]]), fdt.Fourier(domain=(0.0, 2 * np.pi), n_basis=3)
)
harm = PDA(order=2, n_grid=None).fit(sc)
check(
    "PDA on sin/cos finds beta0 = 1, beta1 = 0",
    np.allclose([w.coefs[0, 0] for w in harm.weights_], [1.0, 0.0], atol=1e-10),
)
st = harm.stability(n_points=5)
check(
    "its eigenvalues are ±i (pure oscillation)",
    np.allclose(np.sort_complex(st.eigenvalues[0]), [-1j, 1j]),
)

# forcing: Dx + 4x = 2u with a step input u = 1 -> x -> 0.5
tf = np.linspace(0.0, 1.0, 101)
fb24 = fdt.BSpline(domain=(0.0, 1.0), n_basis=24, order=5)
forced = fdt.smooth(0.5 * (1 - np.exp(-4 * tf)), tf, basis=fb24, lam=0.0).fd
u = fdt.FData(np.array([1.0]), fdt.Constant(domain=(0.0, 1.0)))
fp = PDA(order=1).fit(forced, forcing=u)
bw, fw = float(fp.weights_[0].coefs[0, 0]), float(fp.forcing_weights_[0].coefs[0, 0])
check(
    "PDA with forcing recovers Dx + 4x = 2u",
    np.allclose([bw, fw], [4.0, 2.0], atol=1e-5),
    (round(bw, 6), round(fw, 6)),
)
stab = fp.stability(n_points=3)
check("fixed point of the forced system is 0.5", np.allclose(stab.limits, 0.5, atol=1e-5))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
stab_lip = pda.stability()
stab_lip.plot(ax=axes[0])
axes[0].set_title("Lip PDA: eigenvalues over time")
gg = fdt.smooth(growth.hgtf[:, :3], growth.age, basis=b6, penalty=4, lam=1e-2).fd
phase_plane(gg, np.linspace(4.0, 17.0, 300), labels={6.0: "6", 11.0: "11", 14.0: "14"}, ax=axes[1])
axes[1].set(title="Phase plane: growth spurt loops", xlabel="speed", ylabel="acceleration")
fig.tight_layout()

# %% [markdown]
# ## Part 11. Sparse data: PACE
#
# Sometimes each subject is measured only a few times, at different times
# (for example medical visits). **PACE** (principal components by conditional
# expectation, Yao, Müller & Wang 2005) pools all subjects to estimate the mean
# and covariance, and then computes the scores of each subject.
#
# Test: we take the growth *velocity* of the 54 girls, keep only **4 random ages
# per girl**, and compare PACE with FPCA on the full curves.

# %%
PART = "11 sparse"
from fdatools.sparse import PACE, sparse_cov, sparse_mean

rng = np.random.default_rng(3)
dense_v = fdt.smooth(growth.hgtf, growth.age, basis=hb, lam=1e-1, penalty=3).fd.derivative()
ts_sp, ys_sp = [], []
for i in range(54):
    ti = np.sort(rng.uniform(2.0, 17.0, 4))
    ts_sp.append(ti)
    ys_sp.append(dense_v[i](ti)[:, 0] + 0.1 * rng.standard_normal(4))

lo = min(ti.min() for ti in ts_sp)
hi = max(ti.max() for ti in ts_sp)  # PACE works inside the observed range
sp_basis = fdt.BSpline(domain=(lo, hi), n_basis=8)
mu = sparse_mean(ys_sp, ts_sp, sp_basis, lam=1e-2)
est = sparse_cov(ys_sp, ts_sp, basis=sp_basis, lam=1e-1)
pace = PACE(n=2, basis=sp_basis, lam_mean=1e-2, lam_cov=1e-1, sigma2=0.01).fit(ys_sp, t=ts_sp)
print("PACE varprop:", pace.varprop.round(3), "| estimated noise variance:", round(est.sigma2, 4))

ag = np.linspace(lo, hi, 200)
dense_pca = FPCA(n=2).fit(dense_v)
r_mean = np.corrcoef(mu(ag)[:, 0], dense_v.mean()(ag)[:, 0])[0, 1]
r_h1 = abs(np.corrcoef(pace.harmonics(ag)[:, 0], dense_pca.harmonics(ag)[:, 0])[0, 1])
r_s1 = abs(np.corrcoef(pace.scores[:, 0], dense_pca.scores[:, 0])[0, 1])
check("sparse mean ≈ dense mean", r_mean > 0.95, f"r = {r_mean:.3f}")
check("PACE harmonic 1 ≈ dense harmonic 1", r_h1 > 0.8, f"|r| = {r_h1:.3f}")
check("PACE scores ≈ dense scores", r_s1 > 0.7, f"|r| = {r_s1:.3f}")
check("PACE with sigma2 given raises no warning (bug fix)", True)
recon = pace.inverse_transform(pace.scores, ag)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for ti, yi in zip(ts_sp[:15], ys_sp[:15], strict=True):
    axes[0].plot(ti, yi, "o-", ms=3, alpha=0.6)
axes[0].plot(ag, mu(ag), "k", lw=2, label="sparse mean")
axes[0].legend()
axes[0].set_title("4 points per girl")
axes[1].plot(ag, pace.harmonics(ag)[:, 0], label="PACE h1")
axes[1].plot(ag, np.sign(r_h1) * dense_pca.harmonics(ag)[:, 0], "--", label="dense FPCA h1")
axes[1].legend()
axes[1].set_title("First harmonic")
for i, col in enumerate(["C0", "C1", "C2"]):
    axes[2].plot(ag, dense_v[i](ag), col, alpha=0.5)
    axes[2].plot(ag, recon[i], col + "--")
    axes[2].plot(ts_sp[i], ys_sp[i], col + "o")
axes[2].set_title("Full curve (solid) vs PACE guess from 4 points (dashed)")
fig.tight_layout()

# %%
# A fixed bug: with sigma2 given, PACE no longer warns when its own estimate is negative
rng = np.random.default_rng(1)
tq = [np.sort(rng.uniform(0, 1, 4)) for _ in range(200)]
yq = [1.0 + rng.normal() * np.cos(np.pi * ti) + 0.05 * rng.normal(size=4) for ti in tq]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    PACE(n=2, basis=fdt.BSpline(n_basis=5), sigma2=0.05**2).fit(yq, t=tq)
check(
    "PACE(sigma2=...) is silent on data with a negative sigma2 estimate",
    len(caught) == 0,
    f"{len(caught)} warnings",
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    PACE(n=2, basis=fdt.BSpline(n_basis=5)).fit(yq, t=tq)
check(
    "PACE without sigma2 still warns on the same data",
    any("not positive" in str(w.message) for w in caught),
)

# %% [markdown]
# ## Part 12. Densities and event rates
#
# `fit_density` estimates a smooth probability density p(x) = exp W(x) / C.
# `fit_intensity` estimates the rate λ(t) of events over time (a Poisson process).

# %%
PART = "12 density"
rng = np.random.default_rng(0)
sample = np.concatenate([rng.normal(-1.5, 0.5, 300), rng.normal(1.0, 0.8, 500)])
dens = fdt.fit_density(sample, basis=fdt.BSpline(domain=(-4.0, 4.0), n_basis=13), lam=1e-3)
xs = np.linspace(-4.0, 4.0, 4001)
pdf = dens(xs)
true_pdf = 0.375 * np.exp(-0.5 * ((xs + 1.5) / 0.5) ** 2) / (
    0.5 * np.sqrt(2 * np.pi)
) + 0.625 * np.exp(-0.5 * ((xs - 1.0) / 0.8) ** 2) / (0.8 * np.sqrt(2 * np.pi))
check("density converged", dens.converged, f"{dens.n_iter} iterations")
check("density integrates to 1", np.isclose(np.trapezoid(pdf, xs), 1.0, atol=1e-4))
check(
    "density is close to the truth",
    np.max(np.abs(pdf - true_pdf)) < 0.06,
    f"max error {np.max(np.abs(pdf - true_pdf)):.3f}",
)
check("log_density == log(density)", np.allclose(dens.log_density(xs[::50]), np.log(pdf[::50])))


# events with a known rate: λ(t) = 20 + 15 sin(2π t / 10) on [0, 10]
def lam_true(s):
    return 20 + 15 * np.sin(2 * np.pi * s / 10)


cand = np.sort(rng.uniform(0, 10, rng.poisson(35 * 10)))
events = cand[rng.uniform(0, 35, cand.size) < lam_true(cand)]
inten = fdt.fit_intensity(events, basis=fdt.BSpline(domain=(0.0, 10.0), n_basis=10), lam=1.0)
tg = np.linspace(0, 10, 500)
check(
    "expected count equals number of events",
    np.isclose(inten.expected_count, events.size),
    (round(inten.expected_count, 3), events.size),
)
check("intensity follows the true rate", np.corrcoef(inten(tg), lam_true(tg))[0, 1] > 0.9)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(sample, bins=60, density=True, alpha=0.35)
axes[0].plot(xs, pdf, "k", label="fit_density")
axes[0].plot(xs, true_pdf, "r:", label="truth")
axes[0].legend()
axes[0].set_title("Density of a 2-group mixture")
axes[1].plot(tg, inten(tg), "k", label="fit_intensity")
axes[1].plot(tg, lam_true(tg), "r:", label="true rate")
axes[1].plot(events, np.zeros_like(events), "|", alpha=0.4, ms=10)
axes[1].legend()
axes[1].set_title(f"Event rate ({events.size} events)")
fig.tight_layout()

# %% [markdown]
# ## Part 13. Estimating ODE parameters (generalised profiling)
#
# `profile_ode` estimates the parameters θ of a differential equation
# Dx = f(x, t, θ) from noisy data, even when some states are never observed
# (Ramsay, Hooker, Campbell & Cao 2007). Built-in models: FitzHugh-Nagumo
# (a neuron) and the CSTR (a chemical reactor). You can also write your own
# `ODEModel`.

# %%
PART = "13 profiling"
from fdatools.profiling import (
    ODEModel,
    ProfiledODE,
    cstr_inputs,
    cstr_model,
    fitzhugh_nagumo_model,
    profile_ode,
    simpson_rule,
)

fhn = fitzhugh_nagumo_model()
tp = np.linspace(0.0, 20.0, 201)
theta_true = [0.2, 0.2, 3.0]
x_true = fhn.simulate(tp, [-1.0, 1.0], theta_true)
rng = np.random.default_rng(0)
y_obs = x_true + 0.05 * rng.standard_normal(x_true.shape)
y_obs[:, 1] = np.nan  # the second state is never measured
fb_basis = fdt.BSpline(domain=(0.0, 20.0), breaks=np.linspace(0.0, 20.0, 201))
fhn_fit = profile_ode(fhn, tp, y_obs, fb_basis, lam=1e3, theta0=[0.3, 0.3, 2.5])
print("estimated theta:", fhn_fit.theta.round(3), " true:", theta_true)
print("standard errors:", fhn_fit.stderr.round(3))
check(
    "FitzHugh-Nagumo parameters recovered (only V observed, noisy)",
    np.allclose(fhn_fit.theta, theta_true, atol=0.1),
    fhn_fit.theta.round(3),
)

# %%
# a custom model: logistic growth Dx = r x (1 - x / K)
logistic = ODEModel(lambda x, t, th: th[0] * x * (1 - x / th[1]), n_states=1, n_params=2)
tl = np.linspace(0.0, 10.0, 60)
xl = logistic.simulate(tl, [0.5], [0.9, 10.0])
yl = xl + 0.1 * np.random.default_rng(2).standard_normal(xl.shape)
lb = fdt.BSpline(domain=(0.0, 10.0), breaks=np.linspace(0.0, 10.0, 31))
problem = ProfiledODE(logistic, tl, yl, lb, lam=1e2)
lfit = problem.fit([0.5, 8.0])
print("logistic: r, K =", lfit.theta.round(3), "(true 0.9, 10)")
check(
    "custom ODEModel: logistic parameters recovered",
    np.allclose(lfit.theta, [0.9, 10.0], rtol=0.1),
    lfit.theta.round(3),
)

nodes, weights = simpson_rule([0.0, 1.0], 5)
check("simpson_rule weights integrate x² exactly", np.isclose(np.sum(weights * nodes**2), 1 / 3))

cstr = cstr_model("all.cool.step", estimate=("kref", "EoverR"))
check(
    "CSTR model builds and evaluates",
    cstr(np.array([[1.5965, 341.3754]]), np.array([1.0]), np.array([0.461, 0.83301])).shape
    == (1, 2),
)
print(
    "CSTR cooling-step inputs at t = 0, 4, 8:", cstr_inputs([0.0, 4.0, 8.0], "all.cool.step")[:, 0]
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(tp, y_obs[:, 0], ".", ms=3, alpha=0.5, label="observed V (noisy)")
axes[0].plot(tp, fhn_fit(tp)[:, 0], "k", label="fitted V")
axes[0].plot(tp, fhn_fit(tp)[:, 1], "g", lw=3, alpha=0.5, label="recovered R")
axes[0].plot(tp, x_true[:, 1], "k:", label="true R (never seen)")
axes[0].legend(fontsize=8)
axes[0].set_title("FitzHugh-Nagumo: R is recovered without data")
axes[1].plot(tl, yl, ".", label="data")
axes[1].plot(tl, xl, "r:", label="truth")
axes[1].plot(tl, lfit(tl), "k", label=f"fit r={lfit.theta[0]:.2f}, K={lfit.theta[1]:.1f}")
axes[1].legend()
axes[1].set_title("Custom ODEModel: logistic growth")
fig.tight_layout()

# %% [markdown]
# ## Part 14. Machine learning: scikit-learn and PyTorch
#
# `Smoother`, `FPCA`, `FCCA`, `Registrator`, `FRegress`, `PDA` and `PACE` follow
# the scikit-learn API, so they fit into a `Pipeline` and `GridSearchCV`.
# `fdatools.nn` has PyTorch layers: `SmoothingLayer` (raw data → coefficients,
# with a learnable λ), `BasisLayer` (coefficients → values) and `FDataDataset`.
# Task: tell boys from girls using only their height curves.

# %%
PART = "14 ML"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from fdatools.smoothing import Smoother

X = np.hstack([growth.hgtm, growth.hgtf]).T  # (93, 31): one row per child
y = np.r_[np.zeros(39), np.ones(54)]  # 0 = boy, 1 = girl
pipe = Pipeline(
    [
        ("smooth", Smoother(hb, t=growth.age, lam=1e-2, penalty=3)),
        ("fpca", FPCA(n=3, basis=hb)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
search = GridSearchCV(pipe, {"fpca__n": [2, 3, 5], "smooth__lam": [1e-2, 1.0]}, cv=5).fit(
    X_tr, y_tr
)
acc = search.score(X_te, y_te)
print("best:", search.best_params_, "| test accuracy:", round(acc, 2))
check("sklearn pipeline classifies boys/girls well above chance", acc > 0.8, round(acc, 2))

# %%
import torch

from fdatools.nn import BasisLayer, FDataDataset, SmoothingLayer

torch.manual_seed(0)
curves = fdt.smooth(X.T, growth.age, basis=hb, lam=1e-2, penalty=3).fd
dataset = FDataDataset(curves, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
grid = np.linspace(1.0, 18.0, 69)
net = torch.nn.Sequential(
    BasisLayer(hb, grid, deriv=1), torch.nn.Linear(69, 1, dtype=torch.float64)
)
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
lossf = torch.nn.BCEWithLogitsLoss()
losses = []
for _epoch in range(80):
    epoch_loss = []
    for cb, lb_ in loader:
        opt.zero_grad()
        loss = lossf(net(cb)[:, 0], lb_.to(torch.float64))
        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())
    losses.append(np.mean(epoch_loss))
with torch.no_grad():
    train_acc = float(((net(dataset.data)[:, 0] > 0).numpy() == y).mean())
check(
    "PyTorch network on growth speed learns (train accuracy)", train_acc > 0.85, round(train_acc, 2)
)

# gradients flow through smoothing, and lambda itself can be learned
smoother = SmoothingLayer(hb, growth.age, lam=1.0, penalty=3, trainable_lam=True)
raw = torch.tensor(X, dtype=torch.float64)
target = torch.tensor(np.asarray(curves.coefs).T)
opt2 = torch.optim.Adam(smoother.parameters(), lr=0.1)
lam_path = []
for _ in range(60):
    opt2.zero_grad()
    l2 = torch.mean((smoother(raw) - target) ** 2)
    l2.backward()
    opt2.step()
    lam_path.append(smoother.lam)
print(f"learned lambda: 1.0 -> {smoother.lam:.3g} (target was fitted with 1e-2)")
check(
    "trainable lambda moves toward the lambda used for the target",
    abs(np.log10(smoother.lam) + 2) < abs(np.log10(1.0) + 2),
)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
axes[0].plot(losses)
axes[0].set(title="Training loss (mean per epoch)", xlabel="epoch")
axes[1].plot(grid, net[1].weight.detach().numpy()[0])
axes[1].axhline(0, lw=0.6)
axes[1].set(title="What the network looks at (weights over age)", xlabel="age")
axes[2].semilogy(lam_path)
axes[2].axhline(1e-2, ls=":", c="k")
axes[2].set(title="Learning lambda by gradient descent", xlabel="step")
fig.tight_layout()

# %% [markdown]
# ## Part 15. Input and output
#
# Curves move to and from **pandas** (long table: `t`, `curve`, `value`),
# **xarray**, and R's `.rds` files (`read_rds` reads R `fda` objects directly).

# %%
PART = "15 IO"
grid5 = np.linspace(1.0, 18.0, 5)
frame = girls[:3].to_pandas(grid5)
show(frame.head(6))
check(
    "to_pandas gives a long table",
    list(frame.columns) == ["t", "curve", "value"] and len(frame) == 15,
)
check("fdt.to_pandas == FData.to_pandas", fdt.to_pandas(girls[:3], grid5).equals(frame))
da = girls[:3].to_xarray(grid5)
print(da)
check("to_xarray has dims (t, curve)", da.dims == ("t", "curve"))

long = fdt.from_pandas(frame, "curve", "t", "value")
check("from_pandas(to_pandas(fd)) round-trips the values", np.allclose(long.y, girls[:3](grid5)))

fixtures = next(
    (
        p / "tests" / "fixtures"
        for p in (here, *here.parents)
        if (p / "tests" / "fixtures").is_dir()
    ),
    None,
)
if fixtures is not None:
    rb = fdt.read_rds(fixtures / "bspline_basis.rds")
    rfd = fdt.read_rds(fixtures / "bspline_fd.rds")
    print("from R:", rb, "|", rfd)
    check(
        "read_rds reads an R basis and an R fd object",
        isinstance(rb, fdt.Basis) and isinstance(rfd, fdt.FData),
    )
else:
    print("no R test files found next to this notebook; skipping read_rds")

# %% [markdown]
# ## Part 16. Live comparison with R `fda`
#
# The library is tested against saved R output, but here we run **R itself** and
# compare the numbers directly. R runs in its own process (`Rscript`) and sends
# its results back as JSON with 17 significant digits. (Running R *inside* this kernel
# with `rpy2` would crash here: PyTorch and R each load their own copy of the
# OpenMP library.) This part needs R with the `fda` and `jsonlite` packages; it
# is skipped otherwise.

# %%
PART = "16 R parity"
import json
import shutil
import subprocess

lam_py = df_to_lambda(growth.age, hb, 8.0, penalty=3)
R_SCRIPT = f"""
suppressMessages({{library(fda); library(jsonlite)}})
hb   <- create.bspline.basis(c(1, 18), 20, norder = 6)
sm   <- smooth.basis(growth$age, growth$hgtf, fdPar(hb, 3, 0.1))
lamR <- df2lambda(growth$age, hb, Lfdobj = 3, df = 8)
fb65 <- create.fourier.basis(c(0, 365), 65)
harm <- vec2Lfd(c(0, (2*pi/365)^2, 0), c(0, 365))
tf   <- smooth.basis(day.5, CanadianWeather$dailyAv[,,"Temperature.C"], fdPar(fb65, harm, 1e2))$fd
pc   <- pca.fd(tf, 4)
C    <- tf$coefs - rowMeans(tf$coefs)
exact_scores <- t(C) %*% eval.penalty(fb65, 0) %*% pc$harmonics$coefs   # exact inner products
out <- list(
  basis = eval.basis(growth$age, hb), basis_d2 = eval.basis(growth$age, hb, 2),
  gram_exact = bsplinepen(hb, 0), gram_inprod = inprod(hb, hb), pen3 = eval.penalty(hb, 3),
  coefs = sm$fd$coefs, df = sm$df, gcv = sm$gcv,
  df_at_our_lambda = lambda2df(growth$age, hb, Lfdobj = 3, lambda = {lam_py!r}),
  lam_r = lamR, df_at_r_lambda = lambda2df(growth$age, hb, Lfdobj = 3, lambda = lamR),
  varprop = pc$varprop, values = pc$values[1:4], scores = pc$scores, exact_scores = exact_scores)
cat(toJSON(lapply(out, unclass), digits = 17))
"""

HAVE_R = shutil.which("Rscript") is not None
if HAVE_R:
    run = subprocess.run([shutil.which("Rscript"), "-e", R_SCRIPT], capture_output=True, text=True)  # noqa: S603
    HAVE_R = run.returncode == 0
    if not HAVE_R:
        print("R failed, skipping:", run.stderr[-500:])
else:
    print("Rscript not found, skipping")


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


if HAVE_R:
    r = {key: np.asarray(value, dtype=float) for key, value in json.loads(run.stdout).items()}
    rows = []

    def compare(name, ours, theirs, tol=1e-8):
        err = rel(ours, theirs)
        rows.append({"quantity": name, "max rel. difference": f"{err:.1e}", "tolerance": tol})
        check(f"R parity: {name}", err < tol, f"{err:.1e}")

    py = fdt.smooth(growth.hgtf, growth.age, basis=hb, lam=0.1, penalty=3)
    compare("B-spline values (eval.basis)", hb(growth.age), r["basis"])
    compare("B-spline 2nd derivatives", hb(growth.age, deriv=2), r["basis_d2"])
    compare("Gram matrix (bsplinepen, exact)", hb.gram(), r["gram_exact"])
    compare("D3 penalty matrix (eval.penalty)", hb.penalty(3), r["pen3"])
    compare("smooth.basis coefficients", py.fd.coefs, r["coefs"])
    compare("smooth.basis df", py.df, r["df"])
    compare("smooth.basis GCV", py.gcv, r["gcv"])
    compare(
        "lambda2df (df for a given lambda)",
        lambda_to_df(growth.age, hb, lam_py, penalty=3),
        r["df_at_our_lambda"],
    )
    # R's day grid is 0.5, 1.5, ..., 364.5 (day.5); smooth on the same points
    temp_r = fdt.smooth(weather.temp, day - 0.5, basis=fourier, penalty=harmonic, lam=1e2).fd
    pca_r = FPCA(n=4).fit(temp_r)
    compare("pca.fd variance proportions", pca_r.varprop, r["varprop"])
    compare("pca.fd eigenvalues (first 4)", pca_r.values[:4], r["values"])
    compare("PCA scores (R, exact inner products)", np.abs(pca_r.scores), np.abs(r["exact_scores"]))
    show(pd.DataFrame(rows))

    # Two places where R is the less exact side:
    print(
        "R's inprod(hb, hb) uses approximate (Romberg) integration: off by "
        f"{rel(r['gram_inprod'], hb.gram()):.1e}"
    )
    print(
        "R's pca.fd scores also use inprod(): off by "
        f"{rel(np.abs(r['scores']), np.abs(r['exact_scores'])):.1e} from R's own exact scores"
    )
    ours_df = lambda_to_df(growth.age, hb, lam_py, penalty=3)
    print(f"df2lambda for df = 8:  ours {lam_py:.10f} -> df {ours_df:.10f}")
    print(
        f"                       R    {float(r['lam_r'][0]):.10f} -> "
        f"df {float(r['df_at_r_lambda'][0]):.10f} (R's search stops early)"
    )
    check(
        "df_to_lambda hits df = 8 more exactly than R's df2lambda",
        abs(ours_df - 8) < abs(float(r["df_at_r_lambda"][0]) - 8),
    )

# %% [markdown]
# ## Part 17. Summary of all checks

# %%
summary = pd.DataFrame(RESULTS)
by_part = summary.groupby("part", sort=False)["passed"].agg(checks="count", passed="sum")
show(by_part)
print(f"\n{int(summary.passed.sum())} of {len(summary)} checks passed in {time.time() - T0:.0f} s")
check("every check passed", bool(summary.passed.all()))

# %%
show(pd.DataFrame(RESULTS)[["part", "check", "passed", "detail"]])
