"""Bridges to NumPy, SciPy, scikit-fda, panel econometrics and linear estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import fdatools as fdt
from fdatools.interop import (
    FunctionalDesign,
    PanelCurves,
    from_grid,
    from_panel,
    from_scipy,
    from_skfda,
    functional_design,
    to_grid,
    to_panel,
    to_scipy,
    to_skfda,
)

RNG = np.random.default_rng(20261021)


def _curves(n_curves: int = 4, basis: fdt.Basis | None = None) -> fdt.FData:
    basis = basis or fdt.BSpline(domain=(0.0, 2.0), n_basis=8)
    return fdt.FData(RNG.standard_normal((basis.n_basis, n_curves)), basis)


# ------------------------------------------------------------------ NumPy


@pytest.mark.parametrize("axis", [0, 1])
def test_grid_round_trip_is_exact_in_the_span(axis: int) -> None:
    fd = _curves()
    t, values = to_grid(fd, n_points=60, curve_axis=axis)
    assert values.shape == ((4, 60) if axis == 0 else (60, 4))
    back = from_grid(values, t, basis=fd.basis, lam=0.0, curve_axis=axis)
    assert_allclose(back.coefs, fd.coefs, atol=1e-10)


def test_to_grid_derivative_and_explicit_points() -> None:
    fd = _curves()
    t = np.array([0.2, 1.1, 1.9])
    _, values = to_grid(fd, t, deriv=1)
    assert_allclose(values, fd.derivative()(t), atol=1e-12)


def test_from_grid_defaults_to_integer_points_and_gcv() -> None:
    values = np.sin(np.linspace(0.0, 3.0, 30))[:, None] + 0.01 * RNG.standard_normal((30, 2))
    fd = from_grid(values)
    assert fd.domain == (0.0, 29.0)
    assert fd.n_curves == 2


def test_grid_argument_errors() -> None:
    fd = _curves()
    with pytest.raises(ValueError, match="curve_axis"):
        to_grid(fd, curve_axis=2)
    with pytest.raises(ValueError, match="n_points"):
        to_grid(fd, n_points=1)
    with pytest.raises(ValueError, match="t must hold"):
        from_grid(np.ones((5, 2)), np.arange(4.0))


# ------------------------------------------------------------------ SciPy


def test_scipy_round_trip_evaluates_differentiates_and_integrates_alike() -> None:
    basis = fdt.BSpline(domain=(0.0, 2.0), breaks=[0.0, 0.3, 0.3, 1.2, 2.0], order=4)
    fd = _curves(3, basis)
    spl = to_scipy(fd)
    t = np.linspace(0.0, 2.0, 41)
    assert_allclose(spl(t), fd(t), atol=1e-12)
    assert_allclose(spl.derivative(2)(t[1:-1]), fd(t[1:-1], 2), atol=1e-9)
    integral = fdt.inprod(fd, fdt.FData(np.ones(1), fdt.Constant((0.0, 2.0))))[:, 0]
    assert_allclose(spl.integrate(0.0, 2.0), integral, atol=1e-12)
    back = from_scipy(spl)
    assert back.basis == basis
    assert_allclose(back.coefs, fd.coefs)


def test_scipy_outside_domain_is_nan_unless_extrapolating() -> None:
    fd = _curves(1)
    assert np.isnan(to_scipy(fd)(np.array([3.0]))).all()
    assert np.isfinite(to_scipy(fd, extrapolate=True)(np.array([3.0]))).all()


def test_polygonal_converts_as_a_linear_spline() -> None:
    basis = fdt.Polygonal(argvals=(0.0, 0.5, 2.0))
    fd = _curves(2, basis)
    t = np.linspace(0.0, 2.0, 17)
    assert_allclose(to_scipy(fd)(t), fd(t), atol=1e-12)


def test_from_scipy_reads_scipy_constructors() -> None:
    from scipy.interpolate import make_interp_spline, make_lsq_spline

    x = np.linspace(0.0, 1.0, 12)
    y = np.column_stack([np.cos(3 * x), x**3])
    spl = make_interp_spline(x, y, k=3)
    fd = from_scipy(spl)
    assert fd.n_curves == 2
    assert_allclose(fd(x), y, atol=1e-12)
    knots = np.r_[[0.0] * 3, np.linspace(0.0, 1.0, 5), [1.0] * 3]
    lsq = make_lsq_spline(x, y[:, 0], knots, k=3)
    assert_allclose(from_scipy(lsq)(x)[:, 0], lsq(x), atol=1e-12)


def test_scipy_errors() -> None:
    from scipy.interpolate import BSpline as ScipyBSpline

    with pytest.raises(TypeError, match="BSpline and Polygonal"):
        to_scipy(_curves(1, fdt.Fourier((0.0, 1.0), 5)))
    unclamped = ScipyBSpline(np.arange(8.0), np.ones(4), 3)
    with pytest.raises(ValueError, match="clamped"):
        from_scipy(unclamped)
    clamped = ScipyBSpline(np.r_[0.0, 0.0, 1.0, 1.0], np.ones((2, 3)), 1, axis=1)
    with pytest.raises(ValueError, match="axis=0"):
        from_scipy(clamped)


# ------------------------------------------------------------- scikit-fda


@pytest.mark.parametrize(
    "basis",
    [
        fdt.BSpline(domain=(0.0, 2.0), breaks=[0.0, 0.5, 0.5, 2.0], order=3),
        fdt.Fourier(domain=(0.0, 2.0), n_basis=5),
        fdt.Monomial(domain=(0.0, 2.0), exponents=(0, 1, 2)),
        fdt.Constant(domain=(0.0, 2.0)),
    ],
    ids=["bspline", "fourier", "monomial", "constant"],
)
def test_skfda_basis_round_trip_is_exact(basis: fdt.Basis) -> None:
    pytest.importorskip("skfda")
    from skfda import FDataBasis

    fd = _curves(3, basis)
    sk = to_skfda(fd)
    assert isinstance(sk, FDataBasis)
    t = np.linspace(0.0, 2.0, 9)
    assert_allclose(np.asarray(sk(t))[:, :, 0].T, fd(t), atol=1e-12)
    back = from_skfda(sk)
    assert type(back.basis) is type(basis)
    assert_allclose(back.coefs, fd.coefs)


def test_skfda_polygonal_is_exact_and_other_bases_fall_back_to_a_grid() -> None:
    pytest.importorskip("skfda")
    from skfda import FDataBasis, FDataGrid

    poly = _curves(2, fdt.Polygonal(argvals=(0.0, 1.0, 2.0)))
    t = np.linspace(0.0, 2.0, 7)
    sk = to_skfda(poly)
    assert isinstance(sk, FDataBasis)
    assert_allclose(np.asarray(sk(t))[:, :, 0].T, poly(t), atol=1e-12)

    expo = _curves(2, fdt.Exponential(domain=(0.0, 2.0), rates=(0.0, 1.0)))
    grid = to_skfda(expo, n_points=31)
    assert isinstance(grid, FDataGrid)
    back = from_skfda(grid, basis=expo.basis, lam=0.0)
    assert_allclose(back.coefs, expo.coefs, atol=1e-10)


def test_from_skfda_refits_an_unsupported_basis_onto_a_given_one() -> None:
    pytest.importorskip("skfda")
    fd = _curves(2, fdt.Fourier(domain=(0.0, 2.0), n_basis=5))
    target = fdt.Fourier(domain=(0.0, 2.0), n_basis=7)
    back = from_skfda(to_skfda(fd), basis=target, lam=0.0)
    t = np.linspace(0.0, 2.0, 13)
    assert_allclose(back(t), fd(t), atol=1e-10)


def test_skfda_errors() -> None:
    pytest.importorskip("skfda")
    multi = fdt.FData(np.ones((4, 2, 2)), fdt.BSpline(n_basis=4))
    with pytest.raises(ValueError, match="variable axis"):
        to_skfda(multi)

    class Fake:
        dim_domain = 1
        dim_codomain = 1

    with pytest.raises(TypeError, match="FDataBasis or FDataGrid"):
        from_skfda(Fake())


# ------------------------------------------------------------------ panels


def _panel(balanced: bool = True) -> Any:
    pd = pytest.importorskip("pandas")
    years = np.arange(2000.0, 2015.0)
    idx = pd.MultiIndex.from_product([["acme", "bolt", "core"], years], names=["firm", "year"])
    trend = np.tile(np.sin((years - 2000.0) / 4.0), 3)
    frame = pd.DataFrame(
        {"sales": trend + 0.05 * RNG.standard_normal(trend.size), "other": 0.0}, index=idx
    )
    if not balanced:
        frame = frame.drop(index=[("bolt", 2003.0), ("core", 2010.0)])
        frame.loc[("acme", 2005.0), "sales"] = np.nan
    return frame


def test_panel_round_trip_balanced() -> None:
    frame = _panel()
    basis = fdt.BSpline(domain=(2000.0, 2014.0), n_basis=15)
    panel = from_panel(frame, "sales", basis=basis, lam=0.0)
    assert isinstance(panel, PanelCurves)
    assert panel.entities == ["acme", "bolt", "core"]
    years = np.arange(2000.0, 2015.0)
    out = to_panel(panel.fd, years, entities=panel.entities, names=("firm", "year"), value="sales")
    assert_allclose(out["sales"].to_numpy(), frame["sales"].to_numpy(), atol=1e-8)
    assert list(out.index.names) == ["firm", "year"]


def test_panel_unbalanced_is_fitted_per_entity() -> None:
    panel = from_panel(_panel(balanced=False)["sales"], lam=1e-2)
    assert panel.fd.n_curves == 3
    assert panel.fd.domain == (2000.0, 2014.0)


def test_panel_errors() -> None:
    pd = pytest.importorskip("pandas")
    frame = _panel()
    with pytest.raises(ValueError, match="name the column"):
        from_panel(frame)
    with pytest.raises(ValueError, match="not found"):
        from_panel(frame, "missing")
    with pytest.raises(ValueError, match="MultiIndex"):
        from_panel(pd.Series([1.0, 2.0]))
    labelled = pd.Series([1.0, 2.0], index=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")]))
    with pytest.raises(ValueError, match="numeric"):
        from_panel(labelled)
    with pytest.raises(ValueError, match="entity labels"):
        to_panel(_curves(2), entities=["only-one"])
    with pytest.raises(ValueError, match="variable axis"):
        to_panel(fdt.FData(np.ones((4, 2, 2)), fdt.BSpline(n_basis=4)))


def test_to_panel_feeds_linearmodels_panelols() -> None:
    pytest.importorskip("pandas")
    lm = pytest.importorskip("linearmodels")
    fd = _curves(5)
    x = to_panel(fd, np.linspace(0.0, 2.0, 12), value="x")
    effect = np.repeat(np.arange(5.0), 12)
    y = 2.5 * x["x"] + effect
    result = lm.PanelOLS(y, x, entity_effects=True).fit()
    assert_allclose(float(result.params["x"]), 2.5, rtol=1e-8)


# -------------------------------------------------------- functional design


def _scalar_on_function(
    n: int = 200, noise: float = 0.0
) -> tuple[fdt.FData, Any, Any, fdt.Basis, fdt.FData]:
    basis = fdt.BSpline(domain=(0.0, 1.0), n_basis=10)
    x = fdt.FData(RNG.standard_normal((10, n)), basis)
    beta_basis = fdt.BSpline(domain=(0.0, 1.0), n_basis=5)
    true_beta = fdt.FData(np.array([1.0, -0.5, 2.0, 0.0, 1.5]), beta_basis)
    w = RNG.standard_normal(n)
    y = 0.7 + 0.3 * w + fdt.inprod(x, true_beta)[:, 0] + noise * RNG.standard_normal(n)
    return x, w, y, beta_basis, true_beta


def test_design_recovers_the_true_coefficient_function() -> None:
    x, w, y, beta_basis, true_beta = _scalar_on_function()
    design = functional_design({"x": x}, {"w": w}, beta_basis=beta_basis)
    assert isinstance(design, FunctionalDesign)
    assert design.columns == ("const", "w", "x[0]", "x[1]", "x[2]", "x[3]", "x[4]")
    params, *_ = np.linalg.lstsq(design.matrix, y, rcond=None)
    assert_allclose(params[:2], [0.7, 0.3], atol=1e-9)
    assert_allclose(design.beta(params).coefs, true_beta.coefs, atol=1e-9)


def test_design_with_ols_equals_fregress_without_penalty() -> None:
    x, _, y, beta_basis, _ = _scalar_on_function(noise=0.3)
    design = functional_design(x, beta_basis=beta_basis)
    params, *_ = np.linalg.lstsq(design.matrix, y, rcond=None)
    model = fdt.fregress(y, {"const": 1.0, "x": x}, beta={"x": beta_basis}, lam=0.0)
    assert_allclose(design.beta(params).coefs, model.beta[1].coefs, rtol=1e-8, atol=1e-10)


def test_design_with_statsmodels_labels_and_band() -> None:
    pytest.importorskip("pandas")
    sm = pytest.importorskip("statsmodels.api")
    x, w, y, beta_basis, true_beta = _scalar_on_function(noise=0.5)
    design = functional_design({"x": x}, {"w": w}, beta_basis=beta_basis)
    result = sm.OLS(y, design.to_frame()).fit(cov_type="HC3")
    beta = design.beta(result.params)
    shuffled = result.params.iloc[::-1]
    assert_allclose(design.beta(shuffled).coefs, beta.coefs)
    t, est, lower, upper = design.band(result.params, result.cov_params(), level=0.99)
    assert_allclose(est, beta(t)[:, 0])
    assert np.all(lower <= est)
    assert np.all(est <= upper)
    covered = (lower <= true_beta(t)[:, 0]) & (true_beta(t)[:, 0] <= upper)
    assert covered.mean() > 0.8
    t2, *_ = design.band(np.asarray(result.params), np.asarray(result.cov_params()), t=t[:5])
    assert_allclose(t2, t[:5])


def test_design_feeds_a_glm() -> None:
    pytest.importorskip("pandas")
    sm = pytest.importorskip("statsmodels.api")
    x, w, eta, beta_basis, true_beta = _scalar_on_function(n=3000)
    scale = 2.0
    counts = RNG.poisson(np.exp(scale * (eta - eta.mean())))
    design = functional_design({"x": x}, {"w": w}, beta_basis=beta_basis)
    fit = sm.GLM(counts, design.to_frame(), family=sm.families.Poisson()).fit()
    estimate = design.beta(fit.params).coefs[:, 0]
    stderr = fit.bse[[f"x[{k}]" for k in range(5)]].to_numpy()
    assert np.all(np.abs(estimate - scale * true_beta.coefs[:, 0]) < 4.0 * stderr)


def test_design_with_two_functional_covariates_and_defaults() -> None:
    x1 = _curves(30, fdt.BSpline(domain=(0.0, 1.0), n_basis=8))
    x2 = _curves(30, fdt.Fourier(domain=(0.0, 2.0), n_basis=5))
    design = functional_design(
        {"a": x1, "b": x2},
        beta_basis={"b": fdt.Fourier((0.0, 2.0), 3)},
        n_basis=3,
        intercept=False,
    )
    assert design.functional_names == ("a", "b")
    assert design.shape == (30, 6)
    beta_a = design.beta(np.arange(6.0), "a").basis
    assert isinstance(beta_a, fdt.BSpline)
    assert beta_a.order == 3
    assert isinstance(design.beta(np.arange(6.0), "b").basis, fdt.Fourier)
    with pytest.raises(ValueError, match="name one"):
        design.beta(np.arange(6.0))
    with pytest.raises(KeyError, match="no functional covariate"):
        design.beta(np.arange(6.0), "c")


def test_design_errors() -> None:
    x = _curves(6)
    with pytest.raises(ValueError, match="at least one"):
        functional_design({})
    with pytest.raises(ValueError, match="n_basis"):
        functional_design(x, n_basis=0)
    with pytest.raises(ValueError, match="disagree"):
        functional_design(x, {"w": np.ones(5)})
    with pytest.raises(ValueError, match="unique"):
        functional_design(x, {"const": np.ones(6)})
    with pytest.raises(ValueError, match="variable axis"):
        functional_design(fdt.FData(np.ones((4, 6, 2)), fdt.BSpline(n_basis=4)))
    design = functional_design(x, n_basis=4)
    with pytest.raises(ValueError, match="parameters"):
        design.beta(np.ones(3))
    with pytest.raises(ValueError, match="covariance"):
        design.band(np.ones(5), np.eye(3))
    with pytest.raises(ValueError, match="level"):
        design.band(np.ones(5), np.eye(5), level=1.0)


def test_panel_unbalanced_matches_per_entity_smooths_and_rejects_duplicates() -> None:
    pd = pytest.importorskip("pandas")
    frame = _panel(balanced=False).sample(frac=1.0, random_state=3)  # shuffled rows
    basis = fdt.BSpline(domain=(2000.0, 2014.0), n_basis=6)
    panel = from_panel(frame, "sales", basis=basis, lam=1e-3)
    clean = frame["sales"].dropna()
    for k, firm in enumerate(panel.entities):
        rows = clean.xs(firm, level="firm").sort_index()
        alone = fdt.smooth(rows.to_numpy(), rows.index.to_numpy(), basis=basis, lam=1e-3)
        assert_allclose(panel.fd.coefs[:, k], alone.fd.coefs[:, 0], rtol=1e-8, atol=1e-10)
    doubled = pd.concat([frame["sales"], frame["sales"].iloc[:1]])
    with pytest.raises(ValueError, match="repeats"):
        from_panel(doubled)
    with pytest.raises(ValueError, match="no non-missing"):
        from_panel(frame["sales"] * np.nan)


# ------------------------------------------- prediction, FPCA, simultaneous


def test_transform_and_predict_rebuild_the_design_for_new_curves() -> None:
    x, w, y, beta_basis, _ = _scalar_on_function()
    design = functional_design({"x": x}, {"w": w}, beta_basis=beta_basis)
    assert_allclose(design.transform({"x": x}, {"w": w}), design.matrix)
    params, *_ = np.linalg.lstsq(design.matrix, y, rcond=None)
    fresh, _, _, _, _ = _scalar_on_function(n=25)
    # The same curves in another (nested, finer) basis give the same integrals.
    finer = fdt.BSpline(domain=(0.0, 1.0), breaks=np.linspace(0.0, 1.0, 15).tolist())
    t = np.linspace(0.0, 1.0, 400)
    refit = fdt.smooth(fresh(t), t, basis=finer, lam=0.0).fd
    w_fresh = RNG.standard_normal(25)
    exact = design.predict(params, {"x": fresh}, {"w": w_fresh})
    assert_allclose(design.predict(params, refit, {"w": w_fresh}), exact, atol=1e-6)
    with pytest.raises(ValueError, match="missing covariates"):
        design.transform({"x": fresh})
    with pytest.raises(ValueError, match="disagree"):
        design.transform({"x": fresh}, {"w": np.ones(3)})


def test_fpca_design_is_principal_component_regression() -> None:
    x, _, y, _, _ = _scalar_on_function(noise=0.2)
    design = functional_design(x, method="fpca", n_components=3)
    assert design.columns == ("const", "x[pc1]", "x[pc2]", "x[pc3]")
    pca = fdt.FPCA(n=3).fit(x)
    # Centred scores of an unpenalised FPCA are the integrals against the harmonics.
    assert_allclose(design.matrix[:, 1:], pca.scores, atol=1e-8)
    params, *_ = np.linalg.lstsq(design.matrix, y, rcond=None)
    beta = design.beta(params)
    expected = pca.harmonics.coefs @ params[1:]
    assert_allclose(beta.coefs[:, 0], expected)
    # Keeping every component spans the covariate's own basis: same fit as method="basis".
    full = functional_design(x, method="fpca", n_components=x.basis.n_basis)
    same = functional_design(x, beta_basis=x.basis)
    fit_full, *_ = np.linalg.lstsq(full.matrix, y, rcond=None)
    fit_same, *_ = np.linalg.lstsq(same.matrix, y, rcond=None)
    assert_allclose(full.matrix @ fit_full, same.matrix @ fit_same, atol=1e-8)
    t = np.linspace(0.0, 1.0, 9)
    assert_allclose(full.beta(fit_full)(t), same.beta(fit_same)(t), atol=1e-6)
    # Prediction re-centres new curves at the training mean.
    assert_allclose(design.predict(params, x), design.matrix @ params)


def test_explicit_functions_can_be_passed_as_fdata() -> None:
    x, _, _, _, _ = _scalar_on_function()
    pca = fdt.FPCA(n=2).fit(x)
    design = functional_design({"x": x}, beta_basis={"x": pca.harmonics})
    assert design.columns == ("const", "x[0]", "x[1]")
    assert_allclose(design.matrix[:, 1:], fdt.inprod(x, pca.harmonics))


def test_simultaneous_band_is_wider_and_has_simultaneous_coverage() -> None:
    x, _, _, beta_basis, true_beta = _scalar_on_function(n=80)
    design = functional_design(x, beta_basis=beta_basis)
    xtx_inv = np.linalg.inv(design.matrix.T @ design.matrix)
    t = np.linspace(0.0, 1.0, 51)
    truth = true_beta(t)[:, 0]
    hits_point = hits_sup = 0
    trials = 300
    for _ in range(trials):
        y = design.matrix @ np.r_[0.7, true_beta.coefs[:, 0]] + RNG.standard_normal(80)
        params = xtx_inv @ design.matrix.T @ y
        cov = xtx_inv  # known unit noise variance
        _, _, lo, hi = design.band(params, cov, t=t)
        _, _, slo, shi = design.band(params, cov, t=t, simultaneous=True)
        assert np.all(shi - slo >= hi - lo - 1e-12)
        hits_point += bool(np.all((lo <= truth) & (truth <= hi)))
        hits_sup += bool(np.all((slo <= truth) & (truth <= shi)))
    assert hits_sup / trials > 0.9  # nominal 0.95 for the whole curve
    assert hits_point / trials < hits_sup / trials


def test_band_with_degrees_of_freedom_and_reproducible_simulation() -> None:
    x = _curves(30, fdt.BSpline(domain=(0.0, 1.0), n_basis=8))
    design = functional_design(x, n_basis=4)
    params, cov = np.ones(5), np.eye(5)
    _, _, lo_z, hi_z = design.band(params, cov, n_points=11)
    _, _, lo_t, hi_t = design.band(params, cov, n_points=11, df=5)
    assert np.all(hi_t - lo_t > hi_z - lo_z)
    first = design.band(params, cov, n_points=11, simultaneous=True, df=5)
    again = design.band(params, cov, n_points=11, simultaneous=True, df=5)
    assert_allclose(first[2], again[2])
    zero = design.band(params, np.zeros((5, 5)), n_points=11, simultaneous=True)
    assert_allclose(zero[2], zero[3])
    with pytest.raises(ValueError, match="df"):
        design.band(params, cov, df=0)


def test_new_design_argument_errors() -> None:
    x = _curves(6)
    with pytest.raises(ValueError, match="method"):
        functional_design(x, method="pls")
    with pytest.raises(ValueError, match="n_components"):
        functional_design(x, method="fpca", n_components=0)
    with pytest.raises(ValueError, match="coefficient functions"):
        functional_design(x, beta_basis=fdt.FData(np.ones((8, 2, 2)), x.basis))
