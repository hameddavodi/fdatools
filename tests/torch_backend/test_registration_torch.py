"""PyTorch path of :func:`fdatools.registration.register` (SPEC 1.5 and 5.2).

With tensor coefficients the continuous criterion is differentiated by
autograd instead of the analytic formulas; the Newton iteration is shared, so
the torch result must equal the NumPy one.  The registered curves come back as
tensors that are differentiable with respect to the input coefficients (the
optimal warps held fixed).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fdatools import BSpline, FData, Fourier
from fdatools.registration import (
    RegistrationResult,
    Registrator,
    _CurveProblem,
    _fine_grid,
    _penalty_matrix,
    _WarpQuadrature,
    register,
)

torch = pytest.importorskip("torch", reason="torch extra not installed")

from fdatools._internal.registration_torch import AutogradObjective  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "registration.json"
GROWTH = "register_fd_growth_hgtf_to_mean"
RTOL = 1e-5
DOMAIN = (1.0, 18.0)


def _curves(n_curves: int = 10) -> FData:
    """The benchmark problem: logistic growth spurts in an order-6 spline basis."""
    basis = BSpline(domain=DOMAIN, n_basis=35, order=6)
    grid = np.linspace(*DOMAIN, 400)
    rng = np.random.default_rng(20260927)
    centres = 11.5 + rng.normal(scale=1.2, size=n_curves)
    values = np.stack(
        [80.0 + 4.5 * grid + 20.0 / (1.0 + np.exp(-(grid - c) / 0.8)) for c in centres], axis=1
    )
    return FData(np.linalg.lstsq(basis(grid), values, rcond=None)[0], basis)


def _as_torch(fd: FData, requires_grad: bool = False) -> tuple[FData, Any]:
    coefs = torch.tensor(np.asarray(fd.coefs), dtype=torch.float64, requires_grad=requires_grad)
    return FData(coefs, fd.basis), coefs


def _numpy(values: Any) -> np.ndarray:
    assert isinstance(values, torch.Tensor), f"expected a tensor, got {type(values).__name__}"
    return np.asarray(values.detach().cpu().numpy())


def _assert_same(torch_res: RegistrationResult, numpy_res: RegistrationResult) -> None:
    """Every field of the torch result is a tensor equal to the NumPy result."""
    for name in ("registered", "warp", "latent", "unregistered"):
        got = getattr(torch_res, name).coefs
        want = np.asarray(getattr(numpy_res, name).coefs)
        scale = max(1.0, float(np.max(np.abs(want))))
        np.testing.assert_allclose(_numpy(got), want, rtol=RTOL, atol=1e-12 * scale, err_msg=name)
    np.testing.assert_allclose(_numpy(torch_res.shift), numpy_res.shift, rtol=RTOL, atol=1e-9)
    assert numpy_res.criterion is not None
    np.testing.assert_allclose(_numpy(torch_res.criterion), numpy_res.criterion, rtol=RTOL)
    assert torch_res.target is not None
    assert numpy_res.target is not None
    np.testing.assert_allclose(
        _numpy(torch_res.target.coefs), np.asarray(numpy_res.target.coefs), rtol=1e-12
    )
    assert isinstance(torch_res.n_iter, torch.Tensor)


def _assert_monotone(
    res: RegistrationResult, domain: tuple[float, float], *, strict: bool = True
) -> None:
    t = torch.linspace(domain[0], domain[1], 1001, dtype=torch.float64)
    warps = res.warp_values(t)
    assert isinstance(warps, torch.Tensor)
    steps = torch.diff(warps, dim=0)
    assert bool(torch.all(steps > 0.0 if strict else steps >= 0.0))


# --------------------------------------------------------------------------- #
# same optimum as the NumPy Newton iteration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("criterion", ["eigen", "least_squares"])
def test_torch_matches_numpy_on_benchmark_curves(criterion: str) -> None:
    fd = _curves()
    options: dict[str, Any] = {
        "warp_basis": BSpline(domain=DOMAIN, n_basis=5),
        "lam": 1.0,
        "criterion": criterion,
    }
    expected = register(fd, **options)
    tfd, _ = _as_torch(fd)
    result = register(tfd, **options)
    _assert_same(result, expected)
    _assert_monotone(result, DOMAIN)
    assert result.unregistered is tfd


def test_torch_matches_numpy_on_growth_golden_case() -> None:
    case = next(c for c in json.loads(GOLDEN.read_text())["cases"] if c["name"] == GROWTH)
    inputs = case["input"]

    def spline(spec: dict[str, Any]) -> BSpline:
        interior = np.atleast_1d(np.asarray(spec["params"], dtype=float)).tolist()
        lo, hi = (float(v) for v in spec["rangeval"])
        return BSpline(
            domain=(lo, hi), order=spec["nbasis"] - len(interior), breaks=[lo, *interior, hi]
        )

    basis = spline(inputs["basis"])
    fd = FData(np.asarray(inputs["coefs"], dtype=float), basis)
    target = FData(np.asarray(inputs["y0fd_coefs"], dtype=float), basis)
    options: dict[str, Any] = {
        "warp_basis": spline(inputs["wbasis"]),
        "lam": 1.0,
        "penalty": 2,
        "criterion": "eigen" if inputs["crit"] == 2 else "least_squares",
    }
    expected = register(fd, target, **options)
    tfd, coefs = _as_torch(fd, requires_grad=True)
    result = register(tfd, _as_torch(target)[0], **options)
    _assert_same(result, expected)
    _assert_monotone(result, basis.domain)
    result.registered(torch.linspace(1.0, 18.0, 50, dtype=torch.float64)).sum().backward()
    assert coefs.grad is not None
    assert bool(torch.all(torch.isfinite(coefs.grad)))


@pytest.mark.parametrize("criterion", ["eigen", "least_squares"])
def test_torch_matches_numpy_periodic(criterion: str) -> None:
    basis = Fourier(domain=(0.0, 1.0), n_basis=7)
    grid = np.linspace(0.0, 1.0, 401)
    values = np.stack(
        [
            np.sin(2 * np.pi * (grid - s)) + 0.2 * np.cos(4 * np.pi * grid)
            for s in (-0.05, 0.0, 0.07)
        ],
        axis=1,
    )
    fd = FData(np.linalg.lstsq(basis(grid), values, rcond=None)[0], basis)
    options: dict[str, Any] = {
        "warp_basis": BSpline(domain=(0.0, 1.0), n_basis=4),
        "lam": 1e-2,
        "criterion": criterion,
        "periodic": True,
    }
    expected = register(fd, **options)
    result = register(_as_torch(fd)[0], **options)
    _assert_same(result, expected)
    _assert_monotone(result, (0.0, 1.0))


def test_torch_target_with_numpy_curves() -> None:
    fd = _curves(3)
    target = fd.mean()
    options: dict[str, Any] = {"warp_basis": BSpline(domain=DOMAIN, n_basis=4), "lam": 1.0}
    expected = register(fd, target, **options)
    result = register(fd, _as_torch(target)[0], **options)
    _assert_same(result, expected)


def test_zero_iterations_and_zero_curves() -> None:
    fd = _curves(2)
    options: dict[str, Any] = {"warp_basis": BSpline(domain=DOMAIN, n_basis=4), "lam": 1.0}
    identity = register(_as_torch(fd)[0], max_iter=0, **options)
    np.testing.assert_array_equal(_numpy(identity.n_iter), [0, 0])
    zeros = FData(torch.zeros((6, 2), dtype=torch.float64), BSpline(domain=DOMAIN, n_basis=6))
    flat = register(zeros)
    np.testing.assert_array_equal(_numpy(flat.n_iter), [0, 0])
    np.testing.assert_allclose(_numpy(flat.criterion), [0.0, 0.0])


def test_unpenalised_eigen_torch_does_not_crash() -> None:
    # Growth accelerations with lam=0: girl 4 has no finite optimum (see the
    # NumPy regression test); the torch path must end with the warning too.
    from fdatools.datasets import load_growth
    from fdatools.smoothing import smooth

    growth = load_growth()
    age = np.asarray(growth.age, dtype=float)
    basis = BSpline(domain=(float(age[0]), float(age[-1])), order=6, breaks=age.tolist())
    heights = np.asarray(growth.hgtf, dtype=float)[:, :4]
    acc = smooth(heights, age, basis=basis, lam=0.01, penalty=4).fd.derivative(2)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = register(_as_torch(acc)[0], warp_basis=BSpline(domain=acc.domain, n_basis=8), lam=0.0)
    assert bool(torch.all(torch.isfinite(res.registered.coefs)))
    assert bool(torch.all(torch.isfinite(res.criterion)))
    # Without a finite optimum W runs off to hundreds, and e^{W - max W}
    # underflows on part of the domain: the warp is flat there in float64.
    _assert_monotone(res, acc.domain, strict=False)


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #


def test_gradients_flow_from_registered_curves_to_input_coefficients() -> None:
    fd = _curves(4)
    tfd, coefs = _as_torch(fd, requires_grad=True)
    res = register(tfd, warp_basis=BSpline(domain=DOMAIN, n_basis=5), lam=1.0)
    assert res.registered.coefs.requires_grad
    t = torch.linspace(2.0, 17.0, 60, dtype=torch.float64)
    loss = torch.sum(res.registered(t) ** 2)
    loss.backward()
    grad = _numpy(coefs.grad)
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0)
    # The warps are held fixed, so the loss is quadratic in the coefficients:
    # a directional finite difference through the same linear map matches.
    direction = np.random.default_rng(0).normal(size=grad.shape)
    step = 1e-3

    def fixed_warp_loss(values: np.ndarray) -> float:
        registered = register(
            FData(torch.tensor(values), fd.basis),
            warp_basis=BSpline(domain=DOMAIN, n_basis=5),
            lam=1.0,
            init=res.latent,
            max_iter=0,
        )
        return float(torch.sum(registered.registered(t) ** 2))

    base = np.asarray(fd.coefs)
    numeric = (
        fixed_warp_loss(base + step * direction) - fixed_warp_loss(base - step * direction)
    ) / (2 * step)
    assert numeric == pytest.approx(float(np.sum(grad * direction)), rel=1e-6)


def test_landmark_registration_returns_exact_gradients() -> None:
    fd = _curves(3)
    marks = np.array([10.5, 11.5, 12.5])
    tfd, coefs = _as_torch(fd, requires_grad=True)
    res = register(tfd, landmarks=marks)
    expected = register(fd, landmarks=marks)
    assert res.warp_inverse is not None
    assert expected.warp_inverse is not None
    np.testing.assert_allclose(
        _numpy(res.registered.coefs), np.asarray(expected.registered.coefs), rtol=1e-10, atol=1e-9
    )
    np.testing.assert_allclose(
        _numpy(res.warp_inverse.coefs), np.asarray(expected.warp_inverse.coefs), rtol=1e-12
    )
    assert res.target is None
    assert res.criterion is None
    assert res.n_iter is None
    _assert_monotone(res, DOMAIN)
    t = torch.linspace(2.0, 17.0, 40, dtype=torch.float64)
    torch.sum(res.registered(t) ** 2).backward()
    grad = _numpy(coefs.grad)
    direction = np.random.default_rng(1).normal(size=grad.shape)
    step = 1e-3

    def loss(values: np.ndarray) -> float:
        out = register(FData(values, fd.basis), landmarks=marks).registered(t.numpy())
        return float(np.sum(np.asarray(out) ** 2))

    base = np.asarray(fd.coefs)
    numeric = (loss(base + step * direction) - loss(base - step * direction)) / (2 * step)
    assert numeric == pytest.approx(float(np.sum(grad * direction)), rel=1e-6)


def test_numpy_result_warp_values_follow_torch_points() -> None:
    fd = _curves(2)
    res = register(fd, warp_basis=BSpline(domain=DOMAIN, n_basis=4), lam=1.0)
    t = torch.linspace(1.0, 18.0, 11, dtype=torch.float32)
    values = res.warp_values(t)
    assert isinstance(values, torch.Tensor)
    assert values.dtype == torch.float32
    np.testing.assert_allclose(_numpy(values), res.warp_values(t.numpy()), rtol=1e-6)
    assert isinstance(res.warp_values(np.linspace(1.0, 18.0, 3)), np.ndarray)


# --------------------------------------------------------------------------- #
# the autograd objective
# --------------------------------------------------------------------------- #


def _problem(criterion: str, periodic: bool, lam: float = 0.3) -> _CurveProblem:
    grid = np.linspace(0.0, 1.0, 401)
    if periodic:
        curve_basis: Any = Fourier(domain=(0.0, 1.0), n_basis=7)
        values = np.sin(2 * np.pi * (grid - 0.07)) + 0.3 * np.cos(4 * np.pi * grid)
        target = np.sin(2 * np.pi * grid)
    else:
        curve_basis = BSpline(domain=(0.0, 1.0), n_basis=12, order=5)
        values = np.exp(-(((grid - 0.45) / 0.15) ** 2))
        target = np.exp(-(((grid - 0.55) / 0.15) ** 2))
    curve = FData(np.linalg.lstsq(curve_basis(grid), values[:, None], rcond=None)[0], curve_basis)
    fine = _fine_grid((0.0, 1.0), curve_basis.n_basis)
    warp_basis = BSpline(domain=(0.0, 1.0), n_basis=6)
    return _CurveProblem(
        curve=curve,
        target=np.interp(fine, grid, target),
        quadrature=_WarpQuadrature(warp_basis, fine),
        penalty=_penalty_matrix(warp_basis, 2, lam),
        lam=lam,
        criterion=criterion,
        periodic=periodic,
        has_curvature=True,
    )


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("criterion", ["eigen", "least_squares"])
def test_autograd_derivatives_equal_analytic(criterion: str, periodic: bool) -> None:
    problem = _problem(criterion, periodic)
    size = problem.n_coefs - 1 + int(periodic)
    params = np.random.default_rng(5).normal(scale=0.4, size=size)
    value, grad, hess = AutogradObjective(problem)(params)
    exact_value, exact_grad, exact_hess = problem.evaluate(params)
    assert value == pytest.approx(exact_value, rel=1e-13)
    np.testing.assert_allclose(grad, exact_grad, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(hess, exact_hess, rtol=1e-10, atol=1e-12)


def test_autograd_objective_degenerate_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    objective = AutogradObjective(_problem("least_squares", False))
    # No parameters at all (a one-function warp basis, not periodic).
    monkeypatch.setattr(objective, "criterion", lambda p: 1.0 + torch.sum(p))
    value, grad, hess = objective(np.zeros(0))
    assert np.isfinite(value)
    assert grad.shape == (0,)
    assert hess.shape == (0, 0)
    # A criterion linear in the parameters has a constant gradient.
    monkeypatch.setattr(objective, "criterion", lambda p: torch.sum(3.0 * p))
    value, grad, hess = objective(np.ones(2))
    assert value == pytest.approx(6.0)
    np.testing.assert_array_equal(grad, [3.0, 3.0])
    np.testing.assert_array_equal(hess, np.zeros((2, 2)))
    # A criterion independent of the parameters.
    monkeypatch.setattr(objective, "criterion", lambda p: torch.tensor(2.0, dtype=torch.float64))
    value, grad, hess = objective(np.ones(2))
    assert value == 2.0
    np.testing.assert_array_equal(grad, [0.0, 0.0])


# --------------------------------------------------------------------------- #
# packaging
# --------------------------------------------------------------------------- #


def test_registration_does_not_import_torch() -> None:
    code = (
        "import sys, fdatools, fdatools.registration\n"
        "import numpy as np\n"
        "from fdatools import BSpline, FData\n"
        "fd = FData(np.random.default_rng(0).normal(size=(6, 2)), BSpline(n_basis=6))\n"
        "fdatools.registration.register(fd, criterion='least_squares')\n"
        "fdatools.registration.register(fd, landmarks=[0.4, 0.6])\n"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "False"


def test_registrator_accepts_torch_curves() -> None:
    fd = _curves(3)
    tfd, _ = _as_torch(fd)
    reg = Registrator(BSpline(domain=DOMAIN, n_basis=4), lam=1.0)
    out = reg.fit_transform(tfd)
    expected = Registrator(BSpline(domain=DOMAIN, n_basis=4), lam=1.0).fit_transform(fd)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=RTOL, atol=1e-9)
