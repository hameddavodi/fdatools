"""Gradient-flow tests for the torch backend (SPEC 5.2).

Every operation that returns coefficients or an inner product must stay
differentiable with respect to the coefficients it was built from.  The checks
are gradcheck-style: an analytic gradient from ``backward()`` against a central
finite difference of the same scalar functional.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fdatools import BSpline, FData, Fourier, inprod

torch = pytest.importorskip("torch", reason="torch extra not installed")

DOMAIN = (0.0, 1.0)
GRID = np.linspace(0.05, 0.95, 11)


def coefs(n_basis: int = 6, n_curves: int = 2, seed: int = 3) -> Any:
    """Return a random torch coefficient matrix that tracks gradients."""
    values = np.random.default_rng(seed).normal(size=(n_basis, n_curves))
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def check_gradient(scalar: Any, c: Any, *, rtol: float = 1e-6) -> np.ndarray:
    """Compare ``d scalar / d c`` against a central finite difference."""
    assert scalar.requires_grad, "the result was detached from the graph"
    scalar.backward()
    assert c.grad is not None
    analytic: np.ndarray = c.grad.detach().numpy().copy()
    assert np.all(np.isfinite(analytic))
    assert np.any(analytic != 0.0), "gradient is identically zero"
    return analytic


def finite_difference(fn: Any, values: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central finite difference of ``fn`` at ``values``."""
    out = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        up, down = values.copy(), values.copy()
        up[index] += h
        down[index] -= h
        out[index] = (fn(up) - fn(down)) / (2.0 * h)
    return out


# --------------------------------------------------------------------------- #
# already working before this change -- kept as regression guards
# --------------------------------------------------------------------------- #


def test_gradient_flows_through_evaluation() -> None:
    c = coefs()
    fd = FData(c, BSpline(domain=DOMAIN, n_basis=6))
    check_gradient(fd(torch.tensor(GRID)).sum(), c)


def test_gradient_flows_through_derivative() -> None:
    c = coefs()
    fd = FData(c, BSpline(domain=DOMAIN, n_basis=6))
    check_gradient(fd.derivative(2)(torch.tensor(GRID)).sum(), c)


def test_gradient_flows_through_mean_and_center() -> None:
    c = coefs()
    fd = FData(c, BSpline(domain=DOMAIN, n_basis=6))
    check_gradient((fd.mean().coefs.sum() + fd.center().coefs.pow(2).sum()), c)


def test_gradient_flows_with_numpy_coefs_and_torch_points() -> None:
    t = torch.tensor(GRID, requires_grad=True)
    fd = FData(np.eye(6), BSpline(domain=DOMAIN, n_basis=6))
    values = fd(t)
    assert values.requires_grad
    values.sum().backward()
    assert t.grad is not None


# --------------------------------------------------------------------------- #
# the paths that used to be cut by to_numpy
# --------------------------------------------------------------------------- #


def test_gradient_flows_through_inprod() -> None:
    c = coefs()
    fd = FData(c, BSpline(domain=DOMAIN, n_basis=6))
    analytic = check_gradient(inprod(fd, fd).sum(), c)

    def functional(values: np.ndarray) -> float:
        curve = FData(values, BSpline(domain=DOMAIN, n_basis=6))
        return float(np.sum(inprod(curve, curve)))

    numeric = finite_difference(functional, c.detach().numpy())
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_gradient_flows_through_inprod_with_operators() -> None:
    c = coefs()
    fd = FData(c, BSpline(domain=DOMAIN, n_basis=6, order=5))
    check_gradient(inprod(fd, fd, lfd1=1, lfd2=2).sum(), c)


def test_gradient_flows_through_matmul() -> None:
    c = coefs(n_basis=7)
    fd = FData(c, Fourier(domain=DOMAIN, n_basis=7))
    check_gradient((fd @ fd).sum(), c)


def test_gradient_flows_through_the_product() -> None:
    c = coefs(n_basis=5, n_curves=1)
    basis = BSpline(domain=DOMAIN, n_basis=5)
    analytic = check_gradient((FData(c, basis) * FData(c, basis)).coefs.sum(), c)

    def functional(values: np.ndarray) -> float:
        curve = FData(values, basis)
        return float(np.sum((curve * curve).coefs))

    numeric = finite_difference(functional, c.detach().numpy())
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_gradient_flows_through_an_integer_power() -> None:
    c = coefs(n_basis=5, n_curves=1)
    fd = FData(c, BSpline(domain=DOMAIN, n_basis=5))
    check_gradient((fd**3).coefs.sum(), c)


def test_gradient_flows_through_a_fractional_power() -> None:
    values = np.abs(np.random.default_rng(11).normal(size=(5, 1))) + 1.0
    c = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    basis = BSpline(domain=DOMAIN, n_basis=5)
    analytic = check_gradient((FData(c, basis) ** 0.5).coefs.sum(), c)

    def functional(raw: np.ndarray) -> float:
        return float(np.sum((FData(raw, basis) ** 0.5).coefs))

    numeric = finite_difference(functional, values, h=1e-6)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_gradient_flows_through_std() -> None:
    c = coefs(n_basis=5, n_curves=4)
    basis = BSpline(domain=DOMAIN, n_basis=5)
    analytic = check_gradient(FData(c, basis).std().coefs.sum(), c)

    def functional(values: np.ndarray) -> float:
        return float(np.sum(FData(values, basis).std().coefs))

    numeric = finite_difference(functional, c.detach().numpy())
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_gradient_flows_through_cov() -> None:
    c = coefs(n_basis=5, n_curves=4)
    fd = FData(c, BSpline(domain=DOMAIN, n_basis=5))
    check_gradient(fd.cov().coefs.sum(), c)


def test_gradient_flows_through_a_tensor_scalar_multiplier() -> None:
    scale = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    fd = FData(np.eye(5), BSpline(domain=DOMAIN, n_basis=5))
    out = (fd * scale).coefs.sum()
    assert out.requires_grad
    out.backward()
    assert scale.grad is not None
    np.testing.assert_allclose(float(scale.grad), 5.0, rtol=1e-12)


def test_gradient_flows_through_a_tensor_scalar_addend() -> None:
    shift = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
    fd = FData(np.zeros((5, 1)), BSpline(domain=DOMAIN, n_basis=5))
    out = (fd + shift)(torch.tensor(GRID)).sum()
    assert out.requires_grad
    out.backward()
    assert shift.grad is not None
    np.testing.assert_allclose(float(shift.grad), float(len(GRID)), rtol=1e-9)


def test_penalty_in_the_torch_namespace_is_a_tensor() -> None:
    basis = BSpline(domain=DOMAIN, n_basis=6)
    assert isinstance(basis.penalty(2, xp=torch), torch.Tensor)
