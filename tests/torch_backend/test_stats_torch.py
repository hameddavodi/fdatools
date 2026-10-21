"""Torch backend behaviour of :mod:`fdatools.stats`.

:func:`~fdatools.stats.cov` and :func:`~fdatools.stats.cor` act on coefficients and
stay in the input namespace, so gradients flow through them; the depth and
permutation machinery samples curves and returns NumPy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from fdatools import BSpline, FData
from fdatools.stats import cor, cov, depth, t_test

torch = pytest.importorskip("torch", reason="torch extra not installed")

BASIS = BSpline(domain=(0.0, 1.0), n_basis=6)
GRID = np.linspace(0.0, 1.0, 5)


def _pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(6, 7)), rng.normal(size=(6, 7))


def test_cross_covariance_stays_in_torch_and_is_differentiable() -> None:
    a, b = _pair(0)
    ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    surface = cov(FData(ta, BASIS), FData(torch.tensor(b, dtype=torch.float64), BASIS))
    assert isinstance(surface.coefs, torch.Tensor)
    expected = cov(FData(a, BASIS), FData(b, BASIS)).coefs
    np.testing.assert_allclose(surface.coefs.detach().numpy(), expected, rtol=1e-12, atol=1e-14)
    surface.coefs.sum().backward()
    assert ta.grad is not None


def test_correlation_stays_in_torch() -> None:
    a, b = _pair(1)
    ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    matrix = cor(FData(ta, BASIS), FData(b, BASIS), s=GRID, t=GRID)
    assert isinstance(matrix, torch.Tensor)
    expected = cor(FData(a, BASIS), FData(b, BASIS), s=GRID, t=GRID)
    np.testing.assert_allclose(matrix.detach().numpy(), expected, rtol=1e-12)
    matrix.sum().backward()
    assert ta.grad is not None


def test_depth_and_tests_accept_torch_curves() -> None:
    a, b = _pair(2)
    fa = FData(torch.tensor(a, dtype=torch.float64), BASIS)
    fb = FData(torch.tensor(b, dtype=torch.float64), BASIS)
    np.testing.assert_allclose(depth(fa).depth, depth(FData(a, BASIS)).depth)
    result = t_test(fa, fb, n_perm=5, random_state=0)
    reference = t_test(FData(a, BASIS), FData(b, BASIS), n_perm=5, random_state=0)
    np.testing.assert_allclose(result.null, reference.null, rtol=1e-12)
