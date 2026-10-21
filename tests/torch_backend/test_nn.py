"""Tests for :mod:`fdatools.nn`: PyTorch layers and a dataset for functional data.

Every layer is checked with :func:`torch.autograd.gradcheck` in float64, which
compares the analytic Jacobian against central finite differences.  The device
test runs on CUDA or Apple MPS when one is present and is skipped otherwise.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Any, cast

import numpy as np
import pytest

from fdatools import LDO, BSpline, FData, Fourier
from fdatools.smoothing import smooth

pytest.importorskip("torch", reason="torch extra not installed")

import torch

from fdatools import nn as fnn

DOMAIN = (0.0, 1.0)
GRID = np.linspace(0.0, 1.0, 21)
INTERIOR = np.linspace(0.03, 0.97, 9)  # away from knots, where D_t is smooth


def spline(n_basis: int = 7) -> BSpline:
    return BSpline(domain=DOMAIN, n_basis=n_basis, order=4)


def rand(*shape: int, seed: int = 0, grad: bool = True) -> torch.Tensor:
    values = np.random.default_rng(seed).normal(size=shape)
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


def _gpu_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #


def test_importing_fdatools_does_not_import_torch() -> None:
    code = (
        "import sys, fdatools, fdatools.dynamics, fdatools.smoothing; print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "False"


def test_missing_torch_gives_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "fdatools.nn")
    with pytest.raises(ImportError, match=r"fdatools\[torch\]"):
        importlib.import_module("fdatools.nn")
    monkeypatch.undo()
    importlib.import_module("fdatools.nn")


# --------------------------------------------------------------------------- #
# BasisLayer
# --------------------------------------------------------------------------- #


def test_basis_layer_on_a_fixed_grid_matches_fdata() -> None:
    basis = spline()
    coefs = rand(3, 7, grad=False)
    layer = fnn.BasisLayer(basis, GRID)
    out = layer(coefs)
    assert out.shape == (3, len(GRID))
    expected = FData(coefs.numpy().T, basis)(GRID).T
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-12, atol=1e-13)


def test_basis_layer_derivative_and_batch_axes() -> None:
    basis = Fourier(domain=DOMAIN, n_basis=5)
    coefs = rand(2, 4, 5, grad=False)
    out = fnn.BasisLayer(basis, GRID, deriv=2)(coefs)
    assert out.shape == (2, 4, len(GRID))
    expected = FData(coefs.numpy().reshape(8, 5).T, basis)(GRID, 2).T.reshape(2, 4, -1)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-11, atol=1e-9)


def test_basis_layer_evaluates_at_points_given_to_forward() -> None:
    basis = spline()
    coefs = rand(2, 7, grad=False)
    t = torch.tensor(INTERIOR)
    out = fnn.BasisLayer(basis)(coefs, t)
    expected = FData(coefs.numpy().T, basis)(INTERIOR).T
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-12, atol=1e-13)


def test_basis_layer_with_an_operator() -> None:
    basis = spline(9)
    coefs = rand(1, 9, grad=False)
    op = LDO(weights=[2.0, 0.0])
    out = fnn.BasisLayer(basis, GRID, deriv=op)(coefs)
    expected = FData(coefs.numpy().T, basis)(GRID, op).T
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-11, atol=1e-10)


def test_evaluation_points_must_be_a_vector() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        fnn.BasisLayer(spline(), np.ones((3, 2)))


def test_basis_layer_needs_points() -> None:
    with pytest.raises(ValueError, match="evaluation points"):
        fnn.BasisLayer(spline())(rand(1, 7, grad=False))


def test_basis_layer_checks_the_coefficient_axis() -> None:
    with pytest.raises(ValueError, match="7"):
        fnn.BasisLayer(spline(), GRID)(rand(2, 5, grad=False))


def test_basis_layer_gradcheck_coefficients() -> None:
    layer = fnn.BasisLayer(spline(), GRID, deriv=1)
    assert torch.autograd.gradcheck(layer, (rand(3, 7),))


def test_basis_layer_gradcheck_coefficients_and_points() -> None:
    layer = fnn.BasisLayer(spline())
    t = torch.tensor(INTERIOR, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (rand(2, 7), t))


def test_basis_layer_repr_and_buffers() -> None:
    layer = fnn.BasisLayer(spline(), GRID)
    assert "BSpline" in repr(layer)
    assert "design" in dict(layer.named_buffers())
    assert list(layer.parameters()) == []


# --------------------------------------------------------------------------- #
# SmoothingLayer
# --------------------------------------------------------------------------- #


def test_smoothing_layer_matches_smooth() -> None:
    basis = spline(9)
    y = np.sin(2 * np.pi * GRID)[:, None] * np.array([1.0, 2.0, -0.5])
    layer = fnn.SmoothingLayer(basis, GRID, lam=1e-3)
    coefs = layer(torch.tensor(y.T))
    expected = np.asarray(smooth(y, GRID, basis=basis, lam=1e-3).fd.coefs).T
    np.testing.assert_allclose(coefs.numpy(), expected, rtol=1e-10, atol=1e-12)


def test_smoothing_layer_with_weights_and_operator_matches_smooth() -> None:
    basis = spline(9)
    weights = np.linspace(0.5, 2.0, len(GRID))
    y = np.cos(3 * GRID)
    op = LDO.harmonic(period=1.0)
    layer = fnn.SmoothingLayer(basis, GRID, lam=1e-4, penalty=op, weights=weights)
    coefs = layer(torch.tensor(y)[None, :])
    fit = smooth(y, GRID, basis=basis, lam=1e-4, penalty=op, weights=weights)
    np.testing.assert_allclose(coefs.numpy()[0], np.asarray(fit.fd.coefs)[:, 0], rtol=1e-9)


def test_trainable_smoothing_parameter_matches_the_fixed_one() -> None:
    basis = spline(9)
    y = torch.tensor(np.exp(GRID))[None, :]
    fixed = fnn.SmoothingLayer(basis, GRID, lam=0.1)
    trainable = fnn.SmoothingLayer(basis, GRID, lam=0.1, trainable_lam=True)
    assert [name for name, _ in trainable.named_parameters()] == ["log_lam"]
    assert "trainable_lam=True" in repr(trainable)
    assert "lam=0.1" in repr(fixed)
    assert trainable.lam == pytest.approx(0.1)
    torch.testing.assert_close(trainable(y), fixed(y), rtol=1e-10, atol=1e-12)


def test_trainable_smoothing_parameter_learns() -> None:
    basis = spline(9)
    rng = np.random.default_rng(4)
    clean = np.sin(2 * np.pi * GRID)
    y = torch.tensor(clean + 0.05 * rng.normal(size=GRID.shape))[None, :]
    target = torch.tensor(clean)[None, :]
    smoother = fnn.SmoothingLayer(basis, GRID, lam=1e-8, trainable_lam=True)
    evaluate = fnn.BasisLayer(basis, GRID)
    before = smoother.lam
    optimiser = torch.optim.Adam(smoother.parameters(), lr=0.5)
    for _ in range(40):
        optimiser.zero_grad()
        loss = ((evaluate(smoother(y)) - target) ** 2).mean()
        loss.backward()
        optimiser.step()
    assert smoother.lam > before


def test_smoothing_layer_validation() -> None:
    with pytest.raises(ValueError, match="lam"):
        fnn.SmoothingLayer(spline(), GRID, lam=-1.0)
    with pytest.raises(ValueError, match="positive"):
        fnn.SmoothingLayer(spline(), GRID, lam=0.0, trainable_lam=True)
    with pytest.raises(ValueError, match="weights"):
        fnn.SmoothingLayer(spline(), GRID, weights=np.ones(3))
    with pytest.raises(ValueError, match="observation"):
        fnn.SmoothingLayer(spline(), GRID)(torch.ones(2, 5, dtype=torch.float64))


def test_smoothing_layer_gradcheck_observations() -> None:
    layer = fnn.SmoothingLayer(spline(), GRID, lam=1e-2)
    assert torch.autograd.gradcheck(layer, (rand(2, len(GRID)),))


def test_smoothing_layer_gradcheck_smoothing_parameter() -> None:
    layer = fnn.SmoothingLayer(spline(), GRID, lam=1e-2, trainable_lam=True)
    y = rand(2, len(GRID), grad=False)

    def as_function(log_lam: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.func.functional_call(layer, {"log_lam": log_lam}, (obs,))
        return out

    log_lam = torch.tensor(np.log(1e-2), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(as_function, (log_lam, y.requires_grad_(True)))


# --------------------------------------------------------------------------- #
# FDataDataset
# --------------------------------------------------------------------------- #


def test_dataset_yields_coefficients_and_labels() -> None:
    basis = spline()
    coefs = np.random.default_rng(1).normal(size=(7, 5))
    labels = np.arange(5) % 2
    ds = fnn.FDataDataset(FData(coefs, basis), labels)
    assert len(ds) == 5
    x, y = ds[3]
    assert x.dtype == torch.float64
    np.testing.assert_allclose(x.numpy(), coefs[:, 3])
    assert int(y) == 1


def test_dataset_without_labels_yields_the_inputs_only() -> None:
    ds = fnn.FDataDataset(FData(np.eye(7)[:, :3], spline()))
    x = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (7,)


def test_dataset_on_a_grid_and_through_a_dataloader() -> None:
    basis = spline()
    fd = FData(np.random.default_rng(2).normal(size=(7, 10)), basis)
    ds = fnn.FDataDataset(fd, np.linspace(0.0, 1.0, 10), t=GRID, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    x, y = next(iter(loader))
    assert x.shape == (4, len(GRID))
    assert x.dtype == torch.float32
    np.testing.assert_allclose(x.numpy(), fd(GRID)[:, :4].T, rtol=1e-6, atol=1e-6)
    assert y.shape == (4,)


def test_dataset_multivariate_curves() -> None:
    fd = FData(np.ones((7, 4, 2)), spline())
    assert fnn.FDataDataset(fd)[0].shape == (7, 2)
    assert fnn.FDataDataset(fd, t=GRID)[0].shape == (len(GRID), 2)


def test_dataset_keeps_non_numeric_labels() -> None:
    ds = fnn.FDataDataset(FData(np.eye(7)[:, :2], spline()), ["a", "b"])
    assert ds[1][1] == "b"


def test_dataset_accepts_torch_coefficients_and_detaches_them() -> None:
    fd = FData(rand(7, 3), spline())
    x = fnn.FDataDataset(fd)[0]
    assert not x.requires_grad


def test_dataset_checks_label_count() -> None:
    with pytest.raises(ValueError, match="labels"):
        fnn.FDataDataset(FData(np.eye(7)[:, :3], spline()), [0, 1])


def test_dataset_rejects_non_fdata() -> None:
    with pytest.raises(TypeError, match="FData"):
        fnn.FDataDataset(cast(Any, np.ones((7, 3))))


# --------------------------------------------------------------------------- #
# devices
# --------------------------------------------------------------------------- #


@pytest.mark.gpu
@pytest.mark.skipif(_gpu_device() is None, reason="no CUDA or MPS device available")
def test_layers_run_on_an_accelerator() -> None:
    device = _gpu_device()
    dtype = torch.float32 if device == "mps" else torch.float64
    basis = spline()
    coefs = torch.tensor(np.random.default_rng(3).normal(size=(2, 7)), dtype=dtype)
    expected = FData(coefs.double().numpy().T, basis)(GRID).T

    layer = fnn.BasisLayer(basis, GRID, dtype=dtype).to(device)
    out = layer(coefs.to(device))
    assert out.device.type == device
    np.testing.assert_allclose(out.cpu().double().numpy(), expected, rtol=1e-5, atol=1e-5)

    moving = fnn.BasisLayer(basis, dtype=dtype).to(device)
    t = torch.tensor(GRID, dtype=dtype, device=device, requires_grad=True)
    c = coefs.to(device).requires_grad_(True)
    values = moving(c, t)
    assert values.device.type == device
    values.sum().backward()
    assert t.grad is not None
    assert c.grad is not None

    y = torch.tensor(np.sin(GRID)[None, :], dtype=dtype, device=device)
    smoother = fnn.SmoothingLayer(basis, GRID, lam=1e-3, dtype=dtype).to(device)
    trainable = fnn.SmoothingLayer(basis, GRID, lam=1e-3, trainable_lam=True, dtype=dtype)
    trainable = trainable.to(device)
    np.testing.assert_allclose(
        smoother(y).cpu().numpy(), trainable(y).detach().cpu().numpy(), rtol=1e-4, atol=1e-5
    )
