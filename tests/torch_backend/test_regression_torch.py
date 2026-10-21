"""Torch backend behaviour of :mod:`fdatools.regression`.

A PyTorch tensor among the inputs of :func:`~fdatools.regression.fregress` or
:func:`~fdatools.regression.linmod` (response, covariates or weights) makes the
whole fit run in PyTorch: the coefficients, fitted values and predictions are
tensors equal to the NumPy results, and gradients reach the inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from fdatools import LDO, BSpline, FData, Fourier
from fdatools.regression import FRegressResult, fregress, linmod

torch = pytest.importorskip("torch", reason="torch extra not installed")

BASIS = BSpline(domain=(0.0, 1.0), n_basis=6)
RTOL = 1e-10


def _data(seed: int = 0, n: int = 14) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "x": rng.standard_normal((6, n)),
        "y": rng.standard_normal((6, n)),
        "ys": rng.standard_normal(n),
        "z": rng.standard_normal(n),
        "w": rng.uniform(0.5, 2.0, n),
    }


def _leaf(value: np.ndarray) -> Any:
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _as_numpy(value: Any) -> np.ndarray:
    return np.asarray(value.detach().numpy() if isinstance(value, torch.Tensor) else value)


def _values(result: Any) -> Any:
    return result.coefs if isinstance(result, FData) else result


# (name, builder) -- the builder turns (x, y, scalar y, z, weights) into a fit
_MODELS: dict[str, Callable[[Any, Any, Any, Any, Any], FRegressResult]] = {
    "scalar_on_function": lambda x, y, ys, z, w: fregress(
        ys, [1.0, z, FData(x, BASIS)], beta=[None, None, (BASIS, 1e-3)], weights=w
    ),
    "function_on_scalar": lambda x, y, ys, z, w: fregress(
        FData(y, BASIS), [1.0, z], lam=1e-3, weights=w
    ),
    "concurrent": lambda x, y, ys, z, w: fregress(
        FData(y, BASIS), {"const": 1.0, "x": FData(x, BASIS)}, lam=1e-3, weights=w
    ),
}


#: The inputs each model reads, all of which must receive a gradient.
_INPUTS = {
    "scalar_on_function": ("x", "ys", "z", "w"),
    "function_on_scalar": ("y", "z", "w"),
    "concurrent": ("x", "y", "w"),
}


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_fregress_torch_equals_numpy_and_is_differentiable(name: str) -> None:
    data = _data()
    reference = _MODELS[name](data["x"], data["y"], data["ys"], data["z"], data["w"])
    leaves = {key: _leaf(value) for key, value in data.items()}
    model = _MODELS[name](leaves["x"], leaves["y"], leaves["ys"], leaves["z"], leaves["w"])
    for got, want in zip(model.beta, reference.beta, strict=True):
        assert isinstance(got.coefs, torch.Tensor)
        np.testing.assert_allclose(_as_numpy(got.coefs), want.coefs, rtol=RTOL, atol=1e-13)
    fitted = _values(model.fitted)
    assert isinstance(fitted, torch.Tensor)
    np.testing.assert_allclose(_as_numpy(fitted), _values(reference.fitted), rtol=RTOL, atol=1e-13)
    assert isinstance(model.cmat, torch.Tensor)
    assert isinstance(model.coefficients, torch.Tensor)
    if reference.df is not None:
        assert model.df == pytest.approx(reference.df, rel=RTOL)
        assert model.gcv == pytest.approx(reference.gcv, rel=RTOL)
        assert model.ocv == pytest.approx(reference.ocv, rel=RTOL)
    loss = sum((b.coefs**2).sum() for b in model.beta) + (fitted**2).sum()
    loss.backward()
    for key in _INPUTS[name]:
        grad = leaves[key].grad
        assert grad is not None, key
        assert bool(torch.all(torch.isfinite(grad))), key


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_fregress_torch_predict_stderr_cv(name: str) -> None:
    data = _data(1)
    reference = _MODELS[name](data["x"], data["y"], data["ys"], data["z"], data["w"])
    tensors = {key: torch.tensor(value, dtype=torch.float64) for key, value in data.items()}
    model = _MODELS[name](tensors["x"], tensors["y"], tensors["ys"], tensors["z"], tensors["w"])
    new = _data(2)
    covariates: dict[str, list[Any]] = {
        "scalar_on_function": [1.0, new["z"], FData(new["x"], BASIS)],
        "function_on_scalar": [1.0, new["z"]],
        "concurrent": [1.0, FData(new["x"], BASIS)],
    }
    predicted = _values(model.predict(covariates[name]))
    assert isinstance(predicted, torch.Tensor)
    np.testing.assert_allclose(
        _as_numpy(predicted),
        _values(reference.predict(covariates[name])),
        rtol=RTOL,
        atol=1e-13,
    )
    y2c = None if name == "scalar_on_function" else np.eye(6)
    se, se_ref = model.stderr(0.1, y2c), reference.stderr(0.1, y2c)
    assert isinstance(se.cov, torch.Tensor)
    np.testing.assert_allclose(_as_numpy(se.cov), se_ref.cov, rtol=1e-9, atol=1e-14)
    cv, cv_ref = model.cv(), reference.cv()
    assert cv.sse == pytest.approx(cv_ref.sse, rel=1e-9)
    assert isinstance(_values(cv.errors), torch.Tensor)


def test_numpy_model_predicts_torch_covariates_in_torch() -> None:
    data = _data(3)
    model = fregress(data["ys"], [1.0, FData(data["x"], BASIS)], beta=[None, (BASIS, 1e-2)])
    tx = _leaf(data["x"])
    predicted = model.predict([1.0, FData(tx, BASIS)])
    assert isinstance(predicted, torch.Tensor)
    np.testing.assert_allclose(_as_numpy(predicted), model.fitted, rtol=RTOL)
    predicted.sum().backward()
    assert tx.grad is not None


def test_fregress_gradient_matches_finite_differences() -> None:
    data = _data(4, n=8)
    z = torch.tensor(data["z"], dtype=torch.float64)

    def slope(y: Any) -> Any:
        return fregress(FData(y, BASIS), [1.0, z], lam=1e-2).beta[1].coefs

    assert torch.autograd.gradcheck(slope, (_leaf(data["y"]),))


# --------------------------------------------------------------------------- #
# linmod
# --------------------------------------------------------------------------- #

TBASIS = Fourier(domain=(0.0, 2.0), n_basis=5)


def _linmod(x: Any, y: Any, w: Any = None) -> Any:
    return linmod(
        FData(y, TBASIS),
        FData(x, BASIS),
        s_basis=BSpline(domain=(0.0, 1.0), n_basis=4),
        lam_alpha=1e-2,
        lam_s=1e-3,
        lam_t=1e-3,
        penalty_t=LDO.harmonic(period=2.0),
        weights=w,
    )


def test_linmod_torch_equals_numpy_and_is_differentiable() -> None:
    rng = np.random.default_rng(5)
    x, y, w = rng.standard_normal((6, 12)), rng.standard_normal((5, 12)), rng.uniform(1, 2, 12)
    reference = _linmod(x, y, w)
    tx, ty, tw = _leaf(x), _leaf(y), _leaf(w)
    model = _linmod(tx, ty, tw)
    for got, want in (
        (model.alpha.coefs, reference.alpha.coefs),
        (model.beta.coefs, reference.beta.coefs),
        (model.fitted.coefs, reference.fitted.coefs),
        (model.predict(FData(tx, BASIS)).coefs, reference.fitted.coefs),
        (model.residuals.coefs, reference.residuals.coefs),
    ):
        assert isinstance(got, torch.Tensor)
        np.testing.assert_allclose(_as_numpy(got), want, rtol=RTOL, atol=1e-13)
    loss = (model.beta.coefs**2).sum() + (model.fitted.coefs**2).sum()
    loss.backward()
    for leaf in (tx, ty, tw):
        assert leaf.grad is not None
        assert bool(torch.all(torch.isfinite(leaf.grad)))


def test_linmod_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(6)
    y = torch.tensor(rng.standard_normal((5, 9)), dtype=torch.float64)

    def surface(x: Any) -> Any:
        return _linmod(x, y).beta.coefs

    assert torch.autograd.gradcheck(surface, (_leaf(rng.standard_normal((6, 9))),))
