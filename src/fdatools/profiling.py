r"""Profiling: parameter estimation for nonlinear ODEs by generalized profiling.

A system of ``d`` ordinary differential equations

.. math::

    \dot x_i(t) = f_i\bigl(x(t), t, \theta\bigr), \qquad i = 1, \dots, d,

is fitted to noisy observations of some (not necessarily all) of its states by
the *generalized profiling* method of Ramsay, Hooker, Campbell & Cao (2007,
JRSS-B 69: 741-796; Ramsay, Hooker & Graves 2009, chapter 11).  Each state is a
basis expansion :math:`x_i(t) = \phi_i(t)^T c_i`, and the fit has two levels.

**Inner problem.**  For a fixed parameter vector :math:`\theta` the
coefficients minimise the penalised criterion

.. math::

    J(c \mid \theta) = \sum_{i} w_i \Bigl[ \sum_j \bigl(y_{ij} - x_i(t_{ij})\bigr)^2
        + \lambda_i \int \bigl(\dot x_i(t) - f_i(x(t), t, \theta)\bigr)^2 dt \Bigr],

where :math:`w_i` scales state ``i`` (R's ``Cwt``/``Twt`` are :math:`1/w_i`)
and the integral is computed by a quadrature rule (composite Simpson on the
B-spline break points by default, as R's ``quadset``).  ``J`` is a sum of
squared residuals and is minimised by damped Gauss-Newton.  An unobserved
state has no data term; it is identified through the equations only.

**Outer problem.**  The fitted coefficients :math:`\hat c(\theta)` are an
implicit function of :math:`\theta`, and :math:`\theta` minimises the profiled
data criterion

.. math::

    H(\theta) = \sum_i w_i \sum_j \bigl(y_{ij} - \phi_i(t_{ij})^T \hat c_i(\theta)\bigr)^2 .

Its Jacobian comes from the implicit function theorem: at the inner optimum
:math:`\partial J / \partial c = 0`, so

.. math::

    \frac{d\hat c}{d\theta} = -\Bigl(\frac{\partial^2 J}{\partial c\,\partial c^T}\Bigr)^{-1}
        \frac{\partial^2 J}{\partial c\,\partial \theta^T},

with both second-derivative matrices exact (they include the second derivatives
of ``f``).  Gauss-Newton on :math:`H` then gives :math:`\hat\theta` and its
approximate covariance :math:`\hat\sigma^2 (A^T A)^{-1}`,
:math:`A = d e / d\theta`, :math:`\hat\sigma^2 = H(\hat\theta) / (N - p)`.

This replaces R's CSTR family (``CSTR2in``, ``CSTR2``, ``CSTRfitLS``,
``CSTRfn``, ``CSTRres``, ``CSTRsse``) with one general interface: the user
supplies the right-hand side as an :class:`ODEModel`.  The continuously stirred
tank reactor is provided as :func:`cstr_model` (with its input scenarios in
:func:`cstr_inputs`), and the FitzHugh-Nagumo neuron as
:func:`fitzhugh_nagumo_model`.

All arithmetic is done in float64 NumPy (through the array-API namespace); the
estimation is an iterative optimisation, not a differentiable map, so PyTorch
tensors are accepted as data but results are NumPy arrays.  A right-hand side
written in PyTorch is supported through :meth:`ODEModel.from_torch`, which
obtains every derivative by automatic differentiation.

Examples
--------
>>> import numpy as np
>>> import fdatools as fdt
>>> from fdatools.profiling import ODEModel, profile_ode
>>> decay = ODEModel(
...     rhs=lambda x, t, theta: -theta[0] * x,
...     n_states=1,
...     n_params=1,
...     jac_x=lambda x, t, theta: np.full((len(t), 1, 1), -theta[0]),
...     jac_theta=lambda x, t, theta: -x[:, :, None],
... )
>>> t = np.linspace(0.0, 2.0, 41)
>>> y = 3.0 * np.exp(-1.5 * t)
>>> basis = fdt.BSpline(domain=(0.0, 2.0), breaks=np.linspace(0.0, 2.0, 21))
>>> fit = profile_ode(decay, t, y[:, None], basis, lam=1e4, theta0=[1.0])
>>> round(float(fit.theta[0]), 4)
1.5
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import ModuleType
from typing import Any, Literal, NamedTuple

from scipy.integrate import solve_ivp

from fdatools import _linalg
from fdatools._backend import asarray, default_namespace, to_numpy
from fdatools.basis import Basis, BSpline
from fdatools.core import FData

__all__ = [
    "InnerFit",
    "ODEModel",
    "ProfileResult",
    "ProfiledODE",
    "Residuals",
    "cstr_inputs",
    "cstr_model",
    "fitzhugh_nagumo_model",
    "profile_ode",
    "simpson_rule",
]

Array = Any
Function = Callable[[Array, Array, Array], Any]
OdeMethod = Literal["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"]

#: Relative step of the central differences used for a derivative the model
#: does not supply (the cube root of the float64 machine epsilon).
_FD_STEP = 6.0554544523933395e-06

#: Tolerances handed to :func:`scipy.integrate.solve_ivp` by :meth:`ODEModel.simulate`.
_ODE_RTOL = 1e-10
_ODE_ATOL = 1e-12

#: Largest Marquardt damping tried before an inner or outer step is abandoned.
_MAX_DAMPING = 1e16

#: Relative coefficient step below which the inner Gauss-Newton takes full steps
#: without requiring a decrease (the criterion then changes at rounding level).
_FULL_STEP = 1e-6

#: Relative coefficient step below which an inner step that no longer halves
#: counts as converged (rounding level for ill-conditioned problems).
_STALL_STEP = 1e-12

#: Relative parameter step (max-norm) below which the outer iteration has converged.
_STEP_TOL = 1e-10

#: Largest relative decrease of the profiled criterion, predicted by the
#: Gauss-Newton model, that still counts as converged when no step can lower
#: the criterion (the inner fits are exact only to rounding level).
_STALL_DECREASE = 1e-10

#: Relative slack when checking that observation times lie in a basis domain.
_DOMAIN_TOL = 1e-12

#: Sub-intervals per basis function of the default quadrature grid when no state
#: has a B-spline basis (whose break points would otherwise define the panels).
_PANELS_PER_BASIS = 4

#: The input variables of the CSTR, in column order.
CSTR_INPUTS = ("F", "CA0", "T0", "Tcin", "Fc")

#: R's CSTR input scenarios that fdatools reproduces (see :func:`cstr_inputs`).
CSTR_CONDITIONS = ("all.cool.step", "all.hot.step", "Tc.hot.step", "Tc.cool.step")

#: The CSTR constants and their defaults, as in the example of R's ``CSTR`` help
#: page (Marlin 2000): volume, heat capacity, density, reaction enthalpy,
#: coolant heat capacity and density, and the reference temperature.
CSTR_CONSTANTS = {
    "V": 1.0,
    "Cp": 1.0,
    "rho": 1.0,
    "delH": -130.0,
    "Cpc": 1.0,
    "rhoc": 1.0,
    "Tref": 350.0,
}

#: The CSTR parameters and their true values in R's ``CSTR`` example.
CSTR_PARAMETERS = {"kref": 0.4610, "EoverR": 0.83301, "a": 1.678, "b": 0.5}


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #


def _names(given: Sequence[str], prefix: str, n: int, what: str) -> tuple[str, ...]:
    """Return ``given`` as a tuple, or default names ``prefix0 ...``, checking the count."""
    if not given:
        return tuple(f"{prefix}{i}" for i in range(n))
    names = tuple(str(g) for g in given)
    if len(names) != n:
        raise ValueError(f"expected {n} {what} names, got {len(names)}")
    return names


def _checked(value: Any, shape: tuple[int, ...], what: str) -> Array:
    """Convert a model output to float64 and check its shape."""
    out = asarray(to_numpy(value), xp=default_namespace())
    if tuple(out.shape) != shape:
        raise ValueError(f"the model's {what} returned shape {tuple(out.shape)}, expected {shape}")
    return out


@dataclass(frozen=True, eq=False)
class ODEModel:
    r"""The right-hand side of an ODE system :math:`\dot x = f(x, t, \theta)`.

    Every callable receives the states ``x`` as an ``(n, d)`` array, the times
    ``t`` as an ``(n,)`` array and the parameters ``theta`` as a ``(p,)``
    array, and is evaluated at ``n`` points at once.  The derivatives are
    optional: any that is omitted is computed by central finite differences
    (of ``rhs`` for the Jacobians, of the Jacobians for the second
    derivatives), which is accurate to about 1e-10 relative for the Jacobians
    and 1e-6 for the second derivatives.  Supplying them analytically, or
    building the model with :meth:`from_torch`, makes the fit exact.

    Parameters
    ----------
    rhs : callable
        ``rhs(x, t, theta)`` returning ``(n, d)`` derivatives.
    n_states : int
        Number of states ``d``.
    n_params : int
        Number of parameters ``p``.
    jac_x : callable, optional
        ``(n, d, d)`` array, ``[q, i, k]`` = :math:`\partial f_i / \partial x_k`.
    jac_theta : callable, optional
        ``(n, d, p)`` array, ``[q, i, j]`` = :math:`\partial f_i / \partial \theta_j`.
    hess_xx : callable, optional
        ``(n, d, d, d)`` array, ``[q, i, k, l]`` =
        :math:`\partial^2 f_i / \partial x_k \partial x_l`.
    hess_xtheta : callable, optional
        ``(n, d, d, p)`` array, ``[q, i, k, j]`` =
        :math:`\partial^2 f_i / \partial x_k \partial \theta_j`.
    state_names, param_names : sequence of str, optional
        Labels; default ``x0, x1, ...`` and ``theta0, theta1, ...``.

    Raises
    ------
    ValueError
        If a count is not positive or a name list has the wrong length.

    Examples
    --------
    >>> import numpy as np
    >>> from fdatools.profiling import ODEModel
    >>> growth = ODEModel(lambda x, t, th: th[0] * x, n_states=1, n_params=1)
    >>> growth(np.array([[2.0]]), np.array([0.0]), np.array([0.5])).tolist()
    [[1.0]]
    >>> round(
    ...     float(
    ...         growth.jacobians(np.array([[2.0]]), np.array([0.0]), np.array([0.5]))[1][0, 0, 0]
    ...     ),
    ...     6,
    ... )
    2.0
    """

    rhs: Function
    n_states: int
    n_params: int
    jac_x: Function | None = None
    jac_theta: Function | None = None
    hess_xx: Function | None = None
    hess_xtheta: Function | None = None
    state_names: tuple[str, ...] = ()
    param_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the counts and fill in default names."""
        if int(self.n_states) < 1 or int(self.n_params) < 1:
            raise ValueError(
                f"n_states and n_params must be positive, got {self.n_states} and {self.n_params}"
            )
        object.__setattr__(self, "n_states", int(self.n_states))
        object.__setattr__(self, "n_params", int(self.n_params))
        object.__setattr__(
            self, "state_names", _names(self.state_names, "x", self.n_states, "state")
        )
        object.__setattr__(
            self, "param_names", _names(self.param_names, "theta", self.n_params, "parameter")
        )

    # ---------------------------------------------------------------- inputs

    def _arguments(self, x: Any, t: Any, theta: Any) -> tuple[Array, Array, Array]:
        """Convert and check ``(x, t, theta)``."""
        xp = default_namespace()
        xs = asarray(to_numpy(x), xp=xp)
        ts = asarray(to_numpy(t), xp=xp)
        th = asarray(to_numpy(theta), xp=xp)
        if len(xs.shape) != 2 or xs.shape[1] != self.n_states:
            raise ValueError(f"x must have shape (n, {self.n_states}), got {tuple(xs.shape)}")
        if tuple(ts.shape) != (xs.shape[0],):
            raise ValueError(f"t must have shape ({xs.shape[0]},), got {tuple(ts.shape)}")
        if tuple(th.shape) != (self.n_params,):
            raise ValueError(f"theta must have shape ({self.n_params},), got {tuple(th.shape)}")
        return xs, ts, th

    def __call__(self, x: Any, t: Any, theta: Any) -> Array:
        r"""Evaluate :math:`f(x, t, \theta)` at ``n`` points.

        Parameters
        ----------
        x : array_like
            States, ``(n, d)``.
        t : array_like
            Times, ``(n,)``.
        theta : array_like
            Parameters, ``(p,)``.

        Returns
        -------
        array
            ``(n, d)`` derivatives.

        Examples
        --------
        >>> import numpy as np
        >>> from fdatools.profiling import fitzhugh_nagumo_model
        >>> fhn = fitzhugh_nagumo_model()
        >>> fhn(np.array([[0.0, 0.0]]), np.array([0.0]), np.array([0.2, 0.2, 3.0])).round(
        ...     4
        ... ).tolist()
        [[0.0, 0.0667]]
        """
        xs, ts, th = self._arguments(x, t, theta)
        return self._f(xs, ts, th)

    # ----------------------------------------------------------- derivatives

    def _f(self, x: Array, t: Array, theta: Array) -> Array:
        return _checked(self.rhs(x, t, theta), (x.shape[0], self.n_states), "rhs")

    def _fd(
        self, fn: Callable[[Array, Array, Array], Array], x: Array, t: Array, theta: Array
    ) -> tuple[Array, Array]:
        """Central differences of ``fn`` with respect to ``x`` and to ``theta``.

        Returns arrays with a new trailing axis: ``(..., d)`` for ``x`` and
        ``(..., p)`` for ``theta``.
        """
        xp = default_namespace()
        by_x = []
        for k in range(self.n_states):
            step = _FD_STEP * xp.maximum(xp.abs(x[:, k]), 1.0)
            unit = xp.zeros(x.shape, dtype=xp.float64)
            unit[:, k] = step
            diff = fn(x + unit, t, theta) - fn(x - unit, t, theta)
            scale = xp.reshape(2.0 * step, (x.shape[0],) + (1,) * (len(diff.shape) - 1))
            by_x.append(diff / scale)
        by_theta = []
        for j in range(self.n_params):
            h = _FD_STEP * max(abs(float(theta[j])), 1.0)
            unit = xp.zeros(theta.shape, dtype=xp.float64)
            unit[j] = h
            by_theta.append((fn(x, t, theta + unit) - fn(x, t, theta - unit)) / (2.0 * h))
        return xp.stack(by_x, axis=-1), xp.stack(by_theta, axis=-1)

    def _jac(self, x: Array, t: Array, theta: Array) -> tuple[Array, Array]:
        n, d, p = x.shape[0], self.n_states, self.n_params
        if self.jac_x is not None and self.jac_theta is not None:
            return (
                _checked(self.jac_x(x, t, theta), (n, d, d), "jac_x"),
                _checked(self.jac_theta(x, t, theta), (n, d, p), "jac_theta"),
            )
        fx, ftheta = self._fd(self._f, x, t, theta)
        if self.jac_x is not None:
            fx = _checked(self.jac_x(x, t, theta), (n, d, d), "jac_x")
        if self.jac_theta is not None:
            ftheta = _checked(self.jac_theta(x, t, theta), (n, d, p), "jac_theta")
        return fx, ftheta

    def _hess(self, x: Array, t: Array, theta: Array) -> tuple[Array, Array]:
        n, d, p = x.shape[0], self.n_states, self.n_params
        fxx = (
            _checked(self.hess_xx(x, t, theta), (n, d, d, d), "hess_xx")
            if self.hess_xx is not None
            else self._fd(lambda a, b, c: self._jac(a, b, c)[0], x, t, theta)[0]
        )
        fxt = (
            _checked(self.hess_xtheta(x, t, theta), (n, d, d, p), "hess_xtheta")
            if self.hess_xtheta is not None
            else self._fd(lambda a, b, c: self._jac(a, b, c)[1], x, t, theta)[0]
        )
        if self.hess_xtheta is None:
            # _fd appends the x axis last: [q, i, j, k] -> [q, i, k, j]
            fxt = default_namespace().permute_dims(fxt, (0, 1, 3, 2))
        return fxx, fxt

    def jacobians(self, x: Any, t: Any, theta: Any) -> tuple[Array, Array]:
        r"""Return :math:`\partial f / \partial x` and :math:`\partial f / \partial \theta`.

        Parameters
        ----------
        x : array_like
            States, ``(n, d)``.
        t : array_like
            Times, ``(n,)``.
        theta : array_like
            Parameters, ``(p,)``.

        Returns
        -------
        jac_x : array
            ``(n, d, d)``.
        jac_theta : array
            ``(n, d, p)``.

        Examples
        --------
        >>> import numpy as np
        >>> from fdatools.profiling import fitzhugh_nagumo_model
        >>> fx, ft = fitzhugh_nagumo_model().jacobians(
        ...     np.array([[1.0, 0.5]]), np.array([0.0]), np.array([0.2, 0.2, 3.0])
        ... )
        >>> fx.shape, ft.shape
        ((1, 2, 2), (1, 2, 3))
        """
        return self._jac(*self._arguments(x, t, theta))

    def hessians(self, x: Any, t: Any, theta: Any) -> tuple[Array, Array]:
        r"""Return the second derivatives of ``f`` used by the profiling fit.

        Parameters
        ----------
        x : array_like
            States, ``(n, d)``.
        t : array_like
            Times, ``(n,)``.
        theta : array_like
            Parameters, ``(p,)``.

        Returns
        -------
        hess_xx : array
            ``(n, d, d, d)``, :math:`\partial^2 f_i / \partial x_k \partial x_l`.
        hess_xtheta : array
            ``(n, d, d, p)``, :math:`\partial^2 f_i / \partial x_k \partial \theta_j`.

        Examples
        --------
        >>> import numpy as np
        >>> from fdatools.profiling import fitzhugh_nagumo_model
        >>> fxx, fxt = fitzhugh_nagumo_model().hessians(
        ...     np.array([[1.0, 0.5]]), np.array([0.0]), np.array([0.2, 0.2, 3.0])
        ... )
        >>> float(fxx[0, 0, 0, 0])
        -6.0
        """
        return self._hess(*self._arguments(x, t, theta))

    # ------------------------------------------------------------ simulation

    def simulate(
        self,
        t: Any,
        x0: Any,
        theta: Any,
        *,
        breaks: Sequence[float] = (),
        rtol: float = _ODE_RTOL,
        atol: float = _ODE_ATOL,
        method: OdeMethod = "LSODA",
    ) -> Array:
        """Solve the initial-value problem with :func:`scipy.integrate.solve_ivp`.

        Parameters
        ----------
        t : array_like
            Increasing output times; the integration starts at ``t[0]``.
        x0 : array_like
            ``(d,)`` state at ``t[0]``.
        theta : array_like
            ``(p,)`` parameters.
        breaks : sequence of float, optional
            Times where the right-hand side jumps (for example step inputs).
            The integration is restarted at each, so the solver never steps
            across a discontinuity.
        rtol, atol : float, optional
            Solver tolerances.
        method : str, optional
            A :func:`~scipy.integrate.solve_ivp` method; LSODA by default.

        Returns
        -------
        array
            ``(len(t), d)`` solution.

        Raises
        ------
        ValueError
            If ``t`` is not increasing, a shape is wrong, or the solver fails.

        Examples
        --------
        >>> import numpy as np
        >>> from fdatools.profiling import ODEModel
        >>> decay = ODEModel(lambda x, t, th: -th[0] * x, n_states=1, n_params=1)
        >>> x = decay.simulate([0.0, 1.0], [1.0], [2.0])
        >>> round(float(x[1, 0]), 8) == round(float(np.exp(-2.0)), 8)
        True
        """
        xp = default_namespace()
        ts = asarray(to_numpy(t), xp=xp)
        start = asarray(to_numpy(x0), xp=xp)
        th = asarray(to_numpy(theta), xp=xp)
        if len(ts.shape) != 1 or ts.shape[0] < 1 or bool(xp.any(ts[1:] <= ts[:-1])):
            raise ValueError("t must be a non-empty, strictly increasing 1-D array")
        if tuple(start.shape) != (self.n_states,):
            raise ValueError(f"x0 must have shape ({self.n_states},), got {tuple(start.shape)}")
        lo, hi = float(ts[0]), float(ts[-1])
        cuts = [lo, *sorted(float(b) for b in breaks if lo < float(b) < hi), hi]

        def fun(s: float, state: Array) -> Array:
            now = xp.asarray([s], dtype=xp.float64)
            return self._f(xp.reshape(asarray(state, xp=xp), (1, -1)), now, th)[0, :]

        out = xp.zeros((ts.shape[0], self.n_states), dtype=xp.float64)
        out[0, :] = start
        state = start
        for left, right in pairwise(cuts):
            inside = (ts > left) & (ts <= right)
            if right == hi:
                inside = inside | (ts == right)
            inside[0] = False
            wanted = ts[inside]
            if right <= left:
                continue
            sol = solve_ivp(
                fun,
                (left, right),
                to_numpy(state),
                method=method,
                t_eval=to_numpy(wanted) if wanted.shape[0] else None,
                rtol=rtol,
                atol=atol,
                dense_output=False,
            )
            if not sol.success:
                raise ValueError(f"the ODE solver failed on [{left}, {right}]: {sol.message}")
            values = asarray(sol.y, xp=xp)
            if wanted.shape[0]:
                out[inside, :] = xp.matrix_transpose(values[:, -wanted.shape[0] :])
            state = values[:, -1]
        return out

    # ----------------------------------------------------------------- torch

    @classmethod
    def from_torch(
        cls,
        rhs: Callable[[Any, Any, Any], Any],
        n_states: int,
        n_params: int,
        *,
        state_names: Sequence[str] = (),
        param_names: Sequence[str] = (),
    ) -> ODEModel:
        """Build a model from a PyTorch right-hand side, with autodiff derivatives.

        ``rhs(x, t, theta)`` must use PyTorch operations only and treat the
        ``n`` points independently (it is differentiated one point at a time
        with :func:`torch.func.vmap`).  Every Jacobian and second derivative is
        then exact.  Requires the ``fdatools[torch]`` extra; PyTorch is imported
        only when this method is called.

        Parameters
        ----------
        rhs : callable
            PyTorch function of ``(x, t, theta)`` returning ``(n, d)`` tensors.
        n_states, n_params : int
            Dimensions of ``x`` and ``theta``.
        state_names, param_names : sequence of str, optional
            Labels.

        Returns
        -------
        ODEModel
            A model whose callables accept and return NumPy arrays.

        Examples
        --------
        >>> import numpy as np
        >>> import torch
        >>> from fdatools.profiling import ODEModel
        >>> model = ODEModel.from_torch(lambda x, t, th: -th[0] * x**2, 1, 1)
        >>> float(
        ...     model.hessians(np.array([[3.0]]), np.array([0.0]), np.array([2.0]))[0][0, 0, 0, 0]
        ... )
        -4.0
        """
        import torch
        from torch.func import jacfwd, jacrev, vmap

        def tensor(a: Array) -> Any:
            return torch.as_tensor(to_numpy(a), dtype=torch.float64)

        def point(xq: Any, tq: Any, th: Any) -> Any:
            return rhs(xq[None, :], tq[None], th)[0]

        jx = vmap(jacrev(point, argnums=0), in_dims=(0, 0, None))
        jt = vmap(jacrev(point, argnums=2), in_dims=(0, 0, None))
        hxx = vmap(jacfwd(jacrev(point, argnums=0), argnums=0), in_dims=(0, 0, None))
        hxt = vmap(jacfwd(jacrev(point, argnums=0), argnums=2), in_dims=(0, 0, None))

        def lift(fn: Callable[..., Any]) -> Function:
            def wrapped(x: Array, t: Array, theta: Array) -> Array:
                return to_numpy(fn(tensor(x), tensor(t), tensor(theta)))

            return wrapped

        return cls(
            rhs=lift(rhs),
            n_states=n_states,
            n_params=n_params,
            jac_x=lift(jx),
            jac_theta=lift(jt),
            hess_xx=lift(hxx),
            hess_xtheta=lift(hxt),
            state_names=tuple(state_names),
            param_names=tuple(param_names),
        )


# --------------------------------------------------------------------------- #
# built-in models
# --------------------------------------------------------------------------- #


def fitzhugh_nagumo_model() -> ODEModel:
    r"""Build the FitzHugh-Nagumo neuron model, with analytic derivatives.

    .. math::

        \dot V = c\,(V - V^3/3 + R), \qquad \dot R = -(V - a + b R) / c,

    with parameters :math:`\theta = (a, b, c)`; the classic values are
    ``(0.2, 0.2, 3)`` (Ramsay et al. 2007, section 2.1).

    Returns
    -------
    ODEModel
        States ``("V", "R")``, parameters ``("a", "b", "c")``.

    Examples
    --------
    >>> import numpy as np
    >>> from fdatools.profiling import fitzhugh_nagumo_model
    >>> fhn = fitzhugh_nagumo_model()
    >>> x = fhn.simulate(np.linspace(0.0, 20.0, 5), [-1.0, 1.0], [0.2, 0.2, 3.0])
    >>> x.shape
    (5, 2)
    """
    xp = default_namespace()

    def rhs(x: Array, t: Array, th: Array) -> Array:
        v, r = x[:, 0], x[:, 1]
        a, b, c = th[0], th[1], th[2]
        return xp.stack([c * (v - v**3 / 3.0 + r), -(v - a + b * r) / c], axis=1)

    def jac_x(x: Array, t: Array, th: Array) -> Array:
        v = x[:, 0]
        b, c = th[1], th[2]
        out = xp.zeros((x.shape[0], 2, 2), dtype=xp.float64)
        out[:, 0, 0] = c * (1.0 - v**2)
        out[:, 0, 1] = c
        out[:, 1, 0] = -1.0 / c
        out[:, 1, 1] = -b / c
        return out

    def jac_theta(x: Array, t: Array, th: Array) -> Array:
        v, r = x[:, 0], x[:, 1]
        a, b, c = th[0], th[1], th[2]
        out = xp.zeros((x.shape[0], 2, 3), dtype=xp.float64)
        out[:, 0, 2] = v - v**3 / 3.0 + r
        out[:, 1, 0] = 1.0 / c
        out[:, 1, 1] = -r / c
        out[:, 1, 2] = (v - a + b * r) / c**2
        return out

    def hess_xx(x: Array, t: Array, th: Array) -> Array:
        out = xp.zeros((x.shape[0], 2, 2, 2), dtype=xp.float64)
        out[:, 0, 0, 0] = -2.0 * th[2] * x[:, 0]
        return out

    def hess_xtheta(x: Array, t: Array, th: Array) -> Array:
        v = x[:, 0]
        b, c = th[1], th[2]
        out = xp.zeros((x.shape[0], 2, 2, 3), dtype=xp.float64)
        out[:, 0, 0, 2] = 1.0 - v**2
        out[:, 0, 1, 2] = 1.0
        out[:, 1, 0, 2] = 1.0 / c**2
        out[:, 1, 1, 1] = -1.0 / c
        out[:, 1, 1, 2] = b / c**2
        return out

    return ODEModel(
        rhs=rhs,
        n_states=2,
        n_params=3,
        jac_x=jac_x,
        jac_theta=jac_theta,
        hess_xx=hess_xx,
        hess_xtheta=hess_xtheta,
        state_names=("V", "R"),
        param_names=("a", "b", "c"),
    )


def cstr_inputs(t: Any, condition: str = "all.cool.step") -> Array:
    """Input series of the CSTR experiments (R's ``CSTR2in``).

    The reactor is driven by five inputs: the flow ``F``, the inlet
    concentration ``CA0``, the inlet temperature ``T0``, the coolant inlet
    temperature ``Tcin`` and the coolant flow ``Fc``.  The ``all.*.step``
    scenarios hold them at nominal values (``F = 1``, ``CA0 = 2``,
    ``T0 = 323``, ``Fc = 15``, ``Tcin = 335`` cool or ``365`` hot) and step one
    input at a time on the 4-minute intervals ``[4k, 4k + 4)`` of ``[0, 64)``:
    ``F`` to 1.5 and 0.5, ``CA0`` to 2.2 and 1.8, ``T0`` to 343 and 303,
    ``Tcin`` up and down by 5, ``Fc`` to 20 and 10.  The ``Tc.*.step``
    scenarios lower ``Tcin`` by 15 on ``[2, 12)`` only, with ``F = 1`` (hot)
    or ``0.05`` (cool).

    Parameters
    ----------
    t : array_like
        Times (minutes).
    condition : str, optional
        One of :data:`CSTR_CONDITIONS`.

    Returns
    -------
    array
        ``(len(t), 5)`` inputs, columns :data:`CSTR_INPUTS`.

    Raises
    ------
    ValueError
        If ``condition`` is unknown.

    Examples
    --------
    >>> from fdatools.profiling import cstr_inputs
    >>> cstr_inputs([0.0, 4.0, 8.0], "all.cool.step")[:, 0].tolist()
    [1.0, 1.5, 0.5]
    """
    xp = default_namespace()
    if condition not in CSTR_CONDITIONS:
        raise ValueError(f"unknown CSTR condition {condition!r}; expected one of {CSTR_CONDITIONS}")
    ts = xp.reshape(asarray(to_numpy(t), xp=xp), (-1,))
    n = ts.shape[0]
    hot = condition.startswith("all.hot") or condition == "Tc.hot.step"
    nominal = {
        "F": 0.05 if condition == "Tc.cool.step" else 1.0,
        "CA0": 2.0,
        "T0": 323.0,
        "Tcin": 365.0 if hot else 335.0,
        "Fc": 15.0,
    }
    columns = {name: xp.full((n,), value, dtype=xp.float64) for name, value in nominal.items()}
    if condition.startswith("all."):
        steps = {
            1: ("F", 1.5),
            2: ("F", 0.5),
            4: ("CA0", 2.2),
            5: ("CA0", 1.8),
            7: ("T0", 343.0),
            8: ("T0", 303.0),
            10: ("Tcin", nominal["Tcin"] + 5.0),
            11: ("Tcin", nominal["Tcin"] - 5.0),
            13: ("Fc", 20.0),
            14: ("Fc", 10.0),
        }
        interval = xp.floor(ts / 4.0)
        for k, (name, value) in steps.items():
            columns[name] = xp.where(interval == float(k), value, columns[name])
    else:
        low = (ts >= 2.0) & (ts < 12.0)
        columns["Tcin"] = xp.where(low, nominal["Tcin"] - 15.0, columns["Tcin"])
    return xp.stack([columns[name] for name in CSTR_INPUTS], axis=1)


def cstr_model(
    condition: str | Callable[[Array], Any] = "all.cool.step",
    *,
    estimate: Sequence[str] = ("kref", "EoverR", "a", "b"),
    **values: float,
) -> ODEModel:
    r"""Build the two-state continuously stirred tank reactor (Ramsay et al. 2007).

    States are the concentration ``C`` and temperature ``T``:

    .. math::

        \dot C &= -\bigl(k(T) + F/V\bigr) C + (F/V)\, C_{A0}, \\
        \dot T &= -\bigl(\alpha + F/V\bigr) T - \frac{\Delta H}{V \rho C_p} k(T)\, C
                  + (F/V)\, T_0 + \alpha\, T_{cin},

    with :math:`k(T) = k_{ref} \exp\{-10^4 (E/R) (1/T - 1/T_{ref})\}` and
    :math:`\alpha = a F_c^{b+1} / \{V \rho C_p (F_c + a F_c^b / (2 \rho_c C_{pc}))\}`
    -- the same right-hand side as R's ``CSTR2``.  The parameters are
    ``kref``, ``EoverR`` (in units of ``1e4`` K), ``a`` and ``b``; those named in
    ``estimate`` form :math:`\theta` (in that order) and the others are held at
    their values.  All derivatives are analytic.

    Parameters
    ----------
    condition : str or callable, optional
        A scenario of :func:`cstr_inputs`, or a function mapping ``(n,)``
        times to the ``(n, 5)`` inputs (columns :data:`CSTR_INPUTS`).
    estimate : sequence of str, optional
        The parameters to estimate.
    **values : float
        Overrides of the parameter values (:data:`CSTR_PARAMETERS`, used for
        the ones not estimated) and of the constants (:data:`CSTR_CONSTANTS`).

    Returns
    -------
    ODEModel
        States ``("C", "T")``, parameters ``estimate``.

    Raises
    ------
    ValueError
        If a name in ``estimate`` or ``values`` is unknown, or ``estimate``
        is empty or repeats a name.

    Examples
    --------
    >>> import numpy as np
    >>> from fdatools.profiling import cstr_model
    >>> model = cstr_model("all.cool.step", estimate=("kref", "EoverR"))
    >>> dx = model(np.array([[1.5965, 341.3754]]), np.array([1.0]), np.array([0.461, 0.83301]))
    >>> dx.round(6).tolist()
    [[0.000106, 0.010411]]
    """
    xp = default_namespace()
    known = {**CSTR_PARAMETERS, **CSTR_CONSTANTS}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"unknown CSTR value(s) {unknown}; expected names from {sorted(known)}")
    chosen = tuple(str(e) for e in estimate)
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(CSTR_PARAMETERS):
        raise ValueError(
            f"estimate must be distinct names from {tuple(CSTR_PARAMETERS)}, got {chosen}"
        )
    fixed = {**known, **{k: float(v) for k, v in values.items()}}
    k1 = fixed["V"] * fixed["rho"] * fixed["Cp"]
    k2 = 1.0 / (2.0 * fixed["rhoc"] * fixed["Cpc"])
    heat = -fixed["delH"] / k1
    volume = fixed["V"]
    tref = fixed["Tref"]
    columns = [list(CSTR_PARAMETERS).index(name) for name in chosen]
    if callable(condition):
        user = condition

        def inputs(t: Array) -> Array:
            return _checked(user(t), (t.shape[0], 5), "CSTR inputs")

    else:
        scenario = str(condition)
        cstr_inputs(xp.zeros((1,)), scenario)

        def inputs(t: Array) -> Array:
            return cstr_inputs(t, scenario)

    def terms(x: Array, t: Array, th: Array) -> dict[str, Array]:
        full = [fixed[name] for name in CSTR_PARAMETERS]
        for pos, col in enumerate(columns):
            full[col] = th[pos]
        kref, eoverr, a, b = full
        u = inputs(t)
        flow, ca0, t0, tcin, fc = (u[:, j] for j in range(5))
        conc, temp = x[:, 0], x[:, 1]
        arrh = 1e4 * (1.0 / temp - 1.0 / tref)
        k = kref * xp.exp(-eoverr * arrh)
        dk = k * 1e4 * eoverr / temp**2
        fcb = fc**b
        denom = k1 * (fc + k2 * a * fcb)
        alpha = a * fc * fcb / denom
        return {
            "C": conc,
            "T": temp,
            "q": flow / volume,
            "ca0": ca0,
            "t0": t0,
            "tcin": tcin,
            "k": k,
            "dk": dk,
            "ddk": dk * (1e4 * eoverr / temp**2 - 2.0 / temp),
            "k_kref": k / kref,
            "k_e": -k * arrh,
            "dk_kref": dk / kref,
            "dk_e": -dk * arrh + k * 1e4 / temp**2,
            "alpha": alpha,
            "alpha_a": k1 * fc * fc * fcb / denom**2,
            "alpha_b": alpha * xp.log(fc) * k1 * fc / denom,
        }

    def rhs(x: Array, t: Array, th: Array) -> Array:
        v = terms(x, t, th)
        dc = -(v["k"] + v["q"]) * v["C"] + v["q"] * v["ca0"]
        dt = (
            -(v["alpha"] + v["q"]) * v["T"]
            + heat * v["k"] * v["C"]
            + v["q"] * v["t0"]
            + v["alpha"] * v["tcin"]
        )
        return xp.stack([dc, dt], axis=1)

    def jac_x(x: Array, t: Array, th: Array) -> Array:
        v = terms(x, t, th)
        out = xp.zeros((x.shape[0], 2, 2), dtype=xp.float64)
        out[:, 0, 0] = -(v["k"] + v["q"])
        out[:, 0, 1] = -v["dk"] * v["C"]
        out[:, 1, 0] = heat * v["k"]
        out[:, 1, 1] = -(v["alpha"] + v["q"]) + heat * v["dk"] * v["C"]
        return out

    def full_theta(v: dict[str, Array]) -> tuple[list[Array], list[Array]]:
        zero = xp.zeros(v["C"].shape, dtype=xp.float64)
        dfc = [-v["k_kref"] * v["C"], -v["k_e"] * v["C"], zero, zero]
        gap = v["tcin"] - v["T"]
        dft = [
            heat * v["k_kref"] * v["C"],
            heat * v["k_e"] * v["C"],
            v["alpha_a"] * gap,
            v["alpha_b"] * gap,
        ]
        return dfc, dft

    def jac_theta(x: Array, t: Array, th: Array) -> Array:
        dfc, dft = full_theta(terms(x, t, th))
        rows = [
            xp.stack([dfc[c] for c in columns], axis=1),
            xp.stack([dft[c] for c in columns], axis=1),
        ]
        return xp.stack(rows, axis=1)

    def hess_xx(x: Array, t: Array, th: Array) -> Array:
        v = terms(x, t, th)
        out = xp.zeros((x.shape[0], 2, 2, 2), dtype=xp.float64)
        out[:, 0, 0, 1] = out[:, 0, 1, 0] = -v["dk"]
        out[:, 0, 1, 1] = -v["ddk"] * v["C"]
        out[:, 1, 0, 1] = out[:, 1, 1, 0] = heat * v["dk"]
        out[:, 1, 1, 1] = heat * v["ddk"] * v["C"]
        return out

    def hess_xtheta(x: Array, t: Array, th: Array) -> Array:
        v = terms(x, t, th)
        zero = xp.zeros(v["C"].shape, dtype=xp.float64)
        # [state i][x_k] -> derivative of df_i/dx_k with respect to each full parameter
        full = [
            [
                [-v["k_kref"], -v["k_e"], zero, zero],
                [-v["dk_kref"] * v["C"], -v["dk_e"] * v["C"], zero, zero],
            ],
            [
                [heat * v["k_kref"], heat * v["k_e"], zero, zero],
                [
                    heat * v["dk_kref"] * v["C"],
                    heat * v["dk_e"] * v["C"],
                    -v["alpha_a"],
                    -v["alpha_b"],
                ],
            ],
        ]
        return xp.stack(
            [
                xp.stack(
                    [xp.stack([full[i][k][c] for c in columns], axis=1) for k in range(2)], axis=1
                )
                for i in range(2)
            ],
            axis=1,
        )

    return ODEModel(
        rhs=rhs,
        n_states=2,
        n_params=len(chosen),
        jac_x=jac_x,
        jac_theta=jac_theta,
        hess_xx=hess_xx,
        hess_xtheta=hess_xtheta,
        state_names=("C", "T"),
        param_names=chosen,
    )


# --------------------------------------------------------------------------- #
# quadrature
# --------------------------------------------------------------------------- #


def simpson_rule(breaks: Sequence[float], n_quad: int = 5) -> tuple[Array, Array]:
    """Composite Simpson rule on the panels between consecutive ``breaks``.

    Each panel gets ``n_quad`` equally spaced nodes including both ends, with
    weights ``h/3 * (1, 4, 2, 4, ..., 2, 4, 1)``; a node on an interior break
    appears twice, once per panel -- the rule of R's ``quadset``.

    Parameters
    ----------
    breaks : sequence of float
        Strictly increasing panel boundaries.
    n_quad : int, optional
        Odd number of nodes per panel, at least 3.

    Returns
    -------
    nodes, weights : array
        ``((len(breaks) - 1) * n_quad,)`` each.

    Raises
    ------
    ValueError
        If ``n_quad`` is even or below 3, or ``breaks`` is not increasing.

    Examples
    --------
    >>> from fdatools.profiling import simpson_rule
    >>> nodes, weights = simpson_rule([0.0, 1.0], 5)
    >>> nodes.tolist()
    [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> [round(float(w) * 12, 12) for w in weights]
    [1.0, 4.0, 2.0, 4.0, 1.0]
    """
    xp = default_namespace()
    n = int(n_quad)
    if n < 3 or n % 2 == 0:
        raise ValueError(f"n_quad must be an odd integer >= 3, got {n_quad}")
    edges = [float(b) for b in breaks]
    if len(edges) < 2 or any(b <= a for a, b in pairwise(edges)):
        raise ValueError("breaks must be strictly increasing with at least two values")
    pattern = xp.asarray([1.0] + [4.0 if i % 2 else 2.0 for i in range(1, n - 1)] + [1.0])
    nodes, weights = [], []
    for lo, hi in pairwise(edges):
        nodes.append(xp.linspace(lo, hi, n, dtype=xp.float64))
        weights.append(pattern * ((hi - lo) / (n - 1) / 3.0))
    return xp.concat(nodes), xp.concat(weights)


# --------------------------------------------------------------------------- #
# the profiling problem
# --------------------------------------------------------------------------- #


class Residuals(NamedTuple):
    """Residuals of the inner criterion and their Jacobians at ``(c, theta)``.

    ``J(c | theta)`` is ``data @ data + equation @ equation``.  Rows of the
    Jacobians are the data residuals (observed states in order) followed by the
    equation residuals (state by state, quadrature node by node); columns of
    ``jac_coefs`` are the stacked coefficients of all states.

    Attributes
    ----------
    data : array
        ``sqrt(w_i) (y_ij - x_i(t_ij))``.
    equation : array
        ``sqrt(w_i lambda_i v_q) (Dx_i(s_q) - f_i(x(s_q), s_q, theta))``.
    jac_coefs : array
        Derivative of ``concat(data, equation)`` with respect to the coefficients.
    jac_theta : array
        Derivative of ``concat(data, equation)`` with respect to ``theta``.
    """

    data: Array
    equation: Array
    jac_coefs: Array
    jac_theta: Array


@dataclass(frozen=True, eq=False)
class InnerFit:
    """The state fit for one fixed parameter vector (the inner problem).

    Attributes
    ----------
    coefs : tuple of array
        Fitted basis coefficients, one ``(n_basis_i,)`` array per state.
    criterion : float
        The minimised ``J(c | theta)``.
    sse : float
        Its weighted data part ``sum_i w_i sum_j (y_ij - x_i(t_ij))**2``.
    df : float
        Effective degrees of freedom, the trace of the (linearised) map from
        the data to the fitted values.
    gcv : float
        ``N * sse / (N - df)**2`` for ``N`` observations.
    dcoefs_dtheta : array
        ``(K, p)`` implicit-function derivative of the stacked coefficients.
    n_iter : int
        Gauss-Newton iterations used.
    converged : bool
        Whether the step tolerance was met.
    """

    coefs: tuple[Array, ...]
    criterion: float
    sse: float
    df: float
    gcv: float
    dcoefs_dtheta: Array
    n_iter: int
    converged: bool


@dataclass(frozen=True, eq=False)
class ProfileResult:
    """The result of a generalized-profiling fit.

    Attributes
    ----------
    theta : array
        ``(p,)`` estimated parameters.
    cov : array
        ``(p, p)`` approximate covariance ``sigma2 * inv(A.T @ A)``.
    sigma2 : float
        Residual variance ``sse / (N - p)``.
    states : tuple of FData
        The fitted state functions, one single-curve :class:`~fdatools.FData` per
        state.
    inner : InnerFit
        The inner fit at ``theta``.
    theta_path : array
        ``(n_iter + 1, p)`` parameter values visited, from the start.
    sse_path : array
        ``(n_iter + 1,)`` profiled data criterion at each of them.
    n_iter : int
        Outer Gauss-Newton iterations.
    converged : bool
        Whether the outer tolerance was met.
    param_names, state_names : tuple of str
        Labels from the model.
    """

    theta: Array
    cov: Array
    sigma2: float
    states: tuple[FData, ...]
    inner: InnerFit
    theta_path: Array
    sse_path: Array
    n_iter: int
    converged: bool
    param_names: tuple[str, ...] = field(default=())
    state_names: tuple[str, ...] = field(default=())

    @property
    def stderr(self) -> Array:
        """Standard errors, the square roots of the diagonal of :attr:`cov`.

        Returns
        -------
        array
            ``(p,)`` standard errors.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.profiling import ODEModel, profile_ode
        >>> m = ODEModel(lambda x, t, th: -th[0] * x, 1, 1)
        >>> t = np.linspace(0.0, 1.0, 21)
        >>> y = np.exp(-t) + 0.01 * np.cos(40 * t)
        >>> fit = profile_ode(m, t, y[:, None], fdt.BSpline((0.0, 1.0), 12), 1e3, [0.5])
        >>> fit.stderr.shape
        (1,)
        """
        xp = default_namespace()
        return xp.sqrt(xp.linalg.diagonal(self.cov))

    def __call__(self, t: Any, deriv: int = 0) -> Array:
        """Evaluate the fitted states at ``t``.

        Parameters
        ----------
        t : array_like
            Evaluation times.
        deriv : int, optional
            Derivative order.

        Returns
        -------
        array
            ``(len(t), d)`` state values.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.profiling import ODEModel, profile_ode
        >>> m = ODEModel(lambda x, t, th: -th[0] * x, 1, 1)
        >>> t = np.linspace(0.0, 1.0, 21)
        >>> fit = profile_ode(m, t, np.exp(-t)[:, None], fdt.BSpline((0.0, 1.0), 12), 1e3, [0.5])
        >>> fit(np.array([0.0, 1.0])).shape
        (2, 1)
        """
        xp = default_namespace()
        ts = asarray(to_numpy(t), xp=xp)
        cols = [xp.reshape(asarray(to_numpy(s(ts, deriv)), xp=xp), (-1,)) for s in self.states]
        return xp.stack(cols, axis=1)


def _sequence(value: Any, n: int, what: str) -> list[Any]:
    """Broadcast a scalar-like value, or check a per-state sequence of length ``n``."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{what} must have one entry per state ({n}), got {len(value)}")
        return list(value)
    return [value] * n


def _observations(t: Any, y: Any, d: int, xp: ModuleType) -> tuple[list[Array], list[Array]]:
    """Split the observations into per-state ``(times, values)``, dropping NaNs."""
    times: list[Array] = []
    values: list[Array] = []
    if isinstance(y, (list, tuple)):
        if len(y) != d:
            raise ValueError(f"y must have one entry per state ({d}), got {len(y)}")
        ts = _sequence(t, d, "t") if isinstance(t, (list, tuple)) else [t] * d
        for ti, yi in zip(ts, y, strict=True):
            if yi is None:
                times.append(xp.zeros((0,), dtype=xp.float64))
                values.append(xp.zeros((0,), dtype=xp.float64))
                continue
            tt = xp.reshape(asarray(to_numpy(ti), xp=xp), (-1,))
            yy = xp.reshape(asarray(to_numpy(yi), xp=xp), (-1,))
            if tt.shape != yy.shape:
                raise ValueError(f"times {tuple(tt.shape)} and values {tuple(yy.shape)} differ")
            keep = ~xp.isnan(yy)
            times.append(tt[keep])
            values.append(yy[keep])
        return times, values
    tt = xp.reshape(asarray(to_numpy(t), xp=xp), (-1,))
    yy = asarray(to_numpy(y), xp=xp)
    if len(yy.shape) == 1 and d == 1:
        yy = xp.reshape(yy, (-1, 1))
    if len(yy.shape) != 2 or yy.shape != (tt.shape[0], d):
        raise ValueError(f"y must have shape ({tt.shape[0]}, {d}), got {tuple(yy.shape)}")
    for i in range(d):
        keep = ~xp.isnan(yy[:, i])
        times.append(tt[keep])
        values.append(yy[keep, i])
    return times, values


def _default_breaks(bases: Sequence[Basis]) -> list[float]:
    """Panel boundaries: the union of all B-spline breaks, else an even grid."""
    domain = bases[0].domain
    edges = {float(domain[0]), float(domain[1])}
    splines = [b for b in bases if isinstance(b, BSpline)]
    if splines:
        for b in splines:
            edges.update(float(v) for v in b.breaks)
        return sorted(edges)
    n = _PANELS_PER_BASIS * max(b.n_basis for b in bases)
    step = (domain[1] - domain[0]) / n
    return [domain[0] + i * step for i in range(n)] + [domain[1]]


def _scaled_solve(matrix: Array, rhs: Array, xp: ModuleType, *, spd: bool) -> Array:
    """Solve ``matrix @ x = rhs`` after scaling ``matrix`` to unit diagonal."""
    diag = xp.abs(xp.linalg.diagonal(matrix))
    scale = 1.0 / xp.sqrt(xp.where(diag > 0.0, diag, 1.0))
    scaled = scale[:, None] * matrix * scale[None, :]
    right = scale[:, None] * rhs if len(rhs.shape) == 2 else scale * rhs
    sol = asarray(_linalg.solve_spd(scaled, right)) if spd else xp.linalg.solve(scaled, right)
    return scale[:, None] * sol if len(rhs.shape) == 2 else scale * sol


class ProfiledODE:
    r"""A generalized-profiling problem: model, data, bases and smoothing.

    Parameters
    ----------
    model : ODEModel
        The right-hand side.
    t : array_like or sequence
        Observation times: one ``(n,)`` array shared by all states, or one
        array per state (``None`` for an unobserved state).
    y : array_like or sequence
        Observations: an ``(n, d)`` array (``NaN`` marks a missing value; an
        all-``NaN`` column an unobserved state), or one ``(n_i,)`` array per
        state (``None`` for an unobserved state).
    bases : Basis or sequence of Basis
        One basis for every state, or one per state, all on the same domain.
    lam : float or sequence of float, optional
        Weight :math:`\lambda_i` of the equation term of each state.
    state_weights : float or sequence of float, optional
        Weights :math:`w_i` multiplying both terms of each state (the
        reciprocal of R's ``Cwt``/``Twt``); ``1`` by default.  Use them to put
        states measured on different scales on a common footing.
    quadrature : tuple of array_like, optional
        ``(nodes, weights)`` for the integral.  By default the composite
        Simpson rule (:func:`simpson_rule`) with ``n_quad`` nodes on each panel
        between the B-spline break points of the bases (R's ``quadset``).
    n_quad : int, optional
        Nodes per panel of the default rule.

    Attributes
    ----------
    model : ODEModel
    bases : tuple of Basis
    lam, state_weights : tuple of float
    nodes, weights : array
        The quadrature rule.
    n_obs : int
        Total number of observations ``N``.

    Raises
    ------
    ValueError
        If shapes, domains or weights are inconsistent, or no state is
        observed.

    Examples
    --------
    >>> import numpy as np
    >>> import fdatools as fdt
    >>> from fdatools.profiling import ProfiledODE, fitzhugh_nagumo_model
    >>> fhn = fitzhugh_nagumo_model()
    >>> t = np.linspace(0.0, 20.0, 201)
    >>> x = fhn.simulate(t, [-1.0, 1.0], [0.2, 0.2, 3.0])
    >>> basis = fdt.BSpline(domain=(0.0, 20.0), breaks=np.linspace(0.0, 20.0, 201))
    >>> problem = ProfiledODE(fhn, t, x, basis, lam=1e3)
    >>> fit = problem.fit([0.3, 0.3, 2.5])
    >>> np.round(fit.theta, 3).tolist()
    [0.2, 0.2, 3.0]
    """

    model: ODEModel
    bases: tuple[Basis, ...]
    lam: tuple[float, ...]
    state_weights: tuple[float, ...]
    nodes: Array
    weights: Array
    n_obs: int

    def __init__(
        self,
        model: ODEModel,
        t: Any,
        y: Any,
        bases: Basis | Sequence[Basis],
        lam: float | Sequence[float] = 1.0,
        *,
        state_weights: float | Sequence[float] = 1.0,
        quadrature: tuple[Any, Any] | None = None,
        n_quad: int = 5,
    ) -> None:
        xp = default_namespace()
        d = model.n_states
        basis_list = _sequence(bases, d, "bases")
        if not all(isinstance(b, Basis) for b in basis_list):
            raise ValueError("bases must be Basis objects")
        domain = basis_list[0].domain
        if any(tuple(b.domain) != tuple(domain) for b in basis_list):
            raise ValueError("all state bases must share one domain")
        lams = [float(v) for v in _sequence(lam, d, "lam")]
        wts = [float(v) for v in _sequence(state_weights, d, "state_weights")]
        if any(v < 0.0 or not math.isfinite(v) for v in lams):
            raise ValueError(f"lam must be finite and non-negative, got {lams}")
        if any(v <= 0.0 or not math.isfinite(v) for v in wts):
            raise ValueError(f"state_weights must be finite and positive, got {wts}")
        times, values = _observations(t, y, d, xp)
        span = domain[1] - domain[0]
        for tt in times:
            if tt.shape[0] and (
                float(xp.min(tt)) < domain[0] - _DOMAIN_TOL * span
                or float(xp.max(tt)) > domain[1] + _DOMAIN_TOL * span
            ):
                raise ValueError(f"observation times must lie in the basis domain {domain}")
        n_obs = sum(int(v.shape[0]) for v in values)
        if n_obs == 0:
            raise ValueError("no state has any observation")
        if quadrature is None:
            nodes, qweights = simpson_rule(_default_breaks(basis_list), n_quad)
        else:
            nodes = xp.reshape(asarray(to_numpy(quadrature[0]), xp=xp), (-1,))
            qweights = xp.reshape(asarray(to_numpy(quadrature[1]), xp=xp), (-1,))
            if nodes.shape != qweights.shape or nodes.shape[0] == 0:
                raise ValueError("quadrature nodes and weights must be equal-length 1-D arrays")
            if bool(xp.any(qweights < 0.0)):
                raise ValueError("quadrature weights must be non-negative")
        self.model = model
        self.bases = tuple(basis_list)
        self.lam = tuple(lams)
        self.state_weights = tuple(wts)
        self.nodes = nodes
        self.weights = qweights
        self.n_obs = n_obs
        self._times = times
        self._values = values
        self._design = [
            asarray(to_numpy(b(tt)), xp=xp) for b, tt in zip(basis_list, times, strict=True)
        ]
        self._phi = [asarray(to_numpy(b(nodes)), xp=xp) for b in basis_list]
        self._dphi = [asarray(to_numpy(b(nodes, 1)), xp=xp) for b in basis_list]
        self._sizes = [b.n_basis for b in basis_list]
        self._offsets = [sum(self._sizes[:i]) for i in range(d + 1)]
        self._eq_scale = [xp.sqrt(w * lm * qweights) for w, lm in zip(wts, lams, strict=True)]

    # --------------------------------------------------------------- helpers

    @property
    def n_coefs(self) -> int:
        """Total number of basis coefficients ``K`` over all states.

        Returns
        -------
        int
            ``sum(b.n_basis for b in bases)``.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.profiling import ProfiledODE, fitzhugh_nagumo_model
        >>> t = np.linspace(0.0, 1.0, 5)
        >>> p = ProfiledODE(fitzhugh_nagumo_model(), t, np.ones((5, 2)), fdt.BSpline(n_basis=6))
        >>> p.n_coefs
        12
        """
        return int(self._offsets[-1])

    def _flat(self, coefs: Any) -> Array:
        """Stack per-state coefficient arrays (or check a flat vector)."""
        xp = default_namespace()
        if isinstance(coefs, (list, tuple)):
            parts = [xp.reshape(asarray(to_numpy(c), xp=xp), (-1,)) for c in coefs]
            if [int(p.shape[0]) for p in parts] != self._sizes:
                raise ValueError(f"coefficient sizes must be {self._sizes}")
            return xp.concat(parts)
        flat = xp.reshape(asarray(to_numpy(coefs), xp=xp), (-1,))
        if flat.shape[0] != self.n_coefs:
            raise ValueError(f"expected {self.n_coefs} coefficients, got {flat.shape[0]}")
        return flat

    def _split(self, flat: Array) -> tuple[Array, ...]:
        return tuple(
            flat[self._offsets[i] : self._offsets[i + 1]] for i in range(self.model.n_states)
        )

    def _theta(self, theta: Any) -> Array:
        xp = default_namespace()
        th = xp.reshape(asarray(to_numpy(theta), xp=xp), (-1,))
        if th.shape[0] != self.model.n_params:
            raise ValueError(f"theta must have {self.model.n_params} values, got {th.shape[0]}")
        return th

    def _states(self, flat: Array) -> tuple[Array, Array]:
        """States and their derivatives at the quadrature nodes, ``(Q, d)`` each."""
        xp = default_namespace()
        parts = self._split(flat)
        x = xp.stack([phi @ c for phi, c in zip(self._phi, parts, strict=True)], axis=1)
        dx = xp.stack([dphi @ c for dphi, c in zip(self._dphi, parts, strict=True)], axis=1)
        return x, dx

    # ------------------------------------------------------------- residuals

    def _residuals(self, flat: Array, theta: Array, *, jacobian: bool) -> Residuals:
        xp = default_namespace()
        d, p = self.model.n_states, self.model.n_params
        parts = self._split(flat)
        data = xp.concat(
            [
                math.sqrt(w) * (yy - bm @ c)
                for w, yy, bm, c in zip(
                    self.state_weights, self._values, self._design, parts, strict=True
                )
            ]
        )
        x, dx = self._states(flat)
        f = self.model._f(x, self.nodes, theta)
        equation = xp.concat([self._eq_scale[i] * (dx[:, i] - f[:, i]) for i in range(d)])
        if not jacobian:
            empty = xp.zeros((0, 0), dtype=xp.float64)
            return Residuals(data, equation, empty, empty)
        fx, ftheta = self.model._jac(x, self.nodes, theta)
        n_data, q, k_all = data.shape[0], self.nodes.shape[0], self.n_coefs
        jc = xp.zeros((n_data + d * q, k_all), dtype=xp.float64)
        jt = xp.zeros((n_data + d * q, p), dtype=xp.float64)
        row = 0
        for i in range(d):
            m = self._design[i].shape[0]
            jc[row : row + m, self._offsets[i] : self._offsets[i + 1]] = (
                -math.sqrt(self.state_weights[i]) * self._design[i]
            )
            row += m
        for i in range(d):
            rows = slice(n_data + i * q, n_data + (i + 1) * q)
            s = self._eq_scale[i][:, None]
            for k in range(d):
                block = -fx[:, i, k][:, None] * self._phi[k]
                if k == i:
                    block = block + self._dphi[i]
                jc[rows, self._offsets[k] : self._offsets[k + 1]] = s * block
            jt[rows, :] = -s * ftheta[:, i, :]
        return Residuals(data, equation, jc, jt)

    def residuals(self, coefs: Any, theta: Any) -> Residuals:
        r"""Residuals and Jacobians of the inner criterion (R's ``CSTRfitLS``).

        Parameters
        ----------
        coefs : array_like or sequence of array_like
            The stacked coefficients, or one array per state.
        theta : array_like
            ``(p,)`` parameters.

        Returns
        -------
        Residuals
            ``data``, ``equation``, ``jac_coefs``, ``jac_theta``.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.profiling import ProfiledODE, ODEModel
        >>> m = ODEModel(lambda x, t, th: -th[0] * x, 1, 1)
        >>> t = np.linspace(0.0, 1.0, 5)
        >>> p = ProfiledODE(m, t, np.ones((5, 1)), fdt.BSpline(n_basis=4))
        >>> r = p.residuals(np.ones(4), [0.0])
        >>> float(abs(r.data).max()), float(abs(r.equation).max())
        (0.0, 0.0)
        """
        return self._residuals(self._flat(coefs), self._theta(theta), jacobian=True)

    def criterion(self, coefs: Any, theta: Any) -> float:
        """Evaluate the inner criterion ``J(c | theta)``.

        Parameters
        ----------
        coefs : array_like or sequence of array_like
            Coefficients.
        theta : array_like
            Parameters.

        Returns
        -------
        float
            Sum of squared data and equation residuals.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.profiling import ProfiledODE, ODEModel
        >>> m = ODEModel(lambda x, t, th: -th[0] * x, 1, 1)
        >>> t = np.linspace(0.0, 1.0, 5)
        >>> p = ProfiledODE(m, t, np.zeros((5, 1)), fdt.BSpline(n_basis=4))
        >>> p.criterion(np.ones(4), [0.0])
        5.0
        """
        r = self._residuals(self._flat(coefs), self._theta(theta), jacobian=False)
        return float(r.data @ r.data + r.equation @ r.equation)

    # ------------------------------------------------------------ inner fit

    def _start(self) -> Array:
        """Default start: least-squares fit of each observed state, zero otherwise."""
        xp = default_namespace()
        parts = []
        for bm, yy, size in zip(self._design, self._values, self._sizes, strict=True):
            if yy.shape[0] == 0:
                parts.append(xp.zeros((size,), dtype=xp.float64))
                continue
            gram = xp.matrix_transpose(bm) @ bm
            ridge = 1e-8 * float(xp.max(xp.abs(xp.linalg.diagonal(gram)))) + 1e-300
            parts.append(
                asarray(
                    _linalg.solve_spd(gram + ridge * xp.eye(size), xp.matrix_transpose(bm) @ yy)
                )
            )
        return xp.concat(parts)

    def _hessian(self, flat: Array, theta: Array, res: Residuals) -> tuple[Array, Array]:
        """Exact ``d2J/dc2`` and ``d2J/dc dtheta`` (both halved) at ``(c, theta)``."""
        xp = default_namespace()
        d = self.model.n_states
        x, _ = self._states(flat)
        fxx, fxt = self.model._hess(x, self.nodes, theta)
        jc, jt = res.jac_coefs, res.jac_theta
        hcc = xp.matrix_transpose(jc) @ jc
        hct = xp.matrix_transpose(jc) @ jt
        q = self.nodes.shape[0]
        # weight of the second derivatives of f_i: -(scaled residual) * scale
        g = [-res.equation[i * q : (i + 1) * q] * self._eq_scale[i] for i in range(d)]
        for k in range(d):
            rows = slice(self._offsets[k], self._offsets[k + 1])
            phik_t = xp.matrix_transpose(self._phi[k])
            for m in range(d):
                weight = sum(g[i] * fxx[:, i, k, m] for i in range(d))
                hcc[rows, self._offsets[m] : self._offsets[m + 1]] += phik_t @ (
                    weight[:, None] * self._phi[m]
                )
            mixed = sum(g[i][:, None] * fxt[:, i, k, :] for i in range(d))
            hct[rows, :] += phik_t @ mixed
        return hcc, hct

    def _relative_step(self, step: Array, flat: Array) -> float:
        """Largest step of any state relative to that state's largest coefficient."""
        xp = default_namespace()
        worst = 0.0
        for d_i, c_i in zip(self._split(step), self._split(flat), strict=True):
            size = float(xp.max(xp.abs(c_i)))
            move = float(xp.max(xp.abs(d_i)))
            worst = max(worst, move / size if size > 0.0 else (math.inf if move else 0.0))
        return worst

    def _inner(
        self, flat0: Array, theta: Array, tol: float, max_iter: int
    ) -> tuple[Array, Residuals, int, bool]:
        """Damped Gauss-Newton on ``J(. | theta)`` from ``flat0``.

        Far from the minimum a step must lower the criterion (Marquardt damping
        otherwise).  Once the relative step is below :data:`_FULL_STEP` the
        criterion only changes at rounding level, so full Gauss-Newton steps
        are taken until every state's coefficients have settled: the step is
        below ``tol`` relative, or stops shrinking below :data:`_STALL_STEP`.
        """
        xp = default_namespace()
        flat = flat0
        res = self._residuals(flat, theta, jacobian=True)
        value = float(res.data @ res.data + res.equation @ res.equation)
        damping = 0.0
        previous = math.inf
        for it in range(1, max_iter + 1):
            r = xp.concat([res.data, res.equation])
            jtj = xp.matrix_transpose(res.jac_coefs) @ res.jac_coefs
            grad = xp.matrix_transpose(res.jac_coefs) @ r
            diag = xp.linalg.diagonal(jtj)
            while True:
                system = jtj + damping * xp.diag(diag) if damping else jtj
                step = -_scaled_solve(system, grad, xp, spd=True)
                rel = self._relative_step(step, flat)
                trial = flat + step
                trial_res = self._residuals(trial, theta, jacobian=True)
                trial_value = float(
                    trial_res.data @ trial_res.data + trial_res.equation @ trial_res.equation
                )
                if math.isfinite(trial_value) and (
                    trial_value <= value or (damping == 0.0 and rel <= _FULL_STEP)
                ):
                    damping = damping / 10.0 if damping > 1e-12 else 0.0
                    break
                damping = 1e-6 if damping == 0.0 else damping * 10.0
                if damping > _MAX_DAMPING:
                    return flat, res, it, False
            flat, res, value = trial, trial_res, trial_value
            if rel <= tol or (rel <= _STALL_STEP and rel > previous / 2.0):
                return flat, res, it, True
            previous = rel
        return flat, res, max_iter, False

    def _inner_fit(
        self, flat: Array, theta: Array, res: Residuals, n_iter: int, converged: bool
    ) -> InnerFit:
        xp = default_namespace()
        hcc, hct = self._hessian(flat, theta, res)
        dcdt = -_scaled_solve(hcc, hct, xp, spd=False)
        n_data = res.data.shape[0]
        jd = res.jac_coefs[:n_data, :]
        # d(fitted)/d(y) = -jd H^{-1} jd^T  (in the weighted scale)
        hinv_jdt = _scaled_solve(hcc, xp.matrix_transpose(jd), xp, spd=False)
        df = float(xp.sum(jd * xp.matrix_transpose(hinv_jdt)))
        sse = float(res.data @ res.data)
        n = self.n_obs
        gcv = n * sse / (n - df) ** 2 if n > df else math.inf
        return InnerFit(
            coefs=tuple(xp.asarray(c, copy=True) for c in self._split(flat)),
            criterion=sse + float(res.equation @ res.equation),
            sse=sse,
            df=df,
            gcv=gcv,
            dcoefs_dtheta=dcdt,
            n_iter=n_iter,
            converged=converged,
        )

    def fit_states(
        self,
        theta: Any,
        coef0: Any = None,
        *,
        tol: float = 1e-14,
        max_iter: int = 200,
    ) -> InnerFit:
        """Fit the states for fixed parameters (the inner problem, R's ``CSTRfn``).

        Parameters
        ----------
        theta : array_like
            ``(p,)`` parameters.
        coef0 : array_like or sequence, optional
            Starting coefficients; by default a least-squares fit of each
            observed state and zeros for unobserved ones.
        tol : float, optional
            Step size, relative to each state's largest coefficient, at which
            Gauss-Newton stops.
        max_iter : int, optional
            Iteration cap.

        Returns
        -------
        InnerFit
            Coefficients, criterion, degrees of freedom and ``dc/dtheta``.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.profiling import ProfiledODE, ODEModel
        >>> m = ODEModel(lambda x, t, th: -th[0] * x, 1, 1)
        >>> t = np.linspace(0.0, 1.0, 11)
        >>> p = ProfiledODE(m, t, np.exp(-t)[:, None], fdt.BSpline(n_basis=8), lam=1e2)
        >>> inner = p.fit_states([1.0])
        >>> inner.converged, inner.sse < 1e-8
        (True, True)
        """
        th = self._theta(theta)
        flat0 = self._start() if coef0 is None else self._flat(coef0)
        flat, res, n_iter, converged = self._inner(flat0, th, tol, max_iter)
        return self._inner_fit(flat, th, res, n_iter, converged)

    # ------------------------------------------------------------ outer fit

    def fit(
        self,
        theta0: Any,
        coef0: Any = None,
        *,
        tol: float = 1e-8,
        max_iter: int = 200,
        inner_tol: float = 1e-14,
        inner_max_iter: int = 200,
    ) -> ProfileResult:
        """Estimate the parameters by Gauss-Newton on the profiled criterion.

        Each outer step re-fits the states (warm-started from the previous
        fit), forms the data-residual Jacobian ``A = -Phi dc/dtheta`` by the
        implicit function theorem, and takes a Gauss-Newton step on
        ``theta`` with Marquardt damping when the criterion would increase.
        Iteration stops when the relative-offset criterion of Bates and Watts
        (as R's ``nls``), ``|Q'e| / sqrt(p) / (|e| / sqrt(N - p))``, falls
        below ``tol``, or the step is at rounding level.

        Parameters
        ----------
        theta0 : array_like
            ``(p,)`` starting parameters.
        coef0 : array_like or sequence, optional
            Starting coefficients for the first inner fit.
        tol : float, optional
            Relative-offset tolerance.
        max_iter : int, optional
            Outer iteration cap.
        inner_tol, inner_max_iter : optional
            Passed to the inner fits (see :meth:`fit_states`).

        Returns
        -------
        ProfileResult
            Estimates, covariance, fitted states and the iteration path.

        Raises
        ------
        ValueError
            If there are no more observations than parameters.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.profiling import ProfiledODE, ODEModel
        >>> m = ODEModel(lambda x, t, th: -th[0] * x, 1, 1)
        >>> t = np.linspace(0.0, 1.0, 21)
        >>> basis = fdt.BSpline(domain=(0.0, 1.0), breaks=np.linspace(0.0, 1.0, 11))
        >>> fit = ProfiledODE(m, t, 2 * np.exp(-0.7 * t)[:, None], basis, lam=1e4).fit([0.2])
        >>> round(float(fit.theta[0]), 5)
        0.7
        """
        xp = default_namespace()
        p = self.model.n_params
        n = self.n_obs
        if n <= p:
            raise ValueError(f"need more observations ({n}) than parameters ({p})")
        theta = self._theta(theta0)
        flat0 = self._start() if coef0 is None else self._flat(coef0)
        flat, res, it_in, ok_in = self._inner(flat0, theta, inner_tol, inner_max_iter)
        inner = self._inner_fit(flat, theta, res, it_in, ok_in)
        path = [theta]
        sse_path = [inner.sse]
        damping = 0.0
        converged = False
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            n_data = res.data.shape[0]
            a = res.jac_coefs[:n_data, :] @ inner.dcoefs_dtheta
            e = res.data
            qmat, _ = xp.linalg.qr(a)
            projected = xp.matrix_transpose(qmat) @ e
            norm_e = math.sqrt(float(e @ e))
            offset = (
                math.sqrt(float(projected @ projected) / p) / (norm_e / math.sqrt(n - p))
                if norm_e > 0.0
                else 0.0
            )
            if offset < tol:
                converged = True
                n_iter -= 1
                break
            ata = xp.matrix_transpose(a) @ a
            grad = xp.matrix_transpose(a) @ e
            diag = xp.linalg.diagonal(ata)
            accepted = False
            while damping <= _MAX_DAMPING:
                system = ata + damping * xp.diag(diag) if damping else ata
                step = -_scaled_solve(system, grad, xp, spd=True)
                trial = theta + step
                t_flat, t_res, t_it, t_ok = self._inner(flat, trial, inner_tol, inner_max_iter)
                t_sse = float(t_res.data @ t_res.data)
                if math.isfinite(t_sse) and t_sse <= inner.sse * (1.0 + 1e-14):
                    accepted = True
                    damping = damping / 10.0 if damping > 1e-12 else 0.0
                    break
                damping = 1e-6 if damping == 0.0 else damping * 10.0
            if not accepted:
                # No step lowers the criterion: theta is at its minimum to the
                # precision of the inner fits.  That is convergence when the
                # Gauss-Newton model predicted a negligible decrease anyway.
                converged = float(projected @ projected) <= _STALL_DECREASE * norm_e**2
                n_iter -= 1
                break
            tiny = float(xp.max(xp.abs(step))) <= _STEP_TOL * float(xp.max(xp.abs(theta)))
            theta, flat, res = trial, t_flat, t_res
            inner = self._inner_fit(flat, theta, res, t_it, t_ok)
            path.append(theta)
            sse_path.append(inner.sse)
            if tiny:
                converged = True
                break
        n_data = res.data.shape[0]
        a = res.jac_coefs[:n_data, :] @ inner.dcoefs_dtheta
        sigma2 = inner.sse / (n - p)
        ata = xp.matrix_transpose(a) @ a
        cov = sigma2 * _scaled_solve(ata, xp.eye(p, dtype=xp.float64), xp, spd=True)
        cov = 0.5 * (cov + xp.matrix_transpose(cov))
        states = tuple(
            FData(xp.reshape(c, (-1, 1)), b) for c, b in zip(inner.coefs, self.bases, strict=True)
        )
        return ProfileResult(
            theta=xp.asarray(theta, copy=True),
            cov=cov,
            sigma2=sigma2,
            states=states,
            inner=inner,
            theta_path=xp.stack(path),
            sse_path=xp.asarray(sse_path, dtype=xp.float64),
            n_iter=n_iter,
            converged=converged,
            param_names=self.model.param_names,
            state_names=self.model.state_names,
        )


def profile_ode(
    model: ODEModel,
    t: Any,
    y: Any,
    bases: Basis | Sequence[Basis],
    lam: float | Sequence[float],
    theta0: Any,
    *,
    state_weights: float | Sequence[float] = 1.0,
    quadrature: tuple[Any, Any] | None = None,
    n_quad: int = 5,
    coef0: Any = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> ProfileResult:
    """Estimate ODE parameters from noisy data by generalized profiling.

    A one-call front end to :class:`ProfiledODE`; see there and the module
    documentation for the method and the arguments.

    Parameters
    ----------
    model : ODEModel
        The right-hand side ``f(x, t, theta)``.
    t, y : array_like or sequence
        Observations (see :class:`ProfiledODE`).
    bases : Basis or sequence of Basis
        State bases.
    lam : float or sequence of float
        Equation weights.
    theta0 : array_like
        Starting parameters.
    state_weights : float or sequence of float, optional
        Per-state weights.
    quadrature : tuple of array_like, optional
        ``(nodes, weights)``.
    n_quad : int, optional
        Simpson nodes per panel.
    coef0 : array_like or sequence, optional
        Starting coefficients.
    tol : float, optional
        Outer relative-offset tolerance.
    max_iter : int, optional
        Outer iteration cap.

    Returns
    -------
    ProfileResult
        Estimates, covariance, fitted states and the iteration path.

    Examples
    --------
    >>> import numpy as np
    >>> import fdatools as fdt
    >>> from fdatools.profiling import fitzhugh_nagumo_model, profile_ode
    >>> fhn = fitzhugh_nagumo_model()
    >>> t = np.linspace(0.0, 20.0, 201)
    >>> x = fhn.simulate(t, [-1.0, 1.0], [0.2, 0.2, 3.0])
    >>> y = np.column_stack([x[:, 0], np.full(201, np.nan)])  # R is never observed
    >>> basis = fdt.BSpline(domain=(0.0, 20.0), breaks=np.linspace(0.0, 20.0, 201))
    >>> fit = profile_ode(fhn, t, y, basis, lam=1e3, theta0=[0.25, 0.25, 2.8])
    >>> np.round(fit.theta, 2).tolist()
    [0.2, 0.2, 3.0]
    """
    problem = ProfiledODE(
        model, t, y, bases, lam, state_weights=state_weights, quadrature=quadrature, n_quad=n_quad
    )
    return problem.fit(theta0, coef0, tol=tol, max_iter=max_iter)
