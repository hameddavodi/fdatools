"""Bridges between fdatools and the wider scientific and econometric Python stack.

Functional data rarely starts or ends inside one library.  This module moves
curves in and out of fdatools without loss wherever the mathematics allows it,
and turns functional covariates into ordinary design matrices that any linear
estimator can fit:

* **NumPy** -- :func:`from_grid` and :func:`to_grid` convert between
  :class:`~fdatools.core.FData` and plain sampled arrays, in either the
  fdatools layout ``(n_points, n_curves)`` or the machine-learning layout
  ``(n_curves, n_points)``.
* **SciPy** -- :func:`to_scipy` and :func:`from_scipy` convert spline curves to
  and from :class:`scipy.interpolate.BSpline` exactly (same knots, same
  coefficients).
* **scikit-fda** -- :func:`to_skfda` and :func:`from_skfda` convert to and from
  ``FDataBasis`` (exactly, for B-spline, Fourier, monomial and constant bases)
  and ``FDataGrid``.
* **Panel econometrics** (linearmodels, statsmodels, pandas) --
  :func:`from_panel` smooths a ``(entity, time)`` MultiIndex panel into one
  curve per entity, and :func:`to_panel` writes curves back in that layout.
* **Any linear estimator** -- :func:`functional_design` builds the design matrix
  of a scalar-on-function model ``y_i = a + z_i'g + int x_i(t) beta(t) dt + e_i``.
  Fit it with ``statsmodels.OLS``/``WLS``/``GLM``/``QuantReg``, ``linearmodels``
  ``PanelOLS``/``IV2SLS``, or scikit-learn, then map the estimated parameters
  and their covariance back to ``beta(t)`` with
  :meth:`FunctionalDesign.beta` and :meth:`FunctionalDesign.band`.

``scipy`` is a core dependency; ``pandas`` (the ``pandas`` extra) and
``scikit-fda`` are imported lazily, only by the functions that need them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from fdatools._backend import to_numpy
from fdatools.basis import Basis, BSpline, Constant, Fourier, Monomial, Polygonal
from fdatools.core import FData, inprod
from fdatools.smoothing import smooth

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    from scipy.interpolate import BSpline as ScipyBSpline

__all__ = [
    "FunctionalDesign",
    "PanelCurves",
    "from_grid",
    "from_panel",
    "from_scipy",
    "from_skfda",
    "functional_design",
    "to_grid",
    "to_panel",
    "to_scipy",
    "to_skfda",
]

#: Default number of evaluation points used when no grid is given.
_DEFAULT_POINTS = 101

#: Relative tolerance used to decide that a knot vector is clamped.
_KNOT_TOL = 1e-12


def _curve_axis(axis: int) -> int:
    """Validate a curve-axis argument, which must be ``0`` or ``1``."""
    if axis not in (0, 1):
        raise ValueError(f"curve_axis must be 0 or 1, got {axis}")
    return axis


def _default_grid(fd: FData, n_points: int) -> NDArray[np.float64]:
    """Return ``n_points`` equally spaced points spanning the domain of ``fd``."""
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    lower, upper = fd.domain
    return np.linspace(lower, upper, n_points)


# --------------------------------------------------------------------------- #
# NumPy
# --------------------------------------------------------------------------- #


def from_grid(
    values: Any,
    t: Any = None,
    *,
    basis: Basis | None = None,
    lam: float | str = "gcv",
    curve_axis: int = 1,
) -> FData:
    """Turn curves sampled on a common grid into an :class:`~fdatools.core.FData`.

    A thin, NumPy-friendly front end to :func:`~fdatools.smoothing.smooth`.

    Parameters
    ----------
    values : array_like
        Sampled curves, shape ``(n_points,)``, ``(n_points, n_curves)`` (the
        fdatools layout, ``curve_axis=1``) or ``(n_curves, n_points)`` (the
        scikit-learn / pandas-wide layout, ``curve_axis=0``).  A third axis
        holds variables.
    t : array_like, optional
        The ``n_points`` sampling points.  Defaults to ``0, 1, ..., n_points - 1``.
    basis : Basis, optional
        Expansion basis.  Defaults to the default of
        :func:`~fdatools.smoothing.smooth`.
    lam : float or str, optional
        Smoothing parameter, as in :func:`~fdatools.smoothing.smooth`.  The
        default ``"gcv"`` chooses it by generalised cross-validation; ``0.0``
        gives plain least squares.
    curve_axis : {0, 1}, optional
        Which axis of ``values`` indexes the curves.  Default ``1``.

    Returns
    -------
    FData
        The fitted curves.

    Raises
    ------
    ValueError
        If ``curve_axis`` is not 0 or 1, or ``t`` does not match the number of
        points.

    Examples
    --------
    >>> import numpy as np
    >>> import fdatools as fdt
    >>> from fdatools.interop import from_grid
    >>> t = np.linspace(0.0, 1.0, 50)
    >>> X = np.sin(2 * np.pi * np.outer(np.arange(1, 4), t))  # 3 curves, one per row
    >>> fd = from_grid(X, t, basis=fdt.BSpline(n_basis=15), lam=0.0, curve_axis=0)
    >>> fd.n_curves
    3
    """
    axis = _curve_axis(curve_axis)
    data = np.asarray(to_numpy(values), dtype=np.float64)
    if data.ndim >= 2 and axis == 0:
        data = np.swapaxes(data, 0, 1)
    points = np.arange(data.shape[0], dtype=np.float64) if t is None else np.asarray(t, float)
    if points.ndim != 1 or points.shape[0] != data.shape[0]:
        raise ValueError(f"t must hold {data.shape[0]} points, got shape {points.shape}")
    return smooth(data, points, basis=basis, lam=lam).fd


def to_grid(
    fd: FData,
    t: Any = None,
    *,
    n_points: int = _DEFAULT_POINTS,
    deriv: int = 0,
    curve_axis: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample curves on a grid and return plain NumPy arrays.

    Parameters
    ----------
    fd : FData
        The curves.
    t : array_like, optional
        Evaluation points.  Defaults to ``n_points`` equally spaced points
        across the domain.
    n_points : int, optional
        Grid size used when ``t`` is not given.  Default ``101``.
    deriv : int, optional
        Derivative order to evaluate.  Default ``0``.
    curve_axis : {0, 1}, optional
        Axis of the returned values that indexes the curves: ``1`` gives
        ``(n_points, n_curves)``, ``0`` gives ``(n_curves, n_points)``, the
        layout scikit-learn and statsmodels expect for a feature matrix.

    Returns
    -------
    t : ndarray
        The evaluation points.
    values : ndarray
        The sampled curves.

    Examples
    --------
    >>> import numpy as np
    >>> import fdatools as fdt
    >>> from fdatools.interop import to_grid
    >>> fd = fdt.FData(np.eye(5), fdt.BSpline(n_basis=5))
    >>> t, X = to_grid(fd, n_points=11, curve_axis=0)
    >>> X.shape
    (5, 11)
    """
    axis = _curve_axis(curve_axis)
    points = _default_grid(fd, n_points) if t is None else np.asarray(to_numpy(t), float)
    values = np.asarray(to_numpy(fd(points, deriv)), dtype=np.float64)
    if axis == 0:
        values = np.swapaxes(values, 0, 1)
    return points, values


# --------------------------------------------------------------------------- #
# SciPy
# --------------------------------------------------------------------------- #


def _spline_parts(basis: Basis) -> tuple[tuple[float, ...], int]:
    """Return the knot vector and order of a spline-type basis."""
    if isinstance(basis, BSpline):
        return basis.knots, basis.order
    if isinstance(basis, Polygonal):
        argvals = tuple(float(v) for v in basis.argvals)
        return (argvals[0], *argvals, argvals[-1]), 2
    raise TypeError(
        f"only BSpline and Polygonal bases convert exactly to scipy, got {type(basis).__name__}; "
        "use to_grid() and scipy.interpolate.make_interp_spline for other bases"
    )


def to_scipy(fd: FData, *, extrapolate: bool = False) -> ScipyBSpline:
    """Convert spline curves to one :class:`scipy.interpolate.BSpline`, exactly.

    The knots and coefficients are carried over unchanged, so the SciPy object
    evaluates, differentiates (``.derivative()``) and integrates
    (``.integrate()``) to the same function, to rounding.

    Parameters
    ----------
    fd : FData
        Curves in a :class:`~fdatools.basis.BSpline` or
        :class:`~fdatools.basis.Polygonal` basis.
    extrapolate : bool, optional
        Passed to SciPy.  Default ``False``: outside the domain the spline
        returns NaN, matching the fact that fdatools curves have a domain.

    Returns
    -------
    scipy.interpolate.BSpline
        A vector-valued spline: ``spl(t)`` has shape ``(len(t), n_curves)`` (or
        ``(len(t), n_curves, n_vars)``), like ``fd(t)``.

    Raises
    ------
    TypeError
        If the basis is not a spline basis.

    Examples
    --------
    >>> import numpy as np
    >>> import fdatools as fdt
    >>> from fdatools.interop import to_scipy
    >>> fd = fdt.FData(np.arange(12.0).reshape(6, 2), fdt.BSpline(n_basis=6))
    >>> spl = to_scipy(fd)
    >>> t = np.array([0.1, 0.7])
    >>> bool(np.allclose(spl(t), fd(t)))
    True
    """
    from scipy.interpolate import BSpline as ScipyBSpline

    knots, order = _spline_parts(fd.basis)
    coefs = np.asarray(to_numpy(fd.coefs), dtype=np.float64)
    return ScipyBSpline(np.asarray(knots), coefs, order - 1, extrapolate=extrapolate)


def from_scipy(spline: Any) -> FData:
    """Convert a clamped :class:`scipy.interpolate.BSpline` to an :class:`~fdatools.core.FData`.

    Accepts the output of ``make_interp_spline``, ``make_lsq_spline``,
    ``make_smoothing_spline``, ``splrep`` (as a ``BSpline``) and friends, as
    long as the knot vector is clamped -- both end knots repeated ``k + 1``
    times -- which is what those constructors produce by default.

    Parameters
    ----------
    spline : scipy.interpolate.BSpline
        The spline.  A coefficient array with trailing axes becomes several
        curves (second axis) and variables (third axis).

    Returns
    -------
    FData
        The same function in a :class:`~fdatools.basis.BSpline` basis.

    Raises
    ------
    ValueError
        If the knot vector is not clamped, or ``spline.axis`` is not ``0``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import make_interp_spline
    >>> from fdatools.interop import from_scipy
    >>> x = np.linspace(0.0, 1.0, 9)
    >>> spl = make_interp_spline(x, np.cos(x), k=3)
    >>> fd = from_scipy(spl)
    >>> bool(np.allclose(fd(x)[:, 0], np.cos(x)))
    True
    """
    if int(getattr(spline, "axis", 0)) != 0:
        raise ValueError("from_scipy needs a spline with axis=0")
    knots = np.asarray(spline.t, dtype=np.float64)
    order = int(spline.k) + 1
    coefs = np.asarray(spline.c, dtype=np.float64)
    n_basis = knots.size - order
    lower, upper = float(knots[order - 1]), float(knots[n_basis])
    scale = max(1.0, abs(upper - lower))
    clamped = np.allclose(knots[:order], lower, rtol=0.0, atol=_KNOT_TOL * scale) and np.allclose(
        knots[n_basis:], upper, rtol=0.0, atol=_KNOT_TOL * scale
    )
    if not clamped:
        raise ValueError(
            "from_scipy needs a clamped knot vector (end knots repeated k + 1 times); "
            "refit with make_lsq_spline on clamped knots, or use from_grid()"
        )
    breaks = (lower, *(float(v) for v in knots[order:n_basis]), upper)
    basis = BSpline(domain=(lower, upper), order=order, breaks=breaks)
    return FData(coefs[:n_basis], basis)


# --------------------------------------------------------------------------- #
# scikit-fda
# --------------------------------------------------------------------------- #


def _skfda_basis(basis: Basis) -> Any:
    """Return the scikit-fda basis that spans exactly the same functions, or ``None``."""
    from skfda.representation import basis as skb

    if isinstance(basis, BSpline):
        return skb.BSplineBasis(domain_range=basis.domain, order=basis.order, knots=basis.breaks)
    if isinstance(basis, Polygonal):
        argvals = tuple(float(v) for v in basis.argvals)
        return skb.BSplineBasis(domain_range=basis.domain, order=2, knots=argvals)
    if isinstance(basis, Fourier):
        return skb.FourierBasis(
            domain_range=basis.domain, n_basis=basis.n_basis, period=basis.period
        )
    if isinstance(basis, Monomial) and tuple(basis.exponents) == tuple(range(basis.n_basis)):
        return skb.MonomialBasis(domain_range=basis.domain, n_basis=basis.n_basis)
    if isinstance(basis, Constant):
        return skb.ConstantBasis(domain_range=basis.domain)
    return None


def _fdatools_basis(basis: Any) -> Basis | None:
    """Return the fdatools basis matching a scikit-fda basis, or ``None``."""
    from skfda.representation import basis as skb

    lower, upper = (float(v) for v in basis.domain_range[0])
    domain = (lower, upper)
    if isinstance(basis, skb.BSplineBasis):
        return BSpline(domain=domain, order=basis.order, breaks=list(basis.knots))
    if isinstance(basis, skb.FourierBasis):
        return Fourier(domain=domain, n_basis=basis.n_basis, period=float(basis.period))
    if isinstance(basis, skb.MonomialBasis):
        return Monomial(domain=domain, exponents=tuple(range(basis.n_basis)))
    if isinstance(basis, skb.ConstantBasis):
        return Constant(domain=domain)
    return None


def to_skfda(fd: FData, t: Any = None, *, n_points: int = _DEFAULT_POINTS) -> Any:
    """Convert curves to a scikit-fda object.

    Parameters
    ----------
    fd : FData
        The curves, with no variable axis.
    t : array_like, optional
        Grid used when the basis has no scikit-fda counterpart.  Defaults to
        ``n_points`` points across the domain.
    n_points : int, optional
        Grid size used when ``t`` is not given.  Default ``101``.

    Returns
    -------
    skfda.FDataBasis or skfda.FDataGrid
        An exact ``FDataBasis`` for B-spline, polygonal, Fourier, monomial
        (consecutive exponents from 0) and constant bases; otherwise the
        curves sampled on ``t`` as an ``FDataGrid``.

    Raises
    ------
    ImportError
        If scikit-fda is not installed.
    ValueError
        If ``fd`` carries a variable axis.

    Examples
    --------
    >>> import numpy as np
    >>> import fdatools as fdt
    >>> from fdatools.interop import to_skfda
    >>> fd = fdt.FData(np.eye(5), fdt.BSpline(n_basis=5))
    >>> to_skfda(fd).n_samples  # doctest: +SKIP
    5
    """
    try:
        from skfda import FDataBasis, FDataGrid
    except ImportError as exc:  # pragma: no cover - exercised only without skfda
        raise ImportError("to_skfda requires scikit-fda: uv add scikit-fda") from exc
    if fd.n_vars != 1:
        raise ValueError("to_skfda supports curves without a variable axis")
    target = _skfda_basis(fd.basis)
    if target is not None:
        coefs = np.asarray(to_numpy(fd.coefs), dtype=np.float64).T
        return FDataBasis(target, coefs)
    points, values = to_grid(fd, t, n_points=n_points, curve_axis=0)
    return FDataGrid(values, grid_points=points, domain_range=fd.domain)


def from_skfda(
    obj: Any,
    *,
    basis: Basis | None = None,
    lam: float | str = "gcv",
) -> FData:
    """Convert a scikit-fda ``FDataBasis`` or ``FDataGrid`` to an :class:`~fdatools.core.FData`.

    Parameters
    ----------
    obj : skfda.FDataBasis or skfda.FDataGrid
        One-dimensional-domain, scalar-valued functional data.
    basis : Basis, optional
        Target basis for an ``FDataGrid`` (or for an ``FDataBasis`` whose basis
        has no exact fdatools counterpart).  Defaults to the default of
        :func:`~fdatools.smoothing.smooth`.
    lam : float or str, optional
        Smoothing parameter for the grid path.  Default ``"gcv"``.

    Returns
    -------
    FData
        Exact for ``FDataBasis`` on B-spline, Fourier, monomial and constant
        bases; otherwise a penalised fit to the sampled values.

    Raises
    ------
    ValueError
        If ``obj`` has a multidimensional domain or codomain.
    TypeError
        If ``obj`` is neither an ``FDataBasis`` nor an ``FDataGrid``.

    Examples
    --------
    >>> import numpy as np
    >>> from skfda import FDataGrid  # doctest: +SKIP
    >>> from fdatools.interop import from_skfda  # doctest: +SKIP
    >>> grid = FDataGrid(np.random.rand(3, 20), np.linspace(0, 1, 20))  # doctest: +SKIP
    >>> from_skfda(grid).n_curves  # doctest: +SKIP
    3
    """
    from skfda import FDataBasis, FDataGrid

    if obj.dim_domain != 1 or obj.dim_codomain != 1:
        raise ValueError("from_skfda supports scalar functions of one argument")
    if isinstance(obj, FDataBasis):
        exact = _fdatools_basis(obj.basis) if basis is None else None
        if exact is not None:
            return FData(np.asarray(obj.coefficients, dtype=np.float64).T, exact)
        domain = tuple(float(v) for v in obj.domain_range[0])
        points = np.linspace(domain[0], domain[1], _DEFAULT_POINTS)
        values = np.asarray(obj(points), dtype=np.float64)[:, :, 0]
        return from_grid(values, points, basis=basis, lam=lam, curve_axis=0)
    if isinstance(obj, FDataGrid):
        points = np.asarray(obj.grid_points[0], dtype=np.float64)
        values = np.asarray(obj.data_matrix, dtype=np.float64)[:, :, 0]
        return from_grid(values, points, basis=basis, lam=lam, curve_axis=0)
    raise TypeError(f"expected an FDataBasis or FDataGrid, got {type(obj).__name__}")


# --------------------------------------------------------------------------- #
# Panel data (linearmodels / statsmodels / pandas)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class PanelCurves:
    """Curves smoothed from a panel, one per entity.

    Attributes
    ----------
    fd : FData
        One curve per entity, in the order of :attr:`entities`.
    entities : list
        Entity labels, in first-seen order.
    """

    fd: FData
    entities: list[Any]


def from_panel(
    data: Any,
    value: str | None = None,
    *,
    basis: Basis | None = None,
    lam: float | str = "gcv",
) -> PanelCurves:
    """Smooth a ``(entity, time)`` panel into one curve per entity.

    The input layout is the one used by ``linearmodels.PanelOLS`` and
    ``statsmodels`` panel tools: a two-level :class:`pandas.MultiIndex` with
    the entity first and a numeric time second.  Balanced panels share one
    grid; unbalanced panels (missing periods, NaNs) are fitted per entity.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        The panel.  Rows with a missing value are dropped.
    value : str, optional
        Column to smooth when ``data`` is a DataFrame.  Required unless it has
        exactly one column.
    basis : Basis, optional
        Expansion basis.  Defaults to a cubic B-spline on the observed time
        range (see :func:`~fdatools.smoothing.smooth`).
    lam : float or str, optional
        Smoothing parameter, as in :func:`~fdatools.smoothing.smooth`.

    Returns
    -------
    PanelCurves
        The curves and their entity labels.

    Raises
    ------
    ValueError
        If the index is not a two-level MultiIndex, ``value`` is ambiguous or
        missing, or the times are not numeric.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import fdatools as fdt
    >>> from fdatools.interop import from_panel
    >>> idx = pd.MultiIndex.from_product([["a", "b"], np.arange(10.0)], names=["firm", "year"])
    >>> s = pd.Series(np.tile(np.linspace(0, 1, 10), 2), index=idx)
    >>> panel = from_panel(s, basis=fdt.BSpline(domain=(0.0, 9.0), n_basis=5), lam=0.0)
    >>> panel.entities, panel.fd.n_curves
    (['a', 'b'], 2)
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        if value is None:
            if data.shape[1] != 1:
                raise ValueError(f"name the column to smooth with value=, got {list(data.columns)}")
            value = str(data.columns[0])
        if value not in data.columns:
            raise ValueError(f"column {value!r} not found in the panel")
        series = data[value]
    else:
        series = pd.Series(data)
    index = series.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise ValueError("from_panel needs a two-level (entity, time) MultiIndex")
    series = series.dropna()
    entity_codes = series.index.get_level_values(0)
    try:
        times = np.asarray(series.index.get_level_values(1), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("the time level of the panel must be numeric") from exc
    entities = list(dict.fromkeys(entity_codes))
    observed = series.to_numpy(dtype=np.float64)
    per_t: list[NDArray[np.float64]] = []
    per_y: list[NDArray[np.float64]] = []
    for entity in entities:
        mask = np.asarray(entity_codes == entity)
        order = np.argsort(times[mask], kind="stable")
        per_t.append(times[mask][order])
        per_y.append(observed[mask][order])
    shared = all(arr.shape == per_t[0].shape and np.array_equal(arr, per_t[0]) for arr in per_t[1:])
    if shared:
        fd = smooth(np.column_stack(per_y), per_t[0], basis=basis, lam=lam).fd
    else:
        if basis is None:
            lower = float(min(arr.min() for arr in per_t))
            upper = float(max(arr.max() for arr in per_t))
            distinct = np.unique(np.concatenate(per_t)).size
            basis = BSpline(domain=(lower, upper), n_basis=min(distinct, 40) + 2)
        fd = smooth(per_y, per_t, basis=basis, lam=lam).fd
    return PanelCurves(fd=fd, entities=entities)


def to_panel(
    fd: FData,
    t: Any = None,
    *,
    entities: Sequence[Any] | None = None,
    names: tuple[str, str] = ("entity", "time"),
    value: str = "value",
    n_points: int = _DEFAULT_POINTS,
    deriv: int = 0,
) -> pd.DataFrame:
    """Evaluate curves on ``t`` and return an ``(entity, time)`` panel DataFrame.

    The result plugs straight into ``linearmodels.PanelOLS`` /
    ``RandomEffects`` or a ``statsmodels`` formula with entity effects.

    Parameters
    ----------
    fd : FData
        The curves, with no variable axis.
    t : array_like, optional
        Evaluation times.  Defaults to ``n_points`` points across the domain.
    entities : sequence, optional
        One label per curve.  Defaults to ``0, 1, ..., n_curves - 1``.
    names : tuple of str, optional
        Names of the two index levels.  Default ``("entity", "time")``.
    value : str, optional
        Name of the value column.  Default ``"value"``.
    n_points : int, optional
        Grid size used when ``t`` is not given.  Default ``101``.
    deriv : int, optional
        Derivative order to evaluate.  Default ``0``.

    Returns
    -------
    pandas.DataFrame
        One column, indexed by a two-level MultiIndex sorted by entity then time.

    Raises
    ------
    ValueError
        If ``entities`` has the wrong length, or ``fd`` carries a variable axis.

    Examples
    --------
    >>> import numpy as np
    >>> import fdatools as fdt
    >>> from fdatools.interop import to_panel
    >>> fd = fdt.FData(np.eye(4)[:, :2], fdt.BSpline(n_basis=4))
    >>> to_panel(fd, np.array([0.0, 1.0]), entities=["a", "b"]).shape
    (4, 1)
    """
    import pandas as pd

    if fd.n_vars != 1:
        raise ValueError("to_panel supports curves without a variable axis")
    labels = list(range(fd.n_curves)) if entities is None else list(entities)
    if len(labels) != fd.n_curves:
        raise ValueError(f"{len(labels)} entity labels for {fd.n_curves} curves")
    points, values = to_grid(fd, t, n_points=n_points, deriv=deriv, curve_axis=0)
    index = pd.MultiIndex.from_product([labels, points], names=list(names))
    return pd.DataFrame({value: values.reshape(-1)}, index=index)


# --------------------------------------------------------------------------- #
# Design matrices for any linear estimator
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class _Block:
    """One functional covariate's slice of a design matrix."""

    name: str
    basis: Basis
    columns: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class FunctionalDesign:
    r"""Design matrix of a scalar-on-function linear model.

    Expanding each coefficient function in a basis, ``beta(t) = sum_k b_k
    theta_k(t)``, turns ``int x_i(t) beta(t) dt`` into ``z_i' b`` with
    ``z_ik = int x_i(t) theta_k(t) dt``.  Any linear estimator fitted to
    :attr:`matrix` therefore estimates ``b``, and its covariance matrix gives
    pointwise standard errors of ``beta(t)``.

    Build one with :func:`functional_design`.

    Attributes
    ----------
    matrix : ndarray, shape (n, n_columns)
        Scalar covariates (if any) followed by the functional blocks, with an
        intercept column first when requested.
    columns : tuple of str
        Column labels, ``"name[k]"`` for functional blocks.
    """

    matrix: NDArray[np.float64]
    columns: tuple[str, ...]
    _blocks: tuple[_Block, ...] = field(repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of :attr:`matrix`.

        Returns
        -------
        tuple of int
            ``(n_observations, n_columns)``.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.interop import functional_design
        >>> x = fdt.FData(np.eye(6), fdt.BSpline(n_basis=6))
        >>> functional_design({"x": x}, n_basis=4).shape
        (6, 5)
        """
        rows, cols = self.matrix.shape
        return int(rows), int(cols)

    @property
    def functional_names(self) -> tuple[str, ...]:
        """Names of the functional covariates.

        Returns
        -------
        tuple of str
            In design order.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.interop import functional_design
        >>> x = fdt.FData(np.eye(6), fdt.BSpline(n_basis=6))
        >>> functional_design({"temp": x}, n_basis=4).functional_names
        ('temp',)
        """
        return tuple(block.name for block in self._blocks)

    def to_frame(self, index: Any = None) -> pd.DataFrame:
        """Return :attr:`matrix` as a labelled :class:`pandas.DataFrame`.

        Labelled columns make ``statsmodels`` and ``linearmodels`` summaries
        readable and let :meth:`beta` pick parameters by name.

        Parameters
        ----------
        index : array_like, optional
            Row index, e.g. the entity-time MultiIndex of a panel.

        Returns
        -------
        pandas.DataFrame
            The design matrix.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.interop import functional_design
        >>> x = fdt.FData(np.eye(6), fdt.BSpline(n_basis=6))
        >>> list(functional_design({"x": x}, n_basis=2).to_frame().columns)
        ['const', 'x[0]', 'x[1]']
        """
        import pandas as pd

        return pd.DataFrame(self.matrix, columns=list(self.columns), index=index)

    def _positions(self, block: _Block) -> list[int]:
        return [self.columns.index(column) for column in block.columns]

    def _block(self, name: str | None) -> _Block:
        if name is None:
            if len(self._blocks) != 1:
                raise ValueError(f"name one of the functional covariates {self.functional_names}")
            return self._blocks[0]
        for block in self._blocks:
            if block.name == name:
                return block
        raise KeyError(f"no functional covariate named {name!r}; have {self.functional_names}")

    def _vector(self, params: Any) -> NDArray[np.float64]:
        """Read estimated parameters, by label when they carry labels."""
        keys = getattr(params, "index", None)
        if keys is not None and all(column in keys for column in self.columns):
            return np.asarray([float(params[column]) for column in self.columns])
        vector = np.asarray(to_numpy(params), dtype=np.float64).reshape(-1)
        if vector.shape[0] != len(self.columns):
            raise ValueError(f"expected {len(self.columns)} parameters, got {vector.shape[0]}")
        return vector

    def _matrix(self, cov: Any) -> NDArray[np.float64]:
        """Read a parameter covariance matrix, by label when it carries labels."""
        rows = getattr(cov, "index", None)
        if rows is not None and all(column in rows for column in self.columns):
            return np.asarray(cov.loc[list(self.columns), list(self.columns)], dtype=np.float64)
        matrix = np.asarray(to_numpy(cov), dtype=np.float64)
        size = len(self.columns)
        if matrix.shape != (size, size):
            raise ValueError(f"expected a {size} x {size} covariance, got {matrix.shape}")
        return matrix

    def beta(self, params: Any, name: str | None = None) -> FData:
        """Map estimated parameters back to a coefficient function ``beta(t)``.

        Parameters
        ----------
        params : array_like or pandas.Series
            The estimator's parameters for :attr:`matrix` (e.g. statsmodels'
            ``result.params``).  A labelled Series is matched by column name, so
            extra or reordered entries are fine.
        name : str, optional
            Which functional covariate.  Optional when there is only one.

        Returns
        -------
        FData
            One curve, in that covariate's coefficient basis.

        Raises
        ------
        KeyError
            If ``name`` is not a functional covariate.
        ValueError
            If ``params`` has the wrong length, or ``name`` is needed.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.interop import functional_design
        >>> rng = np.random.default_rng(0)
        >>> x = fdt.FData(rng.standard_normal((8, 40)), fdt.BSpline(n_basis=8))
        >>> design = functional_design({"x": x}, n_basis=5)
        >>> y = design.matrix @ np.arange(6.0)
        >>> params, *_ = np.linalg.lstsq(design.matrix, y, rcond=None)
        >>> design.beta(params).basis.n_basis
        5
        """
        block = self._block(name)
        vector = self._vector(params)
        return FData(vector[self._positions(block)], block.basis)

    def band(
        self,
        params: Any,
        cov: Any,
        name: str | None = None,
        t: Any = None,
        *,
        level: float = 0.95,
        n_points: int = _DEFAULT_POINTS,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""Return a pointwise confidence band for ``beta(t)``.

        ``Var beta(t) = theta(t)' V theta(t)`` with ``V`` the estimator's
        covariance of the block's parameters, so the band honours whatever
        covariance the estimator computed -- classical, HC/HAC-robust,
        clustered or bootstrapped.

        Parameters
        ----------
        params : array_like or pandas.Series
            Estimated parameters (``result.params``).
        cov : array_like or pandas.DataFrame
            Their covariance (statsmodels ``result.cov_params()``,
            linearmodels ``result.cov``).
        name : str, optional
            Which functional covariate.  Optional when there is only one.
        t : array_like, optional
            Evaluation points.  Defaults to ``n_points`` points across the domain.
        level : float, optional
            Two-sided normal coverage.  Default ``0.95``.
        n_points : int, optional
            Grid size used when ``t`` is not given.  Default ``101``.

        Returns
        -------
        t, estimate, lower, upper : ndarray
            Evaluation points and the band.

        Raises
        ------
        ValueError
            If ``level`` is not in ``(0, 1)`` or the shapes disagree.

        Examples
        --------
        >>> import numpy as np
        >>> import fdatools as fdt
        >>> from fdatools.interop import functional_design
        >>> x = fdt.FData(np.random.default_rng(0).standard_normal((8, 40)), fdt.BSpline(n_basis=8))
        >>> design = functional_design({"x": x}, n_basis=4)
        >>> t, est, lo, hi = design.band(np.ones(5), np.eye(5), n_points=11)
        >>> bool(np.all(lo < hi))
        True
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level}")
        from scipy.stats import norm

        block = self._block(name)
        positions = self._positions(block)
        vector = self._vector(params)[positions]
        covariance = self._matrix(cov)[np.ix_(positions, positions)]
        lower_end, upper_end = block.basis.domain
        points = (
            np.linspace(lower_end, upper_end, n_points)
            if t is None
            else np.asarray(to_numpy(t), float)
        )
        design = np.asarray(to_numpy(block.basis(points)), dtype=np.float64)
        estimate = design @ vector
        stderr = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", design, covariance, design), 0.0))
        z = float(norm.ppf(0.5 + level / 2.0))
        return points, estimate, estimate - z * stderr, estimate + z * stderr


def functional_design(
    functional: Mapping[str, FData] | FData,
    scalar: Mapping[str, Any] | None = None,
    *,
    beta_basis: Mapping[str, Basis] | Basis | None = None,
    n_basis: int = 7,
    intercept: bool = True,
) -> FunctionalDesign:
    """Build the design matrix of a scalar-on-function model for any estimator.

    The model is ``y_i = a + sum_s g_s w_is + sum_f int x_if(t) beta_f(t) dt + e_i``.
    Fit ``y`` on :attr:`FunctionalDesign.matrix` with the estimator of your
    choice -- OLS, WLS, GLM (logit, Poisson), quantile regression, IV/2SLS,
    panel fixed effects -- and recover ``beta_f(t)`` with
    :meth:`FunctionalDesign.beta` and its band with
    :meth:`FunctionalDesign.band`.

    With an ordinary least-squares fit the coefficient functions equal those of
    :func:`~fdatools.regression.fregress` with ``lam=0`` and the same bases.

    Parameters
    ----------
    functional : FData or mapping of str to FData
        The functional covariates, ``n`` curves each.  A bare :class:`FData` is
        named ``"x"``.
    scalar : mapping of str to array_like, optional
        Scalar covariates, ``n`` values each, placed after the intercept.
    beta_basis : Basis or mapping of str to Basis, optional
        Basis for each coefficient function.  Defaults to a B-spline with
        ``n_basis`` functions on the covariate's domain (cubic, or of order
        ``n_basis`` when fewer than four functions are asked for).  Keep it small: an
        unpenalised fit needs ``n`` well above the total column count.
    n_basis : int, optional
        Size of the default coefficient basis.  Default ``7``.
    intercept : bool, optional
        Prepend a column of ones named ``"const"``.  Default ``True``; set it
        to ``False`` for fixed-effects estimators that absorb the constant.

    Returns
    -------
    FunctionalDesign
        The matrix, its column labels and what is needed to map parameters
        back to functions.

    Raises
    ------
    ValueError
        If the covariates disagree on the number of observations, or a
        functional covariate carries a variable axis.

    Examples
    --------
    >>> import numpy as np
    >>> import fdatools as fdt
    >>> from fdatools.interop import functional_design
    >>> growth = fdt.datasets.load_growth()
    >>> height = fdt.smooth(growth.hgtf, growth.age, basis=fdt.BSpline((1.0, 18.0), 12)).fd
    >>> design = functional_design({"height": height}, n_basis=5)
    >>> design.shape
    (54, 6)
    """
    covariates = {"x": functional} if isinstance(functional, FData) else dict(functional)
    if not covariates:
        raise ValueError("functional_design needs at least one functional covariate")
    if n_basis < 1:
        raise ValueError(f"n_basis must be positive, got {n_basis}")
    sizes = {name: fd.n_curves for name, fd in covariates.items()}
    n = next(iter(sizes.values()))
    pieces: list[NDArray[np.float64]] = []
    columns: list[str] = []
    if intercept:
        pieces.append(np.ones((n, 1)))
        columns.append("const")
    for name, values in (scalar or {}).items():
        column = np.asarray(to_numpy(values), dtype=np.float64).reshape(-1)
        sizes[name] = column.shape[0]
        pieces.append(column[:, None])
        columns.append(str(name))
    if len(set(sizes.values())) != 1:
        raise ValueError(f"covariates disagree on the number of observations: {sizes}")
    blocks: list[_Block] = []
    for name, fd in covariates.items():
        if fd.n_vars != 1:
            raise ValueError(f"functional covariate {name!r} carries a variable axis")
        if isinstance(beta_basis, Basis):
            target = beta_basis
        elif beta_basis is not None and name in beta_basis:
            target = beta_basis[name]
        else:
            target = BSpline(domain=fd.domain, n_basis=n_basis, order=min(4, n_basis))
        block_columns = tuple(f"{name}[{k}]" for k in range(target.n_basis))
        pieces.append(np.asarray(to_numpy(inprod(fd, target)), dtype=np.float64))
        columns.extend(block_columns)
        blocks.append(_Block(name=name, basis=target, columns=block_columns))
    if len(set(columns)) != len(columns):
        raise ValueError(f"column names must be unique, got {columns}")
    matrix = np.concatenate(pieces, axis=1)
    if not math.isfinite(float(np.abs(matrix).sum())):
        raise ValueError("the design matrix contains non-finite values")
    return FunctionalDesign(matrix=matrix, columns=tuple(columns), _blocks=tuple(blocks))
