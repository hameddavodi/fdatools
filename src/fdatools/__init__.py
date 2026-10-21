"""fdatools: functional data analysis for Python.

A clean-room, production-grade Python implementation of the functional data
analysis toolkit described in Ramsay & Silverman, *Functional Data Analysis*.

Examples
--------
>>> import numpy as np
>>> import fdatools as fdt
>>> basis = fdt.BSpline(domain=(0.0, 1.0), n_basis=5)
>>> basis(np.array([0.0, 0.5, 1.0])).shape
(3, 5)
"""

from typing import TYPE_CHECKING, Any

from fdatools import datasets, density, interop, profiling, sparse, stats
from fdatools.basis import (
    Basis,
    BSpline,
    Constant,
    Exponential,
    Fourier,
    Monomial,
    Polygonal,
    Power,
)
from fdatools.core import LDO, BiFData, FData, inprod
from fdatools.decomposition import FCCA, FPCA
from fdatools.density import DensityResult, IntensityResult, fit_density, fit_intensity
from fdatools.dynamics import PDA, PDAStability, phase_plane
from fdatools.io import from_pandas, read_rds, to_pandas, to_xarray
from fdatools.profiling import ODEModel, ProfiledODE, profile_ode
from fdatools.registration import Registrator, landmark_register, register
from fdatools.regression import FRegress, LinmodResult, fregress, linmod
from fdatools.smoothing import Smoother, SmoothResult, smooth
from fdatools.sparse import PACE, SparseCov, sparse_cov, sparse_mean

if TYPE_CHECKING:
    from fdatools import nn as nn

__all__ = [
    "FCCA",
    "FPCA",
    "LDO",
    "PACE",
    "PDA",
    "BSpline",
    "Basis",
    "BiFData",
    "Constant",
    "DensityResult",
    "Exponential",
    "FData",
    "FRegress",
    "Fourier",
    "IntensityResult",
    "LinmodResult",
    "Monomial",
    "ODEModel",
    "PDAStability",
    "Polygonal",
    "Power",
    "ProfiledODE",
    "Registrator",
    "SmoothResult",
    "Smoother",
    "SparseCov",
    "__version__",
    "datasets",
    "density",
    "fit_density",
    "fit_intensity",
    "fregress",
    "from_pandas",
    "inprod",
    "interop",
    "landmark_register",
    "linmod",
    "phase_plane",
    "profile_ode",
    "profiling",
    "read_rds",
    "register",
    "smooth",
    "sparse",
    "sparse_cov",
    "sparse_mean",
    "stats",
    "to_pandas",
    "to_xarray",
]

__version__ = "1.1.0"


def __getattr__(name: str) -> Any:
    """Import the optional :mod:`fdatools.nn` module on first access.

    ``import fdatools`` never imports PyTorch; ``fdatools.nn`` loads it lazily.
    """
    if name == "nn":
        import importlib

        module = importlib.import_module("fdatools.nn")
        globals()["nn"] = module
        return module
    raise AttributeError(f"module 'fdatools' has no attribute {name!r}")
