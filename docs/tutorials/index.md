# Tutorials

Seven worked analyses, each from raw data to a result you can plot. They follow
the examples of Ramsay, Hooker and Graves, *Functional Data Analysis with R
and MATLAB* (2009), so you can compare with the book and with R's `fda`.

Read the [Quickstart](../quickstart.md) first if you have never used fdatools.
Every code block runs as written, top to bottom, and ends with an
**R equivalent** section for readers who know R's `fda`.

| Tutorial | You will learn | Data |
|---|---|---|
| [Smoothing](smoothing.md) | `smooth()`: automatic basis, λ by GCV, `df=`, the harmonic accelerator, monotone curves, `SmoothResult` | growth, Canadian weather |
| [Functional PCA and CCA](fpca.md) | `FPCA`: harmonics, scores, variance shares, varimax rotation; `FCCA` | Canadian weather |
| [Registration](registration.md) | landmark and continuous `register()`, warping functions, amplitude and phase with `.decompose()` | growth |
| [Functional regression](regression.md) | `fregress()` with scalar and functional responses, formulas, `predict` / `stderr` / `cv`, the permutation F test | Canadian weather |
| [Dynamics](dynamics.md) | principal differential analysis with `PDA`, `solve()`, the stability diagram, phase-plane plots | lip movement, growth |
| [Econometrics and interop](econometrics.md) | `fdatools.interop`: NumPy, SciPy and scikit-fda conversions; panels for linearmodels; scalar-on-function regression with statsmodels (robust bands, GLMs, fixed effects) | growth, simulated panel |
| [Machine learning](machine-learning.md) | scikit-learn `Pipeline` and `GridSearchCV`; PyTorch `BasisLayer`, `SmoothingLayer`, `FDataDataset` | growth |

## Data

The growth, gait and pinch datasets ship inside fdatools. The others (Canadian
weather, lip, and the rest of the book's datasets) are downloaded once on
first use, checked against a SHA-256 checksum, and cached in
`~/.cache/fdatools`. Set the environment variable `FDATOOLS_DATA_DIR` to use
another folder, for example one you copied to a machine without internet.

## Plots

The tutorials draw with matplotlib (the `plot` extra). The machine learning
tutorial also needs PyTorch (the `torch` extra):
`uv add "fdatools[plot,torch]"`.
