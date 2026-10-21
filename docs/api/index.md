# API reference

One page per module of fdatools' public API. Every name in `fdatools.__all__`, every
public function and class of `fdatools.stats`, `fdatools.sparse`, `fdatools.density`,
`fdatools.profiling`, `fdatools.datasets`, `fdatools.io`, `fdatools.interop` and `fdatools.nn`, and every result class is documented here. For worked examples,
see the [tutorials](../tutorials/index.md).

| Module | Contents |
|---|---|
| [Basis systems](basis.md) | `Basis`, `BSpline`, `Fourier`, `Monomial`, `Exponential`, `Power`, `Constant`, `Polygonal` |
| [Core objects](core.md) | `FData`, `BiFData`, `LDO`, `inprod` |
| [Smoothing](smoothing.md) | `smooth`, `SmoothResult`, `Smoother`, `gcv_curve`, `lambda_to_df`, `df_to_lambda` |
| [Decomposition](decomposition.md) | `FPCA`, `FCCA` |
| [Regression](regression.md) | `fregress`, `FRegress`, `FRegressResult`, `FRegressStderr`, `FRegressCV`, `linmod`, `LinmodResult` |
| [Registration](registration.md) | `register`, `landmark_register`, `RegistrationResult`, `AmpPhaseDecomposition`, `Registrator` |
| [Dynamics](dynamics.md) | `PDA`, `PDAStability`, `phase_plane` |
| [Statistics](stats.md) | `cov`, `cor`, `depth`, `boxplot`, `t_test`, `f_test`, `confidence_band`, `ConfidenceBand`, `plot_beta`, `cycleplot`, `plot_scores`, `DepthResult`, `BoxplotResult`, `PermutationTestResult` |
| [Sparse data (PACE)](sparse.md) | `PACE`, `SparseCov`, `sparse_mean`, `sparse_cov` |
| [Density and intensity](density.md) | `fit_density`, `fit_intensity`, `DensityResult`, `IntensityResult` |
| [Profiling (ODE parameters)](profiling.md) | `ODEModel`, `ProfiledODE`, `profile_ode`, `ProfileResult`, `InnerFit`, `Residuals`, `cstr_model`, `cstr_inputs`, `fitzhugh_nagumo_model`, `simpson_rule` |
| [PyTorch layers](nn.md) | `BasisLayer`, `SmoothingLayer`, `FDataDataset` (optional extra) |
| [Datasets](datasets.md) | 14 `load_*` loaders and the dataclasses they return (`Growth`, `CanadianWeather`, `Lip`, ...) |
| [Input and output](io.md) | `from_pandas`, `LongData`, `to_pandas`, `to_xarray`, `read_rds` |
| [Interoperability](interop.md) | `from_grid`, `to_grid`, `to_scipy`, `from_scipy`, `to_skfda`, `from_skfda`, `from_panel`, `to_panel`, `PanelCurves`, `functional_design`, `FunctionalDesign` |
