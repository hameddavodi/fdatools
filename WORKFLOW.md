# WORKFLOW.md — fdatools Autonomous Build Plan

Execute phases in order. Each phase ends with a **GATE** — a shell command that must exit 0.
Check off tasks here as you complete them. Track details in `PROGRESS.md`.

---

## Phase 0 — Repo & Environment

- [x] `src/fdatools` layout, `pyproject.toml` (hatchling, extras: `torch`, `pandas`, `plot`, `dev`)
- [x] `Dockerfile` + `docker-compose.yml`: Python 3.11, R 4.x, R `fda` 6.3.0 pinned, `rpy2`
- [x] Tooling config: ruff (strict), mypy (strict), pytest + pytest-cov + hypothesis + pytest-benchmark
- [x] CI (GitHub Actions): lint → type → test → parity → bench (regression alert) → docs → build
- [x] `PROGRESS.md` created from template
- [x] `tools/make_golden.py`: runs R via rpy2 on seeded inputs → `tests/golden/*.json` (input, output, R version, tolerance)

**GATE 0:** `docker compose run dev pytest --collect-only && ruff check .`

---

## Phase 1 — Backend + Basis + Core

Order matters: `_backend` → `basis` → `core`.

- [x] `_backend.py`: array-api dispatch (numpy default; torch/jax if input is tensor). Property test: same result numpy vs torch, rtol 1e-10
- [x] `_linalg.py`: banded Cholesky, sparse penalty assembly, LRU cache for basis Gram matrices
- [x] `basis.py`: `Basis` ABC + `BSpline, Fourier, Monomial, Exponential, Power, Constant, Polygonal`
  - callable eval with `deriv=`, `.penalty(op)`, `.gram()`, `b1 * b2`
  - Golden: eval + penalty matrices vs R `eval.basis` / `getbasispenalty` for 25 parameter combos
- [x] `core.py`: `FData` (callable, arithmetic, `@` inner product, `.derivative()`, `.mean()`, `.cov()`, indexing), `BiFData`, `LDO`
  - Golden: vs R `eval.fd`, `inprod`, `mean.fd`, `var.fd`, `deriv.fd`
- [x] Plotting mixin (matplotlib): `.plot()`, `.plot_fit()` — smoke-tested, image-hash tested

**GATE 1:** `pytest tests/unit tests/parity/test_basis.py tests/parity/test_core.py --cov=fdatools --cov-fail-under=90`

---

## Phase 2 — Smoothing + Stats + Datasets

- [x] `smoothing.py`: single `smooth()` (auto basis, λ via GCV golden-section; `constraint=` monotone/positive/morph; per-curve irregular t; weights) → `SmoothResult`
  - Golden: vs `smooth.basis`, `smooth.basisPar`, `Data2fd`, `smooth.monotone`, `smooth.pos` on CanadianWeather + growth
  - GCV grid must reuse one factorization (see CLAUDE.md perf rules); benchmark ≥10× R timing recorded in `benchmarks/baseline_r.json`
- [x] `stats.py`: `cov`, `cor`, functional boxplot (fbplot), depth, `f_test` (Fperm), `t_test` (tperm)
- [x] `datasets.py`: all 11 loaders, lazy-download + local cache + checksum; ship growth/gait/pinch in-package
- [x] I/O: `from_pandas`, `to_pandas`, `to_xarray`, `read_rds`

**GATE 2:** `pytest tests/unit tests/parity/test_smoothing.py --cov=fdatools --cov-fail-under=90 && pytest benchmarks -k smooth --benchmark-only`
(2026-09-27: `tests/unit` added. `--cov=fdatools` measures the whole package, and one parity file alone covers only 31% of it. The 90% bar is unchanged.)

---

## Phase 3 — Decomposition + Regression + ML

- [x] `decomposition.py`: `FPCA` (with smoothing penalty, varimax rotation), `FCCA` — sklearn estimator API
  - Golden: eigenvalues, harmonics, scores vs `pca.fd`/`cca.fd` (sign-align eigenvectors before comparing)
- [x] `regression.py`: `fregress()` — scalar↔functional auto-dispatch, formula interface, `.predict/.stderr/.cv`
  - Golden: vs `fRegress` on the 3 book case studies
- [x] sklearn compliance: `sklearn.utils.estimator_checks.check_estimator` passes for `Smoother`, `FPCA`, `FRegress` (and `Registrator`; no exemptions)
- [x] `nn.py` (extra `[torch]`): `BasisLayer`, `SmoothingLayer`, `FDataDataset`; gradcheck on all layers; GPU test skipped-if-unavailable

**GATE 3:** `pytest tests/parity tests/sklearn_compat tests/torch_backend tests/unit --cov=fdatools --cov-fail-under=90`
(2026-09-27: `tests/unit` added, as for GATE 2. Without it the package coverage is 76%. With it: 98.6%. The 90% bar is unchanged.)

---

## Phase 4 — Registration + Dynamics (hardest — budget 2× time)

- [x] `registration.py`: `register()` (continuous, Newton on warping coefs), `landmarks=` mode, `.decompose()` (AmpPhaseDecomp)
  - Golden: growth-data registration vs `register.fd` (`rtol=1e-5`); property test: warps strictly monotone
  - [x] Torch backend: autodiff path benchmarked vs numpy Newton (2026-09-27: `fdatools._internal.registration_torch`, `benchmarks/test_bench_registration.py`; torch about 4x slower)
- [x] `dynamics.py`: `PDA`, `phase_plane()`, ODE solve via scipy `solve_ivp`
  - Golden: vs `pda.fd` on lip/handwriting data

**GATE 4:** `pytest tests/parity/test_registration.py tests/parity/test_dynamics.py`

---

## Phase 5 — Acceptance, Docs, Release

- [ ] `notebooks/book_figures.ipynb`: reproduce all 76 figures from Ramsay-Hooker-Graves 2009; CI executes it with `nbmake`
  - 2026-09-27: 70 figures (ch 1, 3-11) built from `notebooks/book/ch*.py` by `tools/build_book_notebook.py`; nbmake passes. Open: 6 figures short of 76, numbering not checked against the printed book. See PROGRESS.md
- [x] Docs: mkdocs-material + mkdocstrings — quickstart, R-migration table (from SPEC.md §3-4), 6 tutorials, full API reference. `mkdocs build --strict` clean
- [x] `PARITY_REPORT.md`: auto-generated table — every public symbol, R counterpart, max abs/rel error, status
- [x] Packaging: wheels via `python -m build`, `twine check dist/*` clean, `pip install fdatools` smoke test in clean venv, py3.10–3.13 matrix
- [x] `CHANGELOG.md`, `LICENSE` (BSD-3), `CITATION.cff`, README with badges
- [x] Version `1.0.0` tagged. Publish command prepared but **not executed** (`twine upload` is the only human step)
  - 2026-09-27: GATE 5 green, annotated tag `v1.0.0` (local). Publish commands in PROGRESS.md "Human steps"
- [x] Version `1.1.0` (first release named `fdatools`): GATE 5 green, tag `v1.1.0`, GitHub release with wheel + sdist. PyPI upload is the human step

**GATE 5 (final):**
```bash
pytest && \
pytest tests/parity && \
mypy --strict src/fdatools && \
ruff check . && \
mkdocs build --strict && \
python -m build && twine check dist/* && \
pytest --nbmake notebooks/book_figures.ipynb
```

---

## Enhancement round (2026-09-28) — API extension beyond the v1 SPEC

Seven parallel builders, merged on `integration/enh`. SPEC.md was extended on purpose (§2, §4.1, §4.3-§4.9).

- [x] `sparse.py`: PACE for sparse longitudinal data (`sparse_mean`, `sparse_cov`, `SparseCov`, `PACE`) — golden `pace.json`
- [x] `density.py`: `fit_density`, `fit_intensity` — golden `density.json`
- [x] `regression.py`: `linmod` / `LinmodResult`; torch pass-through for `fregress` — golden `linmod.json`
- [x] `registration.py`: multivariate `register` (`var_weights`), `RegistrationResult.apply` — golden `registration_multivariate.json`
- [x] `dynamics.py` / `smoothing.py`: PDA forcing, `PDA.stability` / `PDAStability`; exact monotone/positive derivatives of any order — golden `pda_forcing.json`, `monotone_derivs.json`
- [x] `stats.py`: `confidence_band` / `ConfidenceBand`, `plot_beta`, `cycleplot`, `plot_scores` — golden `bands.json`
- [x] `profiling.py`: generalized profiling for ODE parameters (CSTR family) — golden `profiling.json`
- [x] Integration: SPEC, `__init__` exports, API pages + nav, R migration rows, `tools/parity_report.py` rules, `PARITY_REPORT.md`, CHANGELOG, full gate green
- [ ] Cleanup wave: owner references (after the user names the new owner), independent review, re-run GATE 5, move the local `v1.0.0` tag

---

## Recovery rules

- Context lost / new session → read `PROGRESS.md`, find first unchecked box above, continue.
- Parity mismatch you can't resolve in 3 attempts → log numeric diff + hypothesis in `PROGRESS.md`, tag task `BLOCKED`, continue with next independent task, return after the phase.
- Any change to public API requires updating SPEC.md + PARITY_REPORT in the same commit.

## PROGRESS.md template

```markdown
# PROGRESS
## Status: Phase X, task Y
## Done
- [x] ...
## Decisions
- <date> chose banded Cholesky over sparse LU because ...
## Failed approaches (do not retry)
- <date> Brent for GCV min — non-smooth objective; use golden-section.
## Blocked
- ...
```
