# fdatools engineering conventions (binding for every contributor and agent)

Read `CLAUDE.md`, `SPEC.md`, `WORKFLOW.md` first. This file adds the concrete mechanics.

## Environment
- `uv` only. Venv at `.venv` (Python 3.12). Run tools as `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/mypy`.
- Local R 4.6.1 with `fda` 6.3.0 + `jsonlite` at `Rscript` (Homebrew). `rpy2` works in the venv. Docker image also exists.
- Lint gate: `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy --strict src/fdatools`

## Layout
- `src/fdatools/<module>.py` public modules per SPEC §2. Private helpers → `src/fdatools/_internal/`.
- `fdatools/_backend.py` exposes `xp = array_namespace(*arrays)` (array-api-compat) and `asarray`, `to_numpy`. Core math uses `xp.*` only; never `import numpy` in core modules except inside `_backend.py`, `_linalg.py` and typing blocks (`if TYPE_CHECKING:`).
- Public objects: `@dataclass(frozen=True)`. Methods return new objects.

## Golden files (`tests/golden/<module>.json`) — schema
```json
{
  "meta": {"module": "basis", "r_version": "R version 4.6.1", "fda_version": "6.3.0",
           "generated": "2026-09-06", "seed": 20260906, "rtol": 1e-8},
  "cases": [
    {"name": "bspline_eval_k4_n10_d0",
     "r_call": "eval.basis(t, create.bspline.basis(c(0,1), 10, 4), 0)",
     "input": {"domain": [0, 1], "n_basis": 10, "order": 4, "t": [...]} ,
     "output": {"values": [[...], ...]},
     "rtol": 1e-5}
  ]
}
```
- Matrices are JSON row-major nested lists (`jsonlite::toJSON(x, digits = 17)` — full round-trip double precision, `matrix = "rowmajor"`). Note: `digits = NA` looks like "full precision" but is actually only ~15 significant digits and can silently truncate a value across a float boundary; always use `digits = 17`.
- Vectors are flat lists. Scalars are scalars. `NA`/`NaN` → `null`.
- `"rtol"` on a case is OPTIONAL: a per-case override of `meta.rtol`, for cases whose reference values come from an iterative fit (e.g. `smooth.monotone`, `smooth.pos`) and need a looser tolerance than the module default. Omitted when the module default applies.
- Golden files are generated ONLY by `tools/make_golden.py <module>`; never hand-edited. Never delete.
- Parity tests in `tests/parity/test_<module>.py` load via `tests/parity/conftest.py::golden("<module>")` / `golden_cases("<module>")` and compare with `numpy.testing.assert_allclose(actual, expected, rtol=case_rtol(module, case), atol=1e-12)`, where `case_rtol` honours the per-case `"rtol"` override and falls back to `meta["rtol"]`.

## Testing
- Tests first. `tests/unit/` (pure Python behaviour, hypothesis property tests), `tests/parity/` (golden), `tests/sklearn_compat/`, `tests/torch_backend/` (skip if torch absent; not `tests/torch/`, which would shadow PyTorch as a namespace package when it is not installed).
- Coverage ≥ 90% on touched module.

## Library docs rule (global CLAUDE.md)
Every library call must trace to official docs for the locked version (`uv pip freeze`). Docs cache lives at `~/.claude/docs-cache/<lib>@<major.minor>.md`; create it (distilled summary from official docs) if missing before using an API.

## Commits
Conventional commits, one per green task: `feat(basis): ...`, `test(parity): ...`, `chore: ...`. No attribution trailers.
