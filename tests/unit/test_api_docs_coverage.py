"""Every public symbol of fdatools is rendered by some ``docs/api/*.md`` page.

mkdocstrings renders a ``::: fdatools.module`` directive as every name in the
module's ``__all__`` (or only the names of a ``members:`` option), and a
``::: fdatools.module.Name`` directive as that one object.  This test reads those
directives and checks that the union covers:

* every name in ``fdatools.__all__`` (for the ``datasets`` and ``stats`` modules
  listed there: every name in their ``__all__``);
* every name in the ``__all__`` of each public module, including
  ``fdatools.stats``, ``fdatools.datasets``, ``fdatools.io`` and ``fdatools.nn``;
* every public result class, named explicitly below so that dropping one from
  a module's ``__all__`` cannot hide it.

Objects are compared by identity, so a re-exported name (``LDO`` lives in a
private module and is exported by ``fdatools.core``) counts wherever it is shown.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import fdatools

yaml = pytest.importorskip("yaml", reason="the docs toolchain (mkdocs) brings PyYAML")

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "docs" / "api"
DIRECTIVE = re.compile(r"^:::\s+([\w.]+)\s*$")

PUBLIC_MODULES = [
    "fdatools.basis",
    "fdatools.core",
    "fdatools.smoothing",
    "fdatools.decomposition",
    "fdatools.regression",
    "fdatools.registration",
    "fdatools.dynamics",
    "fdatools.stats",
    "fdatools.sparse",
    "fdatools.density",
    "fdatools.profiling",
    "fdatools.datasets",
    "fdatools.io",
    "fdatools.interop",
]
# fdatools.nn imports torch at the top; without the extra it cannot be inspected.
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    PUBLIC_MODULES.append("fdatools.nn")
RESULT_CLASSES = [
    "fdatools.smoothing.SmoothResult",
    "fdatools.regression.FRegressResult",
    "fdatools.regression.FRegressStderr",
    "fdatools.regression.FRegressCV",
    "fdatools.regression.LinmodResult",
    "fdatools.registration.RegistrationResult",
    "fdatools.registration.AmpPhaseDecomposition",
    "fdatools.stats.DepthResult",
    "fdatools.stats.BoxplotResult",
    "fdatools.stats.PermutationTestResult",
    "fdatools.stats.ConfidenceBand",
    "fdatools.dynamics.PDAStability",
    "fdatools.sparse.SparseCov",
    "fdatools.density.DensityResult",
    "fdatools.density.IntensityResult",
    "fdatools.profiling.ProfileResult",
    "fdatools.profiling.InnerFit",
    "fdatools.io.LongData",
    "fdatools.interop.PanelCurves",
    "fdatools.interop.FunctionalDesign",
]


@dataclass
class Directive:
    """One ``::: target`` line and its optional ``members`` list."""

    target: str
    members: list[str] | bool | None = None
    options: dict[str, Any] = field(default_factory=dict)


def parse_directives(text: str) -> list[Directive]:
    """Return the mkdocstrings directives of a Markdown page, with their options."""
    lines = text.splitlines()
    found: list[Directive] = []
    for i, line in enumerate(lines):
        match = DIRECTIVE.match(line)
        if not match:
            continue
        block: list[str] = []
        for following in lines[i + 1 :]:
            if following.strip() and not following.startswith((" ", "\t")):
                break
            block.append(following)
        config = yaml.safe_load("\n".join(block)) if any(s.strip() for s in block) else None
        options = (config or {}).get("options", {}) if isinstance(config, dict) else {}
        members = options.get("members")
        if isinstance(members, list):
            members = [str(name) for name in members]
        elif members is not False:
            members = None
        found.append(Directive(target=match.group(1), members=members, options=options))
    return found


def module_exports(module: ModuleType) -> list[str]:
    """The names mkdocstrings renders for a bare module directive."""
    names = getattr(module, "__all__", None)
    if names is not None:
        return list(names)
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and getattr(value, "__module__", None) == module.__name__
    ]


def needs_torch(path: str) -> bool:
    return path == "fdatools.nn" or path.startswith("fdatools.nn.")


def resolve(path: str) -> Any:
    """Import ``a.b.c`` as a module, or as attribute ``c`` of module ``a.b``."""
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as error:
        if error.name != path:
            raise
    module_path, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_path), name)


def covered_objects(directives: list[Directive]) -> dict[int, str]:
    """Map ``id(obj)`` of every rendered object to the directive that renders it."""
    covered: dict[int, str] = {}
    for directive in directives:
        if needs_torch(directive.target) and not HAS_TORCH:
            continue
        target = resolve(directive.target)
        if isinstance(target, ModuleType):
            if directive.members is False:
                continue
            names = (
                directive.members if isinstance(directive.members, list) else module_exports(target)
            )
            for name in names:
                covered[id(getattr(target, name))] = f"{directive.target}.{name}"
        else:
            covered[id(target)] = directive.target
    return covered


def all_directives() -> list[Directive]:
    return [d for page in sorted(API_DIR.glob("*.md")) for d in parse_directives(page.read_text())]


def required_symbols() -> dict[str, Any]:
    """Every symbol the API reference must render, by dotted name."""
    required: dict[str, Any] = {}
    for name in fdatools.__all__:
        if name == "__version__":
            continue
        value = getattr(fdatools, name)
        if isinstance(value, ModuleType):
            for member in module_exports(value):
                required[f"{value.__name__}.{member}"] = getattr(value, member)
        else:
            required[f"fdatools.{name}"] = value
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for member in module_exports(module):
            required[f"{module_name}.{member}"] = getattr(module, member)
    for path in RESULT_CLASSES:
        required[path] = resolve(path)
    if HAS_TORCH:
        nn = importlib.import_module("fdatools.nn")
        for member in module_exports(nn):
            required[f"fdatools.nn.{member}"] = getattr(nn, member)
    return required


# --------------------------------------------------------------------------- #
# the parser
# --------------------------------------------------------------------------- #


def test_parse_bare_and_restricted_directives() -> None:
    text = (
        "# Page\n\n::: fdatools.core\n\ntext\n\n"
        "::: fdatools.smoothing\n    options:\n      members:\n        - smooth\n\n"
        "::: fdatools.core.FData\n"
    )
    directives = parse_directives(text)
    assert [d.target for d in directives] == [
        "fdatools.core",
        "fdatools.smoothing",
        "fdatools.core.FData",
    ]
    assert directives[0].members is None
    assert directives[1].members == ["smooth"]


def test_members_false_renders_no_members() -> None:
    import fdatools.smoothing

    (directive,) = parse_directives("::: fdatools.smoothing\n    options:\n      members: false\n")
    assert directive.members is False
    assert id(fdatools.smoothing.smooth) not in covered_objects([directive])


def test_restricted_directive_covers_only_its_members() -> None:
    import fdatools.smoothing

    covered = covered_objects([Directive("fdatools.smoothing", members=["smooth"])])
    assert id(fdatools.smoothing.smooth) in covered
    assert id(fdatools.smoothing.Smoother) not in covered


def test_object_directive_and_reexport_by_identity() -> None:
    import fdatools.core

    covered = covered_objects([Directive("fdatools.core.LDO")])
    assert id(fdatools.LDO) in covered
    assert id(fdatools.core.FData) not in covered


def test_resolve_reports_missing_symbols() -> None:
    with pytest.raises(AttributeError):
        resolve("fdatools.core.NoSuchThing")


# --------------------------------------------------------------------------- #
# the reference itself
# --------------------------------------------------------------------------- #


def test_every_directive_resolves() -> None:
    directives = all_directives()
    assert directives, "docs/api/*.md has no mkdocstrings directives"
    for directive in directives:
        if needs_torch(directive.target) and not HAS_TORCH:
            continue
        resolve(directive.target)


def test_every_public_symbol_is_documented() -> None:
    covered = covered_objects(all_directives())
    missing = sorted(path for path, obj in required_symbols().items() if id(obj) not in covered)
    assert not missing, f"not rendered by any docs/api/*.md page: {missing}"


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_public_definitions_are_exported(module_name: str) -> None:
    """A public class or function defined in a module is listed in its ``__all__``.

    Otherwise a bare ``::: module`` directive would silently leave it out.
    """
    module = importlib.import_module(module_name)
    exported = set(module_exports(module))
    defined = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and getattr(value, "__module__", None) == module_name
    }
    assert defined <= exported, f"{module_name} does not export {sorted(defined - exported)}"


def test_every_api_page_is_in_the_nav() -> None:
    nav = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for page in sorted(API_DIR.glob("*.md")):
        assert f"api/{page.name}" in nav, f"mkdocs.yml nav does not list api/{page.name}"
