"""The tutorials under docs/tutorials/ must run as written.

Each ```python block of a tutorial is executed in order in one shared
namespace, so later blocks may use names from earlier ones (the same approach
as ``test_docs_examples.py``).  The first lines of a block may carry markers:

``# requires: <module>``
    Skip the block when the optional module (``torch``, say) is not installed.
``# requires-data: <name>[, <name>...]``
    Skip the block when the dataset (the suffix of ``fdatools.datasets.load_<name>``)
    cannot be loaded offline -- ``FDATOOLS_DATA_DIR`` unset or the files missing.
    Tests never touch the network: the download hook is replaced by one that
    raises.

Plots use the non-interactive ``Agg`` backend and are closed after each page.
"""

from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

import fdatools.datasets

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = REPO_ROOT / "docs" / "tutorials"
TUTORIALS = [
    "smoothing.md",
    "fpca.md",
    "registration.md",
    "regression.md",
    "dynamics.md",
    "machine-learning.md",
    "econometrics.md",
]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
REQUIRES = re.compile(r"^# requires: (\w+)\s*$")
REQUIRES_DATA = re.compile(r"^# requires-data: ([\w, ]+?)\s*$")


def python_blocks(page: str) -> list[str]:
    return BLOCK.findall((TUTORIAL_DIR / page).read_text(encoding="utf-8"))


def markers(block: str) -> tuple[list[str], list[str]]:
    """Return ``(modules, datasets)`` named by the block's leading marker lines."""
    modules: list[str] = []
    datasets: list[str] = []
    for line in block.splitlines():
        if match := REQUIRES.match(line):
            modules.append(match.group(1))
        elif match := REQUIRES_DATA.match(line):
            datasets.extend(name.strip() for name in match.group(1).split(",") if name.strip())
        else:
            break
    return modules, datasets


def _refuse_download(url: str, dest: Path) -> None:
    raise ConnectionRefusedError(f"tests never download ({url} -> {dest})")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Forbid dataset downloads and force the Agg matplotlib backend."""
    import matplotlib

    monkeypatch.setattr(fdatools.datasets, "_download_file", _refuse_download)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    matplotlib.use("Agg")
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def dataset_available(name: str) -> bool:
    """Whether ``load_<name>()`` succeeds without the network (needs ``offline``)."""
    loader = getattr(fdatools.datasets, f"load_{name}")
    try:
        loader()
    except (OSError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------- #
# the parsing helpers
# --------------------------------------------------------------------------- #


def test_block_extraction() -> None:
    text = "intro\n```python\nx = 1\n```\n\n```bash\nls\n```\n```python\ny = x\n```\n"
    assert BLOCK.findall(text) == ["x = 1\n", "y = x\n"]


def test_marker_parsing() -> None:
    block = "# requires: torch\n# requires-data: lip, canadian_weather\nx = 1\n# requires: jax\n"
    assert markers(block) == (["torch"], ["lip", "canadian_weather"])
    assert markers("x = 1\n") == ([], [])


def test_offline_guard_refuses_downloads(
    offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FDATOOLS_DATA_DIR", str(tmp_path))
    assert not dataset_available("lip")
    assert dataset_available("growth")  # ships inside the package
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# structure of the tutorials
# --------------------------------------------------------------------------- #


def test_tutorial_list_matches_directory() -> None:
    on_disk = {path.name for path in TUTORIAL_DIR.glob("*.md")} - {"index.md"}
    assert on_disk == set(TUTORIALS)


@pytest.mark.parametrize("page", TUTORIALS)
def test_tutorial_structure(page: str) -> None:
    text = (TUTORIAL_DIR / page).read_text(encoding="utf-8")
    assert text.startswith("# "), f"{page} must start with a title"
    assert "## R equivalent" in text, f"{page} needs an 'R equivalent' section"
    assert "import fdatools as fdt" in text
    assert len(python_blocks(page)) >= 4


@pytest.mark.parametrize("page", TUTORIALS)
def test_tutorial_markers_are_valid(page: str) -> None:
    for block in python_blocks(page):
        modules, datasets = markers(block)
        for name in datasets:
            assert hasattr(fdatools.datasets, f"load_{name}"), f"{page}: unknown dataset {name!r}"
        for module in modules:
            assert module.isidentifier()


def test_tutorials_are_linked() -> None:
    index = (TUTORIAL_DIR / "index.md").read_text(encoding="utf-8")
    nav = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for page in TUTORIALS:
        assert f"({page})" in index, f"tutorials/index.md does not link {page}"
        assert f"tutorials/{page}" in nav, f"mkdocs.yml nav does not list {page}"


# --------------------------------------------------------------------------- #
# the tutorials run
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("page", TUTORIALS)
def test_tutorial_runs(page: str, offline: None) -> None:
    namespace: dict[str, object] = {"__name__": "__tutorial__"}
    available: dict[str, bool] = {}
    for number, block in enumerate(python_blocks(page), start=1):
        modules, datasets = markers(block)
        if any(importlib.util.find_spec(module) is None for module in modules):
            continue
        for name in datasets:
            if name not in available:
                available[name] = dataset_available(name)
        if not all(available[name] for name in datasets):
            continue
        exec(compile(block, f"{page} block {number}", "exec"), namespace)


def test_tutorial_data_available_when_configured(offline: None) -> None:
    """With ``FDATOOLS_DATA_DIR`` set, every dataset a tutorial needs must load offline.

    This keeps the data-marked blocks from being skipped silently on a machine
    that is meant to run them all.
    """
    if not os.environ.get("FDATOOLS_DATA_DIR"):
        pytest.skip("FDATOOLS_DATA_DIR is not set")
    needed = {
        name for page in TUTORIALS for block in python_blocks(page) for name in markers(block)[1]
    }
    missing = sorted(name for name in needed if not dataset_available(name))
    assert not missing, f"datasets missing from FDATOOLS_DATA_DIR: {missing}"
