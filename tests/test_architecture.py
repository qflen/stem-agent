"""Architecture lint; guard the ports/adapters discipline at the import layer.

The hexagonal claim in the writeup is only honest if ``core/`` and ``phases/``
never reach into ``adapters/``. Walking the AST of every module under those
directories and refusing any ``stem_agent.adapters.*`` import keeps the
invariant from regressing silently between commits.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "stem_agent"

GUARDED_PACKAGES: tuple[str, ...] = ("core", "phases")
FORBIDDEN_PREFIX = "stem_agent.adapters"


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _imports_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("package", GUARDED_PACKAGES)
def test_guarded_package_does_not_import_adapters(package: str) -> None:
    """Every module under ``core/`` and ``phases/`` must avoid ``adapters/``."""
    pkg_root = SRC_ROOT / package
    assert pkg_root.is_dir(), f"missing package: {pkg_root}"

    offenders: list[tuple[Path, str]] = []
    for module_path in _python_files(pkg_root):
        for name in _imports_in(module_path):
            if name.startswith(FORBIDDEN_PREFIX):
                offenders.append((module_path, name))

    assert not offenders, f"{package}/ must not import {FORBIDDEN_PREFIX}; offenders: " + "; ".join(
        f"{p.relative_to(REPO_ROOT)} → {n}" for p, n in offenders
    )


def test_lint_walks_subdirectories() -> None:
    """The walk must include nested packages, not just top-level files."""
    discovered = {p.name for p in _python_files(SRC_ROOT / "core")}
    assert "agent.py" in discovered
    assert "state_machine.py" in discovered
    assert "__init__.py" in discovered


def test_positive_control_detects_a_planted_offender(tmp_path: Path) -> None:
    """Sanity-check the lint by giving it a synthetic offender to find."""
    offender = tmp_path / "core_double.py"
    offender.write_text(
        "from stem_agent.adapters.json_storage import JsonStorageAdapter\n_ = JsonStorageAdapter\n"
    )
    names = _imports_in(offender)
    assert any(name.startswith(FORBIDDEN_PREFIX) for name in names)


def test_ports_module_is_imported_by_core() -> None:
    """``core/`` must depend on ``ports/`` so the inversion isn't theoretical."""
    core_imports: set[str] = set()
    for path in _python_files(SRC_ROOT / "core"):
        core_imports.update(_imports_in(path))
    ports_used = {n for n in core_imports if n.startswith("stem_agent.ports")}
    assert ports_used, "core/ should import at least one stem_agent.ports.* module"
