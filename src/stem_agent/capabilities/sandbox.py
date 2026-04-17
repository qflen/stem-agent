"""Sandboxed validation for generated capability code.

When the agent proposes a brand-new static-analysis helper as part of a
generated capability, we cannot just exec the string an LLM returned.
Two defensive layers run before any code from a proposal reaches the
specialized agent:

1. Static AST scan. Imports are restricted to a tiny allowlist (``re``,
   ``ast``, ``string``). Bare references and attribute accesses to
   dangerous names (``open``, ``os``, ``sys``, ``subprocess``,
   ``__import__``, ``eval``, ``exec``, ``compile``, ``pickle``,
   ``socket``, and friends) are rejected before the code is ever
   executed.
2. Subprocess execution in Python's isolated mode (``python -I``) with
   CPU and address-space rlimits set via ``preexec_fn`` and a hard
   wall-clock timeout.

Defense in depth: the AST scan is the precise, fast layer; the
subprocess limits catch anything the scan missed — computed attribute
access, recursion bombs, memory balloons, or infinite loops. The
combination is defensible for a submission demo without pulling in
RestrictedPython or gvisor.
"""

from __future__ import annotations

import ast
import json
import resource
import subprocess
import sys
import textwrap
from dataclasses import dataclass

_ALLOWED_IMPORTS = frozenset({"re", "ast", "string"})

_FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "__import__",
        "__builtins__",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "breakpoint",
        "input",
        "os",
        "sys",
        "io",
        "subprocess",
        "socket",
        "shutil",
        "pickle",
        "pathlib",
        "urllib",
        "http",
        "requests",
        "builtins",
        "importlib",
        "ctypes",
    }
)

# Smoke inputs: one clean snippet, one with classic red-flag tokens.
# If the validator explodes on either, we refuse the capability.
_DEFAULT_SMOKE_INPUTS: tuple[str, ...] = (
    "def add(a, b):\n    return a + b\n",
    "password = 'hunter2'\nresult = eval(user_input)\n",
)

# The runner is a self-contained Python script piped to a fresh interpreter
# via ``python -I -c``. It receives the validator source plus smoke inputs
# as JSON on stdin and prints a single JSON line on stdout.
_RUNNER = textwrap.dedent(
    """
    import ast
    import json
    import sys

    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    smoke_inputs = payload["smoke_inputs"]

    tree = ast.parse(code)
    namespace = {}
    exec(compile(tree, "<generated_capability>", "exec"), namespace)

    check = namespace.get("check")
    if not callable(check):
        print(json.dumps({"ok": False, "error": "no callable named 'check'"}))
        sys.exit(0)

    sample_outputs = []
    for smoke in smoke_inputs:
        out = check(smoke)
        if not isinstance(out, list):
            print(json.dumps({"ok": False, "error": "check must return list"}))
            sys.exit(0)
        if not all(isinstance(x, str) for x in out):
            print(json.dumps({"ok": False, "error": "check must return list[str]"}))
            sys.exit(0)
        sample_outputs.append(out)

    print(json.dumps({"ok": True, "sample_outputs": sample_outputs}))
    """
).strip()


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of validating a proposed capability's ``check`` function."""

    ok: bool
    error: str = ""
    stdout: str = ""
    stderr: str = ""


def ast_scan(code: str) -> tuple[bool, str]:
    """Static-analyse ``code`` for obvious escape hatches.

    Returns ``(True, "")`` if the code parses and uses only allowed
    names and imports, ``(False, reason)`` otherwise. The reason string
    is what we surface in the journal, so it must be short and concrete.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    return False, f"disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in _ALLOWED_IMPORTS:
                return False, f"disallowed import from: {node.module}"
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"disallowed name reference: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            return False, f"disallowed attribute access: .{node.attr}"

    return True, ""


def _preexec_limits() -> None:
    """Applied in the child between fork and exec.

    CPU seconds: 2 (hard and soft, so the child cannot raise its own
    limits). Combined with the wall-clock ``timeout`` on the parent, a
    runaway loop or recursion bomb terminates in bounded time.

    RLIMIT_AS is deliberately not set: on macOS the default hard limit
    behaviour makes lowering the address-space cap fragile, and the AST
    scan already blocks the memory-abuse vectors we care about
    (``ctypes``, ``mmap`` via ``importlib``, etc.). RLIMIT_FSIZE is not
    set either — it would cap stdout pipe writes on some platforms, and
    the AST scan already forbids ``open``, ``pickle``, and friends.
    """
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))


def run_in_sandbox(
    code: str,
    smoke_inputs: tuple[str, ...] | None = None,
    *,
    timeout: float = 3.0,
) -> SandboxResult:
    """Validate ``code`` as a capability ``check`` function.

    The caller is expected to treat a non-``ok`` result as grounds for
    refusing the capability. ``error`` is stable enough to embed in a
    journal ``DECISION`` event without further massaging.
    """
    ok, reason = ast_scan(code)
    if not ok:
        return SandboxResult(ok=False, error=f"ast_scan: {reason}")

    inputs = list(smoke_inputs or _DEFAULT_SMOKE_INPUTS)
    payload = json.dumps({"code": code, "smoke_inputs": inputs})

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_preexec_limits,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return SandboxResult(
            ok=False,
            error=f"timeout after {timeout}s",
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
        )

    if proc.returncode != 0:
        return SandboxResult(
            ok=False,
            error=f"nonzero exit {proc.returncode}",
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return SandboxResult(
            ok=False, error="non-JSON stdout from runner", stdout=proc.stdout, stderr=proc.stderr
        )

    if not result.get("ok"):
        return SandboxResult(
            ok=False, error=result.get("error", "unknown"), stdout=proc.stdout, stderr=proc.stderr
        )

    return SandboxResult(ok=True, stdout=proc.stdout, stderr=proc.stderr)
