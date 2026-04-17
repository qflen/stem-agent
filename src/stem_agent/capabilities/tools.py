"""Tool adapters — concrete analysis tools the agent can use during review.

These are lightweight wrappers around Python stdlib capabilities
(ast module, re, etc.) that the agent can optionally incorporate
during specialization.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralMetrics:
    """Structural metrics extracted from Python code via AST analysis."""

    function_count: int
    class_count: int
    max_function_length: int
    max_nesting_depth: int
    total_lines: int
    has_bare_except: bool
    has_eval_or_exec: bool
    import_count: int


def analyze_structure(code: str) -> StructuralMetrics | None:
    """Perform AST-based structural analysis of Python code.

    Returns None if the code cannot be parsed (syntax errors).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    classes: list[ast.ClassDef] = []
    has_bare_except = False
    has_eval_or_exec = False
    imports: list[ast.Import | ast.ImportFrom] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            has_bare_except = True
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("eval", "exec"):
                has_eval_or_exec = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)

    # Calculate max function length
    max_func_len = 0
    for func in functions:
        if hasattr(func, "end_lineno") and func.end_lineno and func.lineno:
            length = func.end_lineno - func.lineno + 1
            max_func_len = max(max_func_len, length)

    # Calculate max nesting depth
    max_depth = _max_nesting_depth(tree)

    return StructuralMetrics(
        function_count=len(functions),
        class_count=len(classes),
        max_function_length=max_func_len,
        max_nesting_depth=max_depth,
        total_lines=len(code.splitlines()),
        has_bare_except=has_bare_except,
        has_eval_or_exec=has_eval_or_exec,
        import_count=len(imports),
    )


def _max_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Recursively calculate maximum nesting depth of control flow."""
    nesting_nodes = (
        ast.If,
        ast.For,
        ast.While,
        ast.With,
        ast.Try,
        ast.AsyncFor,
        ast.AsyncWith,
    )
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, nesting_nodes):
            child_depth = _max_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _max_nesting_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)
    return max_depth


# Common patterns that suggest security issues
SECURITY_PATTERNS: list[tuple[str, str, str]] = [
    (
        r"f['\"].*\{.*\}.*(?:SELECT|INSERT|UPDATE|DELETE)",
        "Possible SQL injection via f-string",
        "security",
    ),
    (
        r"\.format\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)",
        "Possible SQL injection via str.format",
        "security",
    ),
    (
        r"eval\s*\(",
        "Use of eval() — potential code injection",
        "security",
    ),
    (
        r"exec\s*\(",
        "Use of exec() — potential code injection",
        "security",
    ),
    (
        r"(?:password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]",
        "Possible hardcoded credential",
        "security",
    ),
    (
        r"os\.system\s*\(",
        "Use of os.system — prefer subprocess with shell=False",
        "security",
    ),
    (
        r"subprocess\..*shell\s*=\s*True",
        "subprocess with shell=True — potential command injection",
        "security",
    ),
    (
        r"pickle\.loads?\s*\(",
        "Use of pickle — potential insecure deserialization",
        "security",
    ),
]


@dataclass(frozen=True)
class PatternMatch:
    """A regex pattern match found in code."""

    line_number: int
    pattern_description: str
    category: str
    matched_text: str


def scan_patterns(code: str) -> list[PatternMatch]:
    """Scan code for known problematic patterns using regex."""
    matches = []
    lines = code.splitlines()

    for line_num, line in enumerate(lines, start=1):
        for pattern, description, category in SECURITY_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                matches.append(
                    PatternMatch(
                        line_number=line_num,
                        pattern_description=description,
                        category=category,
                        matched_text=line.strip(),
                    )
                )

    return matches
