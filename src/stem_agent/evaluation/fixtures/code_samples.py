"""Benchmark corpus — 20 Python code samples with ground-truth labels.

Distribution:
  - 5 logic bugs
  - 4 security vulnerabilities
  - 4 code smells / maintainability
  - 2 performance issues
  - 5 clean code (true negatives — including adversarial "looks buggy but isn't")

Each sample has ground-truth issue categories and a clean flag.
The clean samples are deliberately non-trivial to test agent judgment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkSample:
    """A single benchmark code sample with ground-truth annotations."""

    sample_id: str
    description: str
    code: str
    issue_categories: list[str] = field(default_factory=list)
    is_clean: bool = False


# ---------------------------------------------------------------------------
# Logic Bugs (5 samples)
# ---------------------------------------------------------------------------

LOGIC_01_OFF_BY_ONE = BenchmarkSample(
    sample_id="logic_01",
    description="Off-by-one error in binary search",
    code='''\
def binary_search(arr: list[int], target: int) -> int:
    """Return index of target in sorted array, or -1 if not found."""
    low, high = 0, len(arr)  # Bug: should be len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
    issue_categories=["logic"],
)

LOGIC_02_WRONG_OPERATOR = BenchmarkSample(
    sample_id="logic_02",
    description="Wrong comparison operator causes incorrect filtering",
    code='''\
def filter_adults(people: list[dict]) -> list[dict]:
    """Return people who are 18 or older."""
    return [p for p in people if p["age"] > 18]  # Bug: should be >= 18
''',
    issue_categories=["logic"],
)

LOGIC_03_MISSING_NONE_CHECK = BenchmarkSample(
    sample_id="logic_03",
    description="Missing None check before attribute access",
    code='''\
def get_user_email(users: dict[str, dict], username: str) -> str:
    """Look up a user's email by username."""
    user = users.get(username)
    return user["email"]  # Bug: user could be None
''',
    issue_categories=["logic"],
)

LOGIC_04_INTEGER_OVERFLOW = BenchmarkSample(
    sample_id="logic_04",
    description="Integer overflow in average calculation",
    code='''\
def average(a: int, b: int) -> int:
    """Return the average of two integers, used for midpoint calculation."""
    return (a + b) // 2  # Bug: a + b can overflow in languages with fixed ints
    # In Python this is technically safe due to arbitrary precision,
    # but it's a well-known anti-pattern worth flagging for portability.
    # Correct: a + (b - a) // 2
''',
    issue_categories=["logic"],
)

LOGIC_05_BOOLEAN_LOGIC = BenchmarkSample(
    sample_id="logic_05",
    description="Incorrect boolean logic in access control",
    code='''\
def can_access_resource(user: dict) -> bool:
    """Check if user can access the resource.
    Requirements: must be active AND (admin OR has explicit permission).
    """
    is_active = user.get("active", False)
    is_admin = user.get("role") == "admin"
    has_permission = user.get("permissions", {}).get("resource_access", False)
    # Bug: wrong precedence — this allows inactive admins
    return is_active or is_admin and has_permission
''',
    issue_categories=["logic"],
)

# ---------------------------------------------------------------------------
# Security Vulnerabilities (4 samples)
# ---------------------------------------------------------------------------

SECURITY_01_SQL_INJECTION = BenchmarkSample(
    sample_id="security_01",
    description="SQL injection via f-string formatting",
    code='''\
import sqlite3

def get_user(db: sqlite3.Connection, username: str) -> dict | None:
    """Fetch user record by username."""
    cursor = db.execute(
        f"SELECT * FROM users WHERE username = '{username}'"
    )
    row = cursor.fetchone()
    if row:
        return {"id": row[0], "username": row[1], "email": row[2]}
    return None
''',
    issue_categories=["security"],
)

SECURITY_02_PATH_TRAVERSAL = BenchmarkSample(
    sample_id="security_02",
    description="Path traversal in file serving endpoint",
    code='''\
import os

def read_user_file(base_dir: str, filename: str) -> str:
    """Read a file from the user's upload directory."""
    # Bug: filename could be "../../../etc/passwd"
    filepath = os.path.join(base_dir, filename)
    with open(filepath) as f:
        return f.read()
''',
    issue_categories=["security"],
)

SECURITY_03_HARDCODED_CREDENTIALS = BenchmarkSample(
    sample_id="security_03",
    description="Hardcoded API key and database credentials",
    code='''\
import requests

API_KEY = "sk-proj-a8f3k29d1m4n5p6q7r8s9t0u"
DB_PASSWORD = "super_secret_production_pw_2024"

def fetch_data(endpoint: str) -> dict:
    """Fetch data from the external API."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(endpoint, headers=headers)
    return response.json()

def get_db_connection():
    """Connect to the production database."""
    import psycopg2
    return psycopg2.connect(
        host="db.production.internal",
        password=DB_PASSWORD,
        user="admin",
    )
''',
    issue_categories=["security"],
)

SECURITY_04_UNSAFE_EVAL = BenchmarkSample(
    sample_id="security_04",
    description="eval() with unsanitized user input",
    code='''\
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression provided by the user."""
    # Bug: user can inject arbitrary Python code
    return eval(expression)

def process_config(config_str: str) -> dict:
    """Parse a config string into a dictionary."""
    # Bug: exec with user input
    namespace = {}
    exec(config_str, namespace)
    return namespace
''',
    issue_categories=["security"],
)

# ---------------------------------------------------------------------------
# Code Smells / Maintainability (4 samples)
# ---------------------------------------------------------------------------

SMELL_01_LONG_FUNCTION = BenchmarkSample(
    sample_id="smell_01",
    description="Function with too many responsibilities and deep nesting",
    code='''\
def process_order(order: dict, db, mailer, logger) -> dict:
    """Process an incoming order — does EVERYTHING in one function."""
    result = {"status": "pending"}
    if order.get("items"):
        total = 0
        for item in order["items"]:
            if item.get("quantity", 0) > 0:
                if item.get("price") is not None:
                    if item["price"] > 0:
                        subtotal = item["price"] * item["quantity"]
                        if order.get("discount_code"):
                            discount = db.get_discount(order["discount_code"])
                            if discount:
                                if discount["type"] == "percent":
                                    subtotal *= (1 - discount["value"] / 100)
                                elif discount["type"] == "fixed":
                                    subtotal -= discount["value"]
                                    if subtotal < 0:
                                        subtotal = 0
                        total += subtotal
                        logger.info(f"Item processed: {item['name']}")
                    else:
                        logger.warning(f"Negative price for {item.get('name')}")
                        result["status"] = "error"
                        result["message"] = "Invalid price"
                        return result
        result["total"] = total
        if total > 0:
            payment = db.charge_payment(order["payment_method"], total)
            if payment["success"]:
                db.save_order(order, total)
                mailer.send_confirmation(order["email"], order, total)
                result["status"] = "completed"
                result["payment_id"] = payment["id"]
            else:
                result["status"] = "payment_failed"
                result["message"] = payment.get("error", "Unknown error")
        else:
            result["status"] = "empty"
    else:
        result["status"] = "no_items"
    return result
''',
    issue_categories=["structure"],
)

SMELL_02_DEEP_NESTING = BenchmarkSample(
    sample_id="smell_02",
    description="Deeply nested conditionals with magic numbers",
    code='''\
def classify_risk(score: float, age: int, history: list) -> str:
    """Classify customer risk level."""
    if score > 0:
        if age >= 18:
            if len(history) > 0:
                if score > 750:
                    if age < 65:
                        return "low"
                    else:
                        if len(history) > 10:
                            return "low"
                        else:
                            return "medium"
                elif score > 600:
                    if len(history) > 5:
                        return "medium"
                    else:
                        return "high"
                else:
                    return "high"
            else:
                return "high"
        else:
            return "rejected"
    else:
        return "invalid"
''',
    issue_categories=["structure"],
)

SMELL_03_DEAD_CODE = BenchmarkSample(
    sample_id="smell_03",
    description="Dead code after unconditional return and unused variables",
    code='''\
def compute_stats(data: list[float]) -> dict:
    """Compute statistics for a dataset."""
    if not data:
        return {"mean": 0, "std": 0, "count": 0}

    total = sum(data)
    count = len(data)
    mean = total / count

    variance = sum((x - mean) ** 2 for x in data) / count
    std = variance ** 0.5

    return {"mean": mean, "std": std, "count": count}

    # Dead code below — unreachable
    median = sorted(data)[len(data) // 2]
    mode = max(set(data), key=data.count)
    return {"mean": mean, "std": std, "median": median, "mode": mode}
''',
    issue_categories=["structure"],
)

SMELL_04_MAGIC_NUMBERS = BenchmarkSample(
    sample_id="smell_04",
    description="Magic numbers scattered throughout business logic",
    code='''\
def calculate_shipping(weight: float, distance: float, express: bool) -> float:
    """Calculate shipping cost."""
    base = weight * 2.35
    if distance > 500:
        base *= 1.45
    elif distance > 100:
        base *= 1.15
    if express:
        base *= 1.75
    if base < 4.99:
        base = 4.99
    if base > 299.99:
        base = 299.99
    tax = base * 0.08875
    return round(base + tax, 2)
''',
    issue_categories=["structure"],
)

# ---------------------------------------------------------------------------
# Performance Issues (2 samples)
# ---------------------------------------------------------------------------

PERFORMANCE_01_N_PLUS_ONE = BenchmarkSample(
    sample_id="perf_01",
    description="N+1 query pattern in loop",
    code='''\
def get_order_summaries(db, user_id: int) -> list[dict]:
    """Get summary of all orders for a user — classic N+1 problem."""
    orders = db.query("SELECT id, created_at FROM orders WHERE user_id = %s", user_id)
    summaries = []
    for order in orders:
        # Bug: one query per order — should be a JOIN or batch query
        items = db.query(
            "SELECT name, quantity, price FROM order_items WHERE order_id = %s",
            order["id"],
        )
        total = sum(i["price"] * i["quantity"] for i in items)
        summaries.append({
            "order_id": order["id"],
            "date": order["created_at"],
            "total": total,
            "item_count": len(items),
        })
    return summaries
''',
    issue_categories=["performance"],
)

PERFORMANCE_02_UNNECESSARY_COPY = BenchmarkSample(
    sample_id="perf_02",
    description="Unnecessary list copies in hot path",
    code='''\
def find_common_elements(lists: list[list[int]]) -> list[int]:
    """Find elements common to all input lists."""
    if not lists:
        return []
    # Bug: creates a full copy on every iteration for no reason
    result = list(lists[0])
    for lst in lists[1:]:
        temp = list(result)  # Unnecessary copy
        result = []
        lookup = list(lst)  # Should use set for O(1) lookup
        for item in temp:
            if item in lookup:  # O(n) lookup instead of O(1) with set
                result.append(item)
    return result
''',
    issue_categories=["performance"],
)

# ---------------------------------------------------------------------------
# Clean Code — True Negatives (5 samples)
# These are non-trivial, correct code. The agent should NOT flag them.
# Some deliberately *look* suspicious but are correct.
# ---------------------------------------------------------------------------

CLEAN_01_WALRUS_OPERATOR = BenchmarkSample(
    sample_id="clean_01",
    description="Walrus operator usage that looks like a bug but is correct",
    code='''\
def find_first_long_line(lines: list[str], threshold: int = 80) -> str | None:
    """Find the first line exceeding the threshold length.

    Uses the walrus operator (:=) for concise filtered-first semantics.
    This looks like an assignment-in-condition bug but is idiomatic Python 3.8+.
    """
    for line in lines:
        if (stripped := line.strip()) and len(stripped) > threshold:
            return stripped
    return None


def process_chunks(data: bytes, chunk_size: int = 4096) -> list[bytes]:
    """Process data in chunks using walrus operator in while loop."""
    chunks = []
    offset = 0
    while (chunk := data[offset:offset + chunk_size]):
        chunks.append(chunk)
        offset += chunk_size
    return chunks
''',
    is_clean=True,
)

CLEAN_02_COMPLEX_REGEX = BenchmarkSample(
    sample_id="clean_02",
    description="Complex but correct regex with documentation",
    code='''\
import re

# RFC 5322 simplified email validation pattern.
# Intentionally permissive — validates syntax, not deliverability.
# The apparent complexity is warranted by the RFC's actual grammar.
EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
)


def validate_email(email: str) -> bool:
    """Validate email format against a simplified RFC 5322 pattern.

    Returns True if the email has valid syntax. Does not check
    whether the domain exists or the mailbox is deliverable.
    """
    if not email or len(email) > 254:
        return False
    return EMAIL_PATTERN.match(email) is not None


def extract_emails(text: str) -> list[str]:
    """Extract all valid email addresses from a block of text."""
    candidates = re.findall(r"[\\w.+-]+@[\\w-]+\\.[\\w.-]+", text)
    return [c for c in candidates if validate_email(c)]
''',
    is_clean=True,
)

CLEAN_03_SAFE_EVAL = BenchmarkSample(
    sample_id="clean_03",
    description="eval() used safely with restricted builtins",
    code='''\
import ast
import operator

# Whitelist of safe operations for the expression evaluator
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_expr(expr: str) -> float:
    """Safely evaluate a mathematical expression.

    Uses ast.parse to build an AST and walks it manually —
    NO use of eval(). Only arithmetic operations are permitted.
    Raises ValueError on anything that isn't pure arithmetic.
    """
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.expr) -> float:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")
''',
    is_clean=True,
)

CLEAN_04_EXCEPTION_HANDLING = BenchmarkSample(
    sample_id="clean_04",
    description="Broad exception handling that is actually appropriate",
    code='''\
import logging
from typing import Any

logger = logging.getLogger(__name__)


def resilient_parse(raw_data: list[str]) -> list[dict[str, Any]]:
    """Parse raw data strings, skipping malformed entries.

    This function is intentionally lenient: it's used at the boundary
    of an ingestion pipeline where partial results are better than
    total failure. The broad except is deliberate — we log and skip
    rather than crash the entire pipeline.
    """
    results = []
    for i, entry in enumerate(raw_data):
        try:
            parts = entry.split("|")
            record = {
                "id": int(parts[0]),
                "name": parts[1].strip(),
                "value": float(parts[2]),
            }
            results.append(record)
        except (IndexError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed entry at index %d: %s", i, exc)
            continue
    return results


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide a by b, returning default on zero division."""
    try:
        return a / b
    except ZeroDivisionError:
        return default
''',
    is_clean=True,
)

CLEAN_05_COMPLEX_COMPREHENSION = BenchmarkSample(
    sample_id="clean_05",
    description="Complex but correct nested comprehension with proper structure",
    code='''\
from collections import defaultdict
from typing import TypeAlias

# Type aliases for readability
Graph: TypeAlias = dict[str, list[str]]
Components: TypeAlias = list[set[str]]


def find_connected_components(graph: Graph) -> Components:
    """Find all connected components in an undirected graph using BFS.

    The nested comprehension for building the adjacency set looks complex
    but is standard graph algorithm boilerplate — not a code smell.
    """
    visited: set[str] = set()
    components: Components = []

    # Build bidirectional adjacency for undirected graph
    adjacency: dict[str, set[str]] = defaultdict(set)
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            adjacency[node].add(neighbor)
            adjacency[neighbor].add(node)

    for node in adjacency:
        if node not in visited:
            # BFS to find all nodes in this component
            component: set[str] = set()
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                queue.extend(n for n in adjacency[current] if n not in visited)
            components.append(component)

    return components


def invert_index(documents: dict[str, list[str]]) -> dict[str, set[str]]:
    """Build an inverted index from document_id -> words to word -> document_ids."""
    index: dict[str, set[str]] = defaultdict(set)
    for doc_id, words in documents.items():
        for word in words:
            index[word.lower()].add(doc_id)
    return dict(index)
''',
    is_clean=True,
)


def get_benchmark_corpus() -> list[BenchmarkSample]:
    """Return the complete benchmark corpus of 20 samples."""
    return [
        # Logic bugs (5)
        LOGIC_01_OFF_BY_ONE,
        LOGIC_02_WRONG_OPERATOR,
        LOGIC_03_MISSING_NONE_CHECK,
        LOGIC_04_INTEGER_OVERFLOW,
        LOGIC_05_BOOLEAN_LOGIC,
        # Security vulnerabilities (4)
        SECURITY_01_SQL_INJECTION,
        SECURITY_02_PATH_TRAVERSAL,
        SECURITY_03_HARDCODED_CREDENTIALS,
        SECURITY_04_UNSAFE_EVAL,
        # Code smells / maintainability (4)
        SMELL_01_LONG_FUNCTION,
        SMELL_02_DEEP_NESTING,
        SMELL_03_DEAD_CODE,
        SMELL_04_MAGIC_NUMBERS,
        # Performance issues (2)
        PERFORMANCE_01_N_PLUS_ONE,
        PERFORMANCE_02_UNNECESSARY_COPY,
        # Clean code — true negatives (5)
        CLEAN_01_WALRUS_OPERATOR,
        CLEAN_02_COMPLEX_REGEX,
        CLEAN_03_SAFE_EVAL,
        CLEAN_04_EXCEPTION_HANDLING,
        CLEAN_05_COMPLEX_COMPREHENSION,
    ]
