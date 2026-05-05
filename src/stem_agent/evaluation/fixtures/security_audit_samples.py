"""Security-audit corpus; 8 Python samples for the second-domain demo.

This fixture exists to prove the differentiation pipeline is not wired
only for generic code-quality review. The samples are all security-
flavoured: six contain distinct vulnerabilities, two are clean code
that looks concerning to a naive eye (eval gated behind a whitelist,
subprocess called with an argument list) but is actually safe.

Categories use the same labels as the default corpus so the metric
pipeline composes with no special-casing.
"""

from __future__ import annotations

from stem_agent.evaluation.fixtures.code_samples import BenchmarkSample

SEC_AUDIT_01_SQLI = BenchmarkSample(
    sample_id="sec_audit_01",
    description="Classic SQL injection via f-string",
    code="""\
def fetch_user(conn, user_id: str):
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
""",
    issue_categories=["security"],
)

SEC_AUDIT_02_HARDCODED_SECRET = BenchmarkSample(
    sample_id="sec_audit_02",
    description="Hardcoded AWS-style secret in source",
    code="""\
AWS_SECRET_KEY = "AKIAIOSFODNN7EXAMPLEKEY"

def make_client():
    import boto3
    return boto3.client("s3", aws_secret_access_key=AWS_SECRET_KEY)
""",
    issue_categories=["security"],
)

SEC_AUDIT_03_PATH_TRAVERSAL = BenchmarkSample(
    sample_id="sec_audit_03",
    description="Path traversal via unsanitised user input",
    code="""\
from pathlib import Path

def serve_file(user_supplied: str) -> bytes:
    base = Path("/srv/public")
    target = base / user_supplied
    return target.read_bytes()
""",
    issue_categories=["security"],
)

SEC_AUDIT_04_COMMAND_INJECTION = BenchmarkSample(
    sample_id="sec_audit_04",
    description="Command injection via subprocess shell=True",
    code="""\
import subprocess

def ping(host: str) -> str:
    return subprocess.check_output(f"ping -c 1 {host}", shell=True).decode()
""",
    issue_categories=["security"],
)

SEC_AUDIT_05_INSECURE_DESERIALIZATION = BenchmarkSample(
    sample_id="sec_audit_05",
    description="Pickle deserialisation of untrusted data",
    code="""\
import pickle

def load_session(blob: bytes):
    return pickle.loads(blob)
""",
    issue_categories=["security"],
)

SEC_AUDIT_06_WEAK_CRYPTO = BenchmarkSample(
    sample_id="sec_audit_06",
    description="MD5 used for password hashing",
    code="""\
import hashlib

def hash_password(password: str) -> str:
    return hashlib.md5(password.encode()).hexdigest()
""",
    issue_categories=["security"],
)

SEC_AUDIT_07_CLEAN_SUBPROCESS = BenchmarkSample(
    sample_id="sec_audit_07",
    description="Clean subprocess call with arg list; looks like shell=True but isn't",
    code="""\
import subprocess

def run_git_log(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", path, "log", "--oneline", "-n", "20"],
        text=True,
    )
""",
    issue_categories=[],
    is_clean=True,
)

SEC_AUDIT_08_CLEAN_SAFE_EVAL = BenchmarkSample(
    sample_id="sec_audit_08",
    description="eval gated behind a strict whitelist; safe in practice",
    code="""\
import ast

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
)


def safe_arith(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed node: {type(node).__name__}")
    return eval(compile(tree, "<safe>", "eval"))
""",
    issue_categories=[],
    is_clean=True,
)


def get_security_audit_corpus() -> list[BenchmarkSample]:
    """Return the 8-sample security-audit benchmark corpus."""
    return [
        SEC_AUDIT_01_SQLI,
        SEC_AUDIT_02_HARDCODED_SECRET,
        SEC_AUDIT_03_PATH_TRAVERSAL,
        SEC_AUDIT_04_COMMAND_INJECTION,
        SEC_AUDIT_05_INSECURE_DESERIALIZATION,
        SEC_AUDIT_06_WEAK_CRYPTO,
        SEC_AUDIT_07_CLEAN_SUBPROCESS,
        SEC_AUDIT_08_CLEAN_SAFE_EVAL,
    ]
