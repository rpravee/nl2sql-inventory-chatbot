"""Safety layer between the LLM and the database.

LLM output is untrusted. Before anything is executed we check that it is exactly one
read-only SELECT, that it only touches known tables, and that it has a row limit.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


class UnsafeSQLError(ValueError):
    """Raised when generated SQL is invalid or not allowed to run."""


_SET_OP = getattr(exp, "SetOperation", None) or exp.Union
_ALLOWED_ROOTS = (exp.Select, _SET_OP)
_FORBIDDEN = tuple(
    getattr(exp, name) for name in
    ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "Merge", "Command", "Pragma",
     "Attach", "Detach", "TruncateTable", "Copy", "LoadData", "Grant", "Set", "Use")
    if hasattr(exp, name)
)

_SQL_START = re.compile(r"\bselect\b|\bwith\s+[\w\"`]+\s+as\s*\(", re.IGNORECASE)


def extract_sql(text: str) -> str | None:
    """Pull the SQL out of a model reply. Returns None if the model declined to answer."""
    if "CANNOT_ANSWER" in text:
        return None
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    match = _SQL_START.search(text)
    if not match:
        return None
    return text[match.start():].strip().rstrip(";").strip()


def validate_sql(sql: str, allowed_tables: set[str], dialect: str = "sqlite", max_rows: int = 100) -> str:
    """Return a safe, normalised version of `sql` or raise UnsafeSQLError."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as exc:
        raise UnsafeSQLError(f"SQL could not be parsed: {exc}") from exc

    if len(statements) != 1:
        raise UnsafeSQLError("Exactly one SQL statement is allowed.")
    stmt = statements[0]

    if not isinstance(stmt, _ALLOWED_ROOTS):
        raise UnsafeSQLError("Only SELECT queries are allowed.")
    if stmt.find(*_FORBIDDEN):
        raise UnsafeSQLError("The query contains a forbidden operation.")

    cte_names = {cte.alias.lower() for cte in stmt.find_all(exp.CTE)}
    used = {t.name.lower() for t in stmt.find_all(exp.Table)} - cte_names
    unknown = used - {t.lower() for t in allowed_tables}
    if unknown:
        raise UnsafeSQLError(f"Unknown or disallowed table(s): {', '.join(sorted(unknown))}")

    limit = stmt.args.get("limit")
    if limit is None:
        stmt = stmt.limit(max_rows)
    else:
        value = limit.expression
        if isinstance(value, exp.Literal) and value.is_int and int(value.name) > max_rows:
            stmt = stmt.limit(max_rows)

    return stmt.sql(dialect=dialect)
