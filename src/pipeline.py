"""The NL -> SQL -> result pipeline that ties everything together."""
from __future__ import annotations

import re
from numbers import Integral, Real
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from . import prompts
from .database import get_schema_text, get_table_names, run_query
from .llm import LLM
from .sql_guard import UnsafeSQLError, extract_sql, validate_sql


def format_value(value) -> str:
    """Human-friendly number/text formatting for the summary prompt (5997914.0 -> 5,997,914)."""
    if value is None or (isinstance(value, float) and value != value):
        return "NULL"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return f"{int(value):,}"
    if isinstance(value, Real):
        return f"{int(value):,}" if float(value).is_integer() else f"{float(value):,.2f}"
    return str(value)


def format_table(data: pd.DataFrame) -> str:
    rows = [" | ".join(format_value(v) for v in row) for row in data.itertuples(index=False, name=None)]
    return "\n".join([" | ".join(map(str, data.columns)), *rows])


def clean_summary(text: str) -> str:
    """Remove a leading 'Here is the explanation...:' line that chat models sometimes add."""
    text = text.strip()
    return re.sub(r"^\s*here(?:'s| is| are)[^\n]{0,150}?:\s*", "", text, count=1, flags=re.IGNORECASE).strip()


class QueryError(RuntimeError):
    """The question could not be turned into a working query."""

    def __init__(self, message: str, sql: str | None = None):
        super().__init__(message)
        self.sql = sql


class CannotAnswer(QueryError):
    """The model decided the database cannot answer this question."""


@dataclass
class QueryResult:
    question: str
    sql: str
    data: pd.DataFrame
    answer: str | None = None
    attempts: int = 1
    errors: list[str] = field(default_factory=list)


class NL2SQLPipeline:
    def __init__(self, llm: LLM, engine: Engine, dialect: str = "sqlite",
                 max_rows: int = 100, max_attempts: int = 2, summarize: bool = True):
        self.llm = llm
        self.engine = engine
        self.dialect = dialect
        self.max_rows = max_rows
        self.max_attempts = max_attempts
        self.summarize = summarize
        self.schema = get_schema_text(engine)
        self.allowed_tables = get_table_names(engine)
        self._system = prompts.build_system_prompt(self.schema, dialect)

    # ------------------------------------------------------------------ #
    def _base_messages(self, history: list[tuple[str, str]] | None) -> list[dict]:
        messages = [{"role": "system", "content": self._system}]
        for question, sql in prompts.FEW_SHOT:
            messages += [{"role": "user", "content": question}, {"role": "assistant", "content": sql}]
        for question, sql in (history or [])[-3:]:  # recent turns allow follow-up questions
            messages += [{"role": "user", "content": question}, {"role": "assistant", "content": sql}]
        return messages

    def ask(self, question: str, history: list[tuple[str, str]] | None = None,
            summarize: bool | None = None) -> QueryResult:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")

        messages = self._base_messages(history) + [{"role": "user", "content": question}]
        errors: list[str] = []
        last_sql: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            raw = self.llm.chat(messages, max_tokens=400)
            candidate = extract_sql(raw)
            if candidate is None:
                raise CannotAnswer("I can't answer that from the inventory database.")
            last_sql = candidate
            try:
                safe_sql = validate_sql(candidate, self.allowed_tables, self.dialect, self.max_rows)
                data = run_query(self.engine, safe_sql, self.max_rows)
            except (UnsafeSQLError, SQLAlchemyError) as exc:
                error = str(getattr(exc, "orig", exc))
                errors.append(error)
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": prompts.REPAIR_TEMPLATE.format(error=error)}]
                continue

            answer = None
            if self.summarize if summarize is None else summarize:
                answer = self._summarize(question, safe_sql, data)
            return QueryResult(question, safe_sql, data, answer, attempt, errors)

        raise QueryError(
            f"Could not produce a working query after {self.max_attempts} attempts. Last error: {errors[-1]}",
            sql=last_sql,
        )

    def _summarize(self, question: str, sql: str, data: pd.DataFrame, max_rows: int = 20) -> str | None:
        shown = data.head(max_rows)
        table = format_table(shown) if not shown.empty else "(no rows)"
        messages = [
            {"role": "system", "content": prompts.SUMMARY_SYSTEM},
            {"role": "user", "content": prompts.SUMMARY_USER_TEMPLATE.format(
                question=question, sql=sql, shown=len(shown), total=len(data), table=table)},
        ]
        try:
            reply = self.llm.chat(messages, max_tokens=200, temperature=0.2)
        except Exception:  # the table is still useful even if the summary call fails
            return None
        return clean_summary(reply)
