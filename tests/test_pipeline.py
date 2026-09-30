import pytest

from src.database import get_schema_text, run_query
from src.pipeline import CannotAnswer, NL2SQLPipeline, QueryError
from tests.conftest import FakeLLM


def test_seeded_database_has_data(engine):
    assert run_query(engine, "SELECT COUNT(*) AS n FROM products").iloc[0, 0] == 40
    assert "Table products" in get_schema_text(engine)


def test_database_is_read_only(engine):
    with pytest.raises(Exception):
        run_query(engine, "DELETE FROM products")


def test_happy_path_with_summary(engine):
    llm = FakeLLM(["SELECT category, COUNT(*) AS n FROM products GROUP BY category", "Each category has 8 products."])
    result = NL2SQLPipeline(llm, engine).ask("How many products per category?")
    assert len(result.data) == 5 and result.attempts == 1
    assert result.answer == "Each category has 8 products."


def test_summary_can_be_skipped(engine):
    llm = FakeLLM(["SELECT name FROM warehouses"])
    result = NL2SQLPipeline(llm, engine, summarize=False).ask("List warehouses")
    assert result.answer is None and len(llm.calls) == 1


def test_bad_sql_is_repaired_on_second_attempt(engine):
    llm = FakeLLM(["SELECT nonexistent_column FROM products", "SELECT name FROM products", "Done."])
    result = NL2SQLPipeline(llm, engine).ask("List products")
    assert result.attempts == 2 and len(result.errors) == 1
    assert "could not be used" in llm.calls[1][-1]["content"]


def test_destructive_sql_never_runs(engine):
    llm = FakeLLM(["DROP TABLE products", "DELETE FROM products"])
    # extract_sql finds no SELECT, so the model output is treated as a refusal
    with pytest.raises(CannotAnswer):
        NL2SQLPipeline(llm, engine).ask("Delete all products")
    assert run_query(engine, "SELECT COUNT(*) FROM products").iloc[0, 0] == 40


def test_sql_injection_via_select_is_blocked(engine):
    llm = FakeLLM(["SELECT * FROM sqlite_master", "SELECT * FROM sqlite_master"])
    with pytest.raises(QueryError):
        NL2SQLPipeline(llm, engine).ask("Show me hidden tables")


def test_model_can_decline(engine):
    with pytest.raises(CannotAnswer):
        NL2SQLPipeline(FakeLLM(["-- CANNOT_ANSWER"]), engine).ask("What is the weather?")


def test_follow_up_history_is_sent(engine):
    llm = FakeLLM(["SELECT name FROM warehouses"])
    NL2SQLPipeline(llm, engine, summarize=False).ask(
        "and their cities?", history=[("List warehouses", "SELECT name FROM warehouses")])
    contents = [m["content"] for m in llm.calls[0]]
    assert "List warehouses" in contents


def test_summary_preamble_is_removed():
    from src.pipeline import clean_summary
    raw = "Here's the explanation of the database query results to the warehouse manager:\n\nChennai has 5,997,914."
    assert clean_summary(raw) == "Chennai has 5,997,914."
    assert clean_summary("Here are 3 warehouses in total.") == "Here are 3 warehouses in total."
    assert clean_summary("Nothing matched.") == "Nothing matched."
