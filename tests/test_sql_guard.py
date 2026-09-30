import pytest

from src.sql_guard import UnsafeSQLError, extract_sql, validate_sql

TABLES = {"products", "stock", "suppliers", "warehouses", "stock_movements"}


def test_plain_select_gets_limit():
    assert validate_sql("SELECT * FROM products", TABLES, max_rows=50).endswith("LIMIT 50")


def test_large_limit_is_capped():
    assert validate_sql("SELECT * FROM products LIMIT 9999", TABLES, max_rows=100).endswith("LIMIT 100")


def test_small_limit_is_kept():
    assert validate_sql("SELECT * FROM products LIMIT 5", TABLES).endswith("LIMIT 5")


def test_cte_and_join_allowed():
    sql = ("WITH low AS (SELECT product_id FROM stock WHERE quantity < 10) "
           "SELECT p.name FROM products p JOIN low ON low.product_id = p.product_id")
    assert "WITH low" in validate_sql(sql, TABLES)


@pytest.mark.parametrize("bad", [
    "DROP TABLE products",
    "DELETE FROM stock",
    "UPDATE stock SET quantity = 0",
    "INSERT INTO suppliers (name) VALUES ('x')",
    "SELECT 1; DROP TABLE products",
    "PRAGMA table_info(products)",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM users",
    "SELEC broken ((",
])
def test_unsafe_sql_is_rejected(bad):
    with pytest.raises(UnsafeSQLError):
        validate_sql(bad, TABLES)


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Sure! SELECT name FROM products;") == "SELECT name FROM products"
    assert extract_sql("-- CANNOT_ANSWER") is None
    assert extract_sql("no query here") is None
