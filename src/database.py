"""Database helpers: demo data seeding, read-only engine, schema introspection, query execution.

The demo uses SQLite so the project runs with zero setup. To use MySQL instead, set
DATABASE_URL and SQL_DIALECT=mysql (and connect with a read-only database user).
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url

from .config import DEFAULT_DB_PATH

# --------------------------------------------------------------------------- #
# Demo data (entirely made up)
# --------------------------------------------------------------------------- #
SCHEMA_SQL = """
CREATE TABLE suppliers (
    supplier_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT,
    country     TEXT
);
CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    sku           TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    supplier_id   INTEGER NOT NULL REFERENCES suppliers(supplier_id),
    unit_price    REAL NOT NULL,
    reorder_level INTEGER NOT NULL
);
CREATE TABLE warehouses (
    warehouse_id INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    city         TEXT
);
CREATE TABLE stock (
    product_id   INTEGER NOT NULL REFERENCES products(product_id),
    warehouse_id INTEGER NOT NULL REFERENCES warehouses(warehouse_id),
    quantity     INTEGER NOT NULL,
    PRIMARY KEY (product_id, warehouse_id)
);
CREATE TABLE stock_movements (
    movement_id   INTEGER PRIMARY KEY,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    warehouse_id  INTEGER NOT NULL REFERENCES warehouses(warehouse_id),
    movement_type TEXT NOT NULL,
    quantity      INTEGER NOT NULL,
    movement_date TEXT NOT NULL
);
"""

SUPPLIERS = [
    ("Kaveri Distributors", "Chennai"), ("Metro Supplies", "Coimbatore"),
    ("Bright Tech Wholesale", "Bengaluru"), ("Coastal Traders", "Kochi"),
    ("Green Leaf Agencies", "Madurai"), ("Southern Hardware Co", "Salem"),
    ("Prime Office Mart", "Hyderabad"), ("Anand Enterprises", "Tiruchirappalli"),
]

WAREHOUSES = [("Chennai Central", "Chennai"), ("Coimbatore Hub", "Coimbatore"),
              ("Trichy Depot", "Tiruchirappalli")]

CATALOG = {
    "Electronics": [("USB-C Cable 1m", 199), ("Wireless Mouse", 599), ("Bluetooth Speaker", 1499),
                    ("LED Bulb 9W", 129), ("Power Bank 10000mAh", 1299), ("HDMI Cable 2m", 349),
                    ("Webcam HD", 1799), ("Keyboard", 899)],
    "Stationery": [("A4 Paper Ream", 320), ("Ball Pen Box (50)", 250), ("Spiral Notebook", 90),
                   ("Stapler", 150), ("Highlighter Set", 180), ("Sticky Notes", 75),
                   ("Marker Pen", 60), ("File Folder", 45)],
    "Furniture": [("Office Chair", 4500), ("Study Desk", 6200), ("Bookshelf", 5400),
                  ("Filing Cabinet", 7800), ("Desk Lamp", 950), ("Footrest", 700),
                  ("Whiteboard", 1600), ("Monitor Stand", 850)],
    "Groceries": [("Basmati Rice 5kg", 620), ("Sunflower Oil 1L", 145), ("Tea Powder 500g", 260),
                  ("Sugar 1kg", 48), ("Toor Dal 1kg", 160), ("Salt 1kg", 22),
                  ("Coffee Powder 200g", 310), ("Biscuit Pack", 30)],
    "Hardware": [("Screwdriver Set", 420), ("Hammer", 280), ("Measuring Tape 5m", 130),
                 ("Cordless Drill", 3200), ("Wrench Set", 900), ("Safety Gloves", 110),
                 ("Padlock", 240), ("Extension Board", 480)],
}


def seed_database(path: Path | str = DEFAULT_DB_PATH, seed: int = 42, reset: bool = False) -> Path:
    """Create the demo inventory database with randomly generated (fake) data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not reset:
            return path
        path.unlink()

    rng = random.Random(seed)
    today = date.today()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)

    conn.executemany(
        "INSERT INTO suppliers (name, city, country) VALUES (?, ?, 'India')", SUPPLIERS)
    conn.executemany("INSERT INTO warehouses (name, city) VALUES (?, ?)", WAREHOUSES)

    product_id = 0
    for category, items in CATALOG.items():
        for name, price in items:
            product_id += 1
            conn.execute(
                "INSERT INTO products (sku, name, category, supplier_id, unit_price, reorder_level) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"SKU-{product_id:04d}", name, category,
                 rng.randint(1, len(SUPPLIERS)), price, rng.randint(40, 120)),
            )
            for warehouse_id in range(1, len(WAREHOUSES) + 1):
                low = rng.random() < 0.2  # ~20% of stock rows are nearly empty
                qty = rng.randint(0, 15) if low else rng.randint(20, 300)
                conn.execute("INSERT INTO stock VALUES (?, ?, ?)", (product_id, warehouse_id, qty))

    for _ in range(800):
        moved_on = today - timedelta(days=rng.randint(0, 120))
        conn.execute(
            "INSERT INTO stock_movements (product_id, warehouse_id, movement_type, quantity, movement_date) "
            "VALUES (?, ?, ?, ?, ?)",
            (rng.randint(1, product_id), rng.randint(1, len(WAREHOUSES)),
             rng.choices(["IN", "OUT"], weights=[35, 65])[0], rng.randint(1, 40), moved_on.isoformat()),
        )
    conn.commit()
    conn.close()
    return path


# --------------------------------------------------------------------------- #
# Engine, schema, execution
# --------------------------------------------------------------------------- #
def make_engine(database_url: str) -> Engine:
    """Create an engine. SQLite databases are opened read-only as a second line of defence."""
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        db_path = Path(url.database).resolve()
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}. Run: python -m src.database")
        uri = f"{db_path.as_uri()}?mode=ro"
        return create_engine("sqlite://", creator=lambda: sqlite3.connect(uri, uri=True))
    return create_engine(database_url, pool_pre_ping=True)


def get_table_names(engine: Engine) -> set[str]:
    return {name.lower() for name in inspect(engine).get_table_names()}


def get_schema_text(engine: Engine, max_distinct: int = 10) -> str:
    """Describe tables, columns, keys and small sets of allowed values, for the LLM prompt."""
    insp = inspect(engine)
    quote = engine.dialect.identifier_preparer.quote
    lines: list[str] = []
    with engine.connect() as conn:
        for table in sorted(insp.get_table_names()):
            pk = set(insp.get_pk_constraint(table).get("constrained_columns", []))
            fks = {fk["constrained_columns"][0]: f"{fk['referred_table']}.{fk['referred_columns'][0]}"
                   for fk in insp.get_foreign_keys(table)}
            columns = insp.get_columns(table)
            cols = []
            for col in columns:
                desc = f"{col['name']} {str(col['type']).split('(')[0]}"
                if col["name"] in pk:
                    desc += " PK"
                if col["name"] in fks:
                    desc += f" -> {fks[col['name']]}"
                cols.append(desc)
            lines.append(f"Table {table} ({', '.join(cols)})")

            for col in columns:
                col_type = str(col["type"]).upper()
                is_text = "CHAR" in col_type or "TEXT" in col_type
                if col["name"] in pk or col["name"] in fks or not is_text:
                    continue
                rows = conn.execute(text(
                    f"SELECT DISTINCT {quote(col['name'])} FROM {quote(table)} LIMIT {max_distinct + 1}"
                )).fetchall()
                if 0 < len(rows) <= max_distinct:
                    values = ", ".join(repr(r[0]) for r in rows)
                    lines.append(f"  {col['name']} values: {values}")
    return "\n".join(lines)


def run_query(engine: Engine, sql: str, max_rows: int = 100) -> pd.DataFrame:
    """Execute an already-validated SELECT and return the rows as a DataFrame."""
    if engine.dialect.paramstyle in ("format", "pyformat"):
        sql = sql.replace("%", "%%")  # keep LIKE '%x%' working with MySQL drivers
    with engine.connect() as conn:
        result = conn.exec_driver_sql(sql)
        rows = result.fetchmany(max_rows)
        return pd.DataFrame(rows, columns=list(result.keys()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create the demo inventory database.")
    parser.add_argument("--reset", action="store_true", help="delete and recreate the database")
    args = parser.parse_args()
    print(f"Database ready: {seed_database(reset=args.reset)}")
