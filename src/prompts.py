"""Prompt templates for SQL generation, self-repair and result summaries."""

DIALECT_HINTS = {
    "sqlite": "Dates are stored as TEXT 'YYYY-MM-DD'. Use date('now', '-30 days') for relative dates.",
    "mysql": "Use CURDATE() - INTERVAL 30 DAY for relative dates.",
}

BUSINESS_NOTES = """\
NOTES
- stock.quantity is the current number of units on hand for a product in one warehouse.
- products.reorder_level is the minimum TOTAL quantity across all warehouses before restocking is needed.
- stock_movements.movement_type: 'IN' means goods received, 'OUT' means goods shipped or sold.
- Stock value = quantity * unit_price."""

SQL_SYSTEM_TEMPLATE = """\
You are a careful data analyst who writes {dialect} SQL for an inventory database.

DATABASE SCHEMA
{schema}

{notes}

RULES
- Reply with exactly ONE read-only SELECT query (a WITH ... SELECT is fine) and nothing else.
- No explanations and no markdown.
- Use only the tables and columns listed in the schema.
- {dialect_hint}
- Use readable column aliases. Add ORDER BY for "top", "most", "least" or "lowest" questions.
- If the question cannot be answered from this schema, reply with exactly: -- CANNOT_ANSWER
"""

FEW_SHOT = [
    ("Which products are below their reorder level?",
     "SELECT p.name, p.category, SUM(s.quantity) AS total_quantity, p.reorder_level "
     "FROM products p JOIN stock s ON s.product_id = p.product_id "
     "GROUP BY p.product_id, p.name, p.category, p.reorder_level "
     "HAVING SUM(s.quantity) < p.reorder_level ORDER BY total_quantity"),
    ("What is the total stock value in each warehouse?",
     "SELECT w.name AS warehouse, SUM(s.quantity * p.unit_price) AS stock_value "
     "FROM stock s JOIN products p ON p.product_id = s.product_id "
     "JOIN warehouses w ON w.warehouse_id = s.warehouse_id "
     "GROUP BY w.warehouse_id, w.name ORDER BY stock_value DESC"),
    ("List the top 3 suppliers by number of products.",
     "SELECT sp.name AS supplier, COUNT(*) AS product_count "
     "FROM suppliers sp JOIN products p ON p.supplier_id = sp.supplier_id "
     "GROUP BY sp.supplier_id, sp.name ORDER BY product_count DESC LIMIT 3"),
]

REPAIR_TEMPLATE = (
    "That query could not be used. Error:\n{error}\n"
    "Return a corrected single SELECT query only, with no explanation."
)

SUMMARY_SYSTEM = (
    "You explain database query results to a warehouse manager in 1 to 3 short sentences. "
    "Start directly with the answer. Do not add an introduction such as 'Here is the explanation'. "
    "Use only the numbers in the result. Never invent data. If the result is empty, say that nothing matched."
)

SUMMARY_USER_TEMPLATE = """\
Question: {question}

SQL used:
{sql}

Result ({shown} of {total} rows shown):
{table}
"""


def build_system_prompt(schema: str, dialect: str) -> str:
    return SQL_SYSTEM_TEMPLATE.format(
        dialect=dialect,
        schema=schema,
        notes=BUSINESS_NOTES,
        dialect_hint=DIALECT_HINTS.get(dialect, ""),
    )
