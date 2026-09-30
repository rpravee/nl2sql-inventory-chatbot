# 📦 Inventory Chatbot: Natural Language to SQL with Llama 3

   **Live demo:** https://praveena-nl2sql-chatbot.streamlit.app

Ask questions about inventory data in plain English and get back the SQL, the result table, and a short explanation.
A Llama 3 model (via Hugging Face) turns each question into a SQL query, which is **validated and run read-only**.

> This is an independent demo built on a made-up inventory dataset. It contains no employer code or data.

## Demo questions

- Which products are below their reorder level?
- What is the total stock value in each warehouse?
- Top 5 products by units shipped in the last 30 days
- How many products does each supplier provide?
- *(follow-up)* "…and only for Furniture" (recent turns are sent back to the model as context)

## How it works

```mermaid
flowchart LR
    Q[User question] --> P[Prompt: schema + rules + examples]
    P --> L[Llama 3 via Hugging Face]
    L --> G{SQL guard}
    G -- unsafe / invalid --> R[Send error back to model, retry once]
    R --> L
    G -- safe --> D[(Read-only database)]
    D --> T[Result table]
    T --> S[Llama 3 summary]
    S --> UI[Streamlit chat or FastAPI response]
```

1. **Schema-aware prompt.** Tables, columns, keys and small sets of allowed values are read from the database
   automatically, so the model sees real category and warehouse names.
2. **Dynamic SQL generation.** The model returns one SELECT query (few-shot examples guide the style).
3. **SQL guard** (`src/sql_guard.py`). The SQL is parsed with `sqlglot` and rejected unless it is a single
   read-only SELECT on known tables. A `LIMIT` is added or capped.
4. **Self-correction.** If validation or execution fails, the error is sent back to the model for one retry.
5. **Read-only execution.** The SQLite file is opened in read-only mode as a second layer of defence.
6. **Plain-English summary** of the result (can be switched off).

## Tech stack

Python · Hugging Face `huggingface_hub` (Llama 3.1 8B Instruct) · SQLAlchemy · SQLite (MySQL supported) ·
sqlglot · Pandas · Streamlit · FastAPI · pytest

## Quick start

```bash
git clone https://github.com/rpravee/nl2sql-inventory-chatbot.git
cd nl2sql-inventory-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Get Llama 3 access on Hugging Face

1. Create a free account at <https://huggingface.co>.
2. Open the model page (default: `meta-llama/Llama-3.1-8B-Instruct`), read the license, and request access.
   Approval is usually quick.
3. Create an access token at <https://huggingface.co/settings/tokens>. Use a fine-grained token that is
   allowed to make calls to **Inference Providers**.
4. Copy `.env.example` to `.env` and paste your token:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

> Model availability and free usage limits on Inference Providers change over time. If a request fails, check
> the model page on Hugging Face, or set `HF_MODEL` to another instruct model that is available to you.

### 2. Run the chat app

```bash
streamlit run app.py
```

The demo database is created automatically on first run (or run `python -m src.database --reset`).

### 3. Or run the API

```bash
uvicorn api:app --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which products are below their reorder level?"}'
```

### 4. Run the tests

```bash
pytest
```

The tests use a fake LLM, so they need no token and no internet connection.

## Safety design

Generated SQL is treated as untrusted input:

| Layer | What it does |
|---|---|
| Prompt rules | Ask for one read-only SELECT; allow the model to decline unanswerable questions |
| SQL guard | Blocks multiple statements, non-SELECT statements, unknown tables (including `sqlite_master`), forces a row limit |
| Read-only connection | SQLite opened with `mode=ro`; for MySQL use a database user with SELECT-only rights |
| Row cap | At most `MAX_ROWS` rows are ever returned |

Tests in `tests/test_sql_guard.py` and `tests/test_pipeline.py` cover injection-style and destructive queries.

## Project structure

```
├── app.py                 # Streamlit chat UI
├── api.py                 # FastAPI service (/ask, /schema, /health)
├── src/
│   ├── config.py          # settings from .env
│   ├── database.py        # demo data, read-only engine, schema introspection
│   ├── prompts.py         # system prompt, few-shot examples, repair + summary prompts
│   ├── llm.py             # Hugging Face API backend + optional local backend
│   ├── sql_guard.py       # SQL extraction and validation
│   └── pipeline.py        # question -> SQL -> validate -> execute -> summarize
├── tests/                 # pytest suite (fake LLM)
├── requirements.txt
├── requirements-local.txt # extra packages for running the model locally
└── .env.example
```

## Using MySQL instead of SQLite

```env
DATABASE_URL=mysql+pymysql://readonly_user:password@localhost/inventory
SQL_DIALECT=mysql
```

Install the driver with `pip install pymysql`. The schema is read automatically. If your tables need business
context, edit `BUSINESS_NOTES` in `src/prompts.py`.

## Running the model locally (optional)

Set `LLM_BACKEND=local` and install `requirements-local.txt`. An 8B model needs a GPU. On CPU, use a small model
such as `meta-llama/Llama-3.2-1B-Instruct` (expect lower SQL quality).

## Limitations

- Accuracy depends on the model. Complex multi-join or ambiguous questions can produce wrong SQL,
  so the generated SQL is always shown for review.
- The demo schema is small. Very large schemas would need table selection or retrieval before prompting.
- Follow-up questions use only the last three turns.

## Ideas for next steps

- Evaluate accuracy on a test set of question and expected-result pairs
- Add charts for numeric results
- Retrieve only the relevant tables for large schemas
- Deploy the API with Docker
