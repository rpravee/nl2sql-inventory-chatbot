"""FastAPI service:  uvicorn api:app --reload   (docs at http://127.0.0.1:8000/docs)"""
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import DEFAULT_DB_PATH, load_settings
from src.database import make_engine, seed_database
from src.llm import LLMConfigError, get_llm
from src.pipeline import CannotAnswer, NL2SQLPipeline, QueryError

app = FastAPI(title="Inventory NL2SQL API", version="1.0.0",
              description="Ask questions about inventory data in plain English.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["Which products are below their reorder level?"])
    summarize: bool = True


class AskResponse(BaseModel):
    question: str
    sql: str
    answer: str | None
    columns: list[str]
    rows: list[dict]
    row_count: int
    attempts: int


@lru_cache(maxsize=1)
def get_pipeline() -> NL2SQLPipeline:
    settings = load_settings()
    if settings.database_url.startswith("sqlite") and not DEFAULT_DB_PATH.exists():
        seed_database(DEFAULT_DB_PATH)
    llm = get_llm(settings.backend, settings.hf_model, settings.hf_token)
    return NL2SQLPipeline(llm, make_engine(settings.database_url), settings.dialect, settings.max_rows)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict:
    try:
        return {"schema": get_pipeline().schema}
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    try:
        result = get_pipeline().ask(body.question, summarize=body.summarize)
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except CannotAnswer as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except QueryError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "sql": exc.sql})
    return AskResponse(
        question=result.question, sql=result.sql, answer=result.answer,
        columns=list(result.data.columns),
        rows=result.data.astype(object).where(result.data.notna(), None).to_dict(orient="records"),
        row_count=len(result.data), attempts=result.attempts,
    )
