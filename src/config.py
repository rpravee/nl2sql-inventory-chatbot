"""Application settings, read from environment variables (and a local .env file)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "inventory.db"

load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    hf_token: str | None
    hf_model: str
    backend: str
    database_url: str
    dialect: str
    max_rows: int


def load_settings() -> Settings:
    return Settings(
        hf_token=os.getenv("HF_TOKEN") or None,
        hf_model=os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        backend=os.getenv("LLM_BACKEND", "api").lower(),
        database_url=os.getenv("DATABASE_URL") or f"sqlite:///{DEFAULT_DB_PATH}",
        dialect=os.getenv("SQL_DIALECT", "sqlite").lower(),
        max_rows=int(os.getenv("MAX_ROWS", "100")),
    )
