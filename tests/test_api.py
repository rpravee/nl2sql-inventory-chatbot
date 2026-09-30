from fastapi.testclient import TestClient

import api
from src.pipeline import NL2SQLPipeline
from tests.conftest import FakeLLM


def test_ask_endpoint(engine, monkeypatch):
    llm = FakeLLM(["SELECT name, city FROM warehouses", "Three warehouses."])
    monkeypatch.setattr(api, "get_pipeline", lambda: NL2SQLPipeline(llm, engine))
    client = TestClient(api.app)

    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/ask", json={"question": "List warehouses"}).json()
    assert body["row_count"] == 3 and body["columns"] == ["name", "city"]
    assert body["answer"] == "Three warehouses."


def test_ask_validation_error(engine):
    assert TestClient(api.app).post("/ask", json={"question": "hi"}).status_code == 422
