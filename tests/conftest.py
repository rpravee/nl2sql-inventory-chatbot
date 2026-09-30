import pytest

from src.database import make_engine, seed_database


class FakeLLM:
    """Returns canned replies in order, so tests never call the real model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, max_tokens=512, temperature=0.1):
        self.calls.append(messages)
        return self.replies.pop(0)


@pytest.fixture()
def engine(tmp_path):
    db = seed_database(tmp_path / "test.db")
    return make_engine(f"sqlite:///{db}")
