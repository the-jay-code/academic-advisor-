import pytest

@pytest.fixture(autouse=True)
def no_real_llm_calls(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
