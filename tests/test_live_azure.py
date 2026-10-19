"""Opt-in checks against the real Azure OpenAI deployments of this installation.

    ARTICLES_AGENT_LIVE_TESTS=1 .venv/bin/python -m pytest -m live

Uses the endpoint, key and deployments from the project's .env. Costs a few cents at most:
one tiny call per deployment plus one question over the 3-paper test library.
"""

import os

import pytest
from dotenv import dotenv_values

from articles_agent import graph, llm
from conftest import CODE_ROOT

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("ARTICLES_AGENT_LIVE_TESTS") != "1",
                       reason="set ARTICLES_AGENT_LIVE_TESTS=1 to call the real Azure deployments"),
]

REAL_GET_CHAT = llm.get_chat  # captured before any fixture swaps in the fakes
REAL_GET_EMBEDDINGS = llm.get_embeddings


@pytest.fixture
def real_azure(monkeypatch):
    settings = {k: v for k, v in dotenv_values(CODE_ROOT / ".env").items() if k.startswith("AZURE_OPENAI_")}
    if not all(settings.get(k) for k in llm.REQUIRED):
        pytest.skip("no complete Azure configuration in .env")
    for k, v in settings.items():
        monkeypatch.setenv(k, v)
    REAL_GET_CHAT.cache_clear()
    REAL_GET_EMBEDDINGS.cache_clear()
    yield
    REAL_GET_CHAT.cache_clear()
    REAL_GET_EMBEDDINGS.cache_clear()


def test_every_deployment_responds(workspace, real_azure, monkeypatch):
    monkeypatch.setattr(llm, "get_chat", REAL_GET_CHAT)
    monkeypatch.setattr(llm, "get_embeddings", REAL_GET_EMBEDDINGS)
    checks = llm.check_connection()
    assert all(ok for _, ok, _ in checks), checks


def test_real_model_follows_the_citation_contract(built, real_azure, monkeypatch):
    """Structured output works with the deployed model, and quotes survive verification."""
    monkeypatch.setattr(graph, "get_chat", REAL_GET_CHAT)
    # retrieval keeps the fake embeddings the test index was built with
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "fake-embed")
    res = graph.ask("By how much did the CNN surrogate reduce finite element evaluations?")
    assert res.findings, res.summary
    assert any("70 percent" in f.quote for f in res.findings)
    assert all(f.page >= 1 for f in res.findings)
    assert res.references
