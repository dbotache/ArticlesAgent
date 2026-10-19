"""Opt-in checks against the real models configured in this installation's .env
(whichever providers they are: Azure OpenAI, OpenAI, Anthropic, an OpenAI-compatible server).

    ARTICLES_AGENT_LIVE_TESTS=1 .venv/bin/python -m pytest -m live

Costs a few cents at most: one tiny call per model plus one question over the 3-paper test library.
"""

import os

import pytest
from dotenv import dotenv_values

from articles_agent import graph, llm
from conftest import CODE_ROOT, SETTING_KEYS

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("ARTICLES_AGENT_LIVE_TESTS") != "1",
                       reason="set ARTICLES_AGENT_LIVE_TESTS=1 to call the real Azure deployments"),
]

REAL_GET_CHAT = llm.get_chat  # captured before any fixture swaps in the fakes
REAL_GET_EMBEDDINGS = llm.get_embeddings


@pytest.fixture
def real_azure(monkeypatch):
    settings = {k: v for k, v in dotenv_values(CODE_ROOT / ".env").items() if k in SETTING_KEYS and v}
    for k in SETTING_KEYS:  # only the real configuration, none of the test fixture's fake settings
        monkeypatch.delenv(k, raising=False)
    for k, v in settings.items():
        monkeypatch.setenv(k, v)
    if missing := llm.missing_settings():
        pytest.skip(f"incomplete model configuration in .env: {missing}")
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
    monkeypatch.setattr(graph, "get_chat", REAL_GET_CHAT)  # retrieval keeps the project's pinned fake embeddings
    res = graph.ask("By how much did the CNN surrogate reduce finite element evaluations?")
    assert res.findings, res.summary
    assert any("70 percent" in f.quote for f in res.findings)
    assert all(f.page >= 1 for f in res.findings)
    assert res.references
