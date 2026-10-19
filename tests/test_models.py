"""Model providers, switchable chat models, and embedding models pinned per project."""

import json

import chromadb
import pytest

from articles_agent import graph, llm, projects
from articles_agent.config import use_project
from articles_agent.graph import ask
from articles_agent.index import reembed_estimate, run_index
from articles_agent.llm import EmbeddingSpec, ModelSpec, chat_spec, default_embedding_spec
from articles_agent.projects import embedding_pin, embedding_spec, pin_embedding
from articles_agent.retriever import HybridRetriever
from conftest import FAKE_ENV, SETTING_KEYS

quiet = {"progress": lambda *a: None}


@pytest.fixture
def clean_env(monkeypatch):
    for k in SETTING_KEYS:
        monkeypatch.delenv(k, raising=False)
    llm.get_chat.cache_clear()
    llm.get_embeddings.cache_clear()
    return monkeypatch


# ------------------------------------------------------------------ settings → specs


def test_legacy_azure_settings_still_work(clean_env):
    clean_env.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/")
    clean_env.setenv("AZURE_OPENAI_API_KEY", "k")
    clean_env.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-mini")
    clean_env.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    assert chat_spec() == chat_spec(fast=True) == ModelSpec("azure", "gpt-4.1-mini")
    clean_env.setenv("AZURE_OPENAI_FAST_DEPLOYMENT", "gpt-4.1-nano")
    assert chat_spec(fast=True).model == "gpt-4.1-nano"
    emb = default_embedding_spec()
    assert emb == EmbeddingSpec("azure", "text-embedding-3-small")
    assert emb.identity == "text-embedding-3-small"  # what indexes built before providers recorded
    assert llm.missing_settings() == []


def test_provider_settings(clean_env):
    clean_env.setenv("CHAT_PROVIDER", "anthropic")
    clean_env.setenv("CHAT_MODEL", "claude-opus-5-5")
    clean_env.setenv("FAST_MODEL", "claude-haiku-5-5")  # provider defaults to the chat provider
    clean_env.setenv("EMBEDDING_PROVIDER", "compatible")
    clean_env.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    clean_env.setenv("COMPATIBLE_BASE_URL", "http://localhost:11434/v1")
    assert chat_spec().label == "anthropic:claude-opus-5-5"
    assert chat_spec(fast=True) == ModelSpec("anthropic", "claude-haiku-5-5")
    emb = default_embedding_spec()
    assert (emb.identity, emb.base_url) == ("compatible:nomic-embed-text", "http://localhost:11434/v1")
    assert llm.missing_settings() == ["ANTHROPIC_API_KEY"]  # compatible needs no key
    clean_env.setenv("FAST_PROVIDER", "openai")
    clean_env.setenv("FAST_MODEL", "gpt-5.4-mini")
    assert llm.missing_settings() == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]


def test_missing_model_and_unknown_provider(clean_env):
    clean_env.setenv("CHAT_PROVIDER", "openai")
    clean_env.setenv("OPENAI_API_KEY", "k")
    clean_env.setenv("EMBEDDING_PROVIDER", "nonsense")
    clean_env.setenv("EMBEDDING_MODEL", "x")
    missing = llm.missing_settings()
    assert "CHAT_MODEL" in missing and "unknown provider 'nonsense'" in missing


def test_clients_per_provider(clean_env):
    """Clients are built without network access; check what each provider gets."""
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    clean_env.setenv("ANTHROPIC_API_KEY", "a-key")
    claude = llm._chat_for(ModelSpec("anthropic", "claude-opus-5-5"))
    assert isinstance(claude, ChatAnthropic) and claude.model == "claude-opus-5-5"
    assert claude.temperature is None  # current Claude models reject a custom temperature
    assert claude.max_tokens == 16000
    clean_env.setenv("OPENAI_API_KEY", "o-key")
    reasoning = llm._chat_for(ModelSpec("openai", "gpt-5.4"))
    classic = llm._chat_for(ModelSpec("openai", "gpt-4.1-mini"))
    assert isinstance(reasoning, ChatOpenAI) and reasoning.temperature is None
    assert classic.temperature == 0
    clean_env.setenv("COMPATIBLE_BASE_URL", "http://localhost:11434/v1")
    local = llm._chat_for(ModelSpec("compatible", "llama3.3"))
    assert local.openai_api_base == "http://localhost:11434/v1"
    assert local.openai_api_key.get_secret_value() == "none"  # local servers need no key
    clean_env.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
    clean_env.setenv("AZURE_OPENAI_API_KEY", "z")
    azure = llm._embeddings_for(EmbeddingSpec("azure", "text-embedding-3-small"))
    assert azure.openai_api_base == "https://res.openai.azure.com/openai/v1/"


@pytest.mark.parametrize("provider, expected", [
    ("openai", {"method": "json_schema", "strict": True}),
    ("azure", {"method": "json_schema", "strict": True}),
    ("anthropic", {"method": "json_schema"}),  # native structured outputs, never forced tool use
    ("compatible", {"method": "json_schema"}),
])
def test_structured_output_method_per_provider(clean_env, provider, expected):
    clean_env.setenv("CHAT_PROVIDER", provider)
    clean_env.setenv("CHAT_MODEL", "m")

    class Recorder:
        def with_structured_output(self, schema, **kwargs):
            self.kwargs = kwargs
            return self
    rec = Recorder()
    llm.structured(rec, dict)
    assert rec.kwargs == expected


def test_structured_output_override_for_compatible(clean_env):
    clean_env.setenv("CHAT_PROVIDER", "compatible")
    clean_env.setenv("STRUCTURED_OUTPUT_METHOD", "function_calling")

    class Recorder:
        def with_structured_output(self, schema, **kwargs):
            self.kwargs = kwargs
    rec = Recorder()
    llm.structured(rec, dict)
    assert rec.kwargs == {"method": "function_calling"}


def test_save_settings_removes_empty_values(app_home, clean_env):
    llm.save_settings({"CHAT_PROVIDER": "openai", "FAST_MODEL": "x"})
    llm.save_settings({"FAST_MODEL": ""})
    assert (app_home / ".env").read_text() == "CHAT_PROVIDER=openai\n"
    import os
    assert "FAST_MODEL" not in os.environ


# ------------------------------------------------------------------ embedding model pinned per project


def test_build_pins_the_embedding_model(built):
    pin = embedding_pin()
    assert (pin["provider"], pin["model"], pin["identity"]) == ("openai", "fake-embed", "openai:fake-embed")
    assert pin["dimensions"] == 64
    meta = json.loads((built / "project.json").read_text())
    assert meta["name"] == "Test project" and meta["embedding"]["identity"] == "openai:fake-embed"


def test_changing_the_default_does_not_touch_existing_projects(built, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "another-embedding-model")
    assert default_embedding_spec().model == "another-embedding-model"
    assert embedding_spec().model == "fake-embed"  # pinned
    hits = HybridRetriever().search("finite element evaluations", k=2)  # still searchable
    assert hits
    new = projects.create_project("New one")
    assert embedding_spec(new.slug).model == "another-embedding-model"  # new projects use the default


def test_project_built_before_pins_gets_pinned_from_its_index(built):
    meta_file = built / "project.json"
    meta = json.loads(meta_file.read_text())
    del meta["embedding"]
    meta_file.write_text(json.dumps(meta))
    # an index built before providers: bare Azure deployment name in the collection metadata
    col = chromadb.PersistentClient(path=str(built / "data" / "chroma")).get_collection("chunks")
    col.modify(metadata={"embedding_model": "text-embedding-3-small"})
    pin = embedding_pin()
    assert (pin["provider"], pin["model"], pin["dimensions"]) == ("azure", "text-embedding-3-small", 64)
    assert json.loads(meta_file.read_text())["embedding"]["identity"] == "text-embedding-3-small"


def test_query_dimension_mismatch_is_explained(built, monkeypatch):
    r = HybridRetriever()
    monkeypatch.setattr(r.embeddings, "embed_query", lambda q: [0.1] * 10)
    with pytest.raises(RuntimeError, match="returned 10 dimensions.*has 64"):
        r.search("motor")


def test_reembed_with_another_model_and_back(built, fake_embeddings):
    old = embedding_spec()
    new = EmbeddingSpec("openai", "other-embed")
    before = reembed_estimate(new)
    assert before["passages"] > 0 and before["cached"] == 0 and before["tokens"] > 0
    pin_embedding("test", new)
    report = run_index(rebuild=True, **quiet)
    assert report["embedded"] == report["added"] == before["passages"]
    assert embedding_pin()["identity"] == "openai:other-embed" and embedding_pin()["dimensions"] == 64
    graph.get_retriever.cache_clear()
    assert HybridRetriever().collection.metadata["embedding_model"] == "openai:other-embed"
    assert reembed_estimate(new)["cached"] == before["passages"]
    # switching back costs nothing: the old model's vectors are still in the shared cache
    calls = fake_embeddings.calls
    pin_embedding("test", old)
    back = run_index(**quiet)  # the pin decides; the index of the other model is replaced
    assert back["embedded"] == 0 and back["reused"] == back["added"] and fake_embeddings.calls == calls


def test_answers_record_the_models(built):
    res = ask("How do CNN surrogates speed up topology optimization?")
    assert res.models == {"chat": "openai:fake-chat", "fast": "openai:fake-chat", "embedding": "openai:fake-embed"}
    assert "_Models: answers openai:fake-chat, embeddings openai:fake-embed_" in res.to_markdown()


def test_cross_project_answers_list_every_embedding_model(built, sample_pdfs, tmp_path, monkeypatch):
    from articles_agent.ingest import run_ingest
    from articles_agent.sources import add_source
    from conftest import bib_entry
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-embed-2")  # the second project is built with another model
    other = projects.create_project("Other")
    bib = tmp_path / "o.bib"
    bib.write_text(bib_entry("muster_fehlererkennung_2019",
                             f"PDF:{sample_pdfs['muster_fehlererkennung_2019']}:application/pdf"))
    with use_project(other.slug):
        add_source("citavi", str(bib))
        run_ingest(**quiet)
        run_index(**quiet)
    res = ask("Klassifikator", projects=["test", other.slug])
    assert res.models["embedding"] == "openai:fake-embed, openai:fake-embed-2"
