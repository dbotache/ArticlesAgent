"""End to end without Azure: ingestion → index → hybrid search → LangGraph answer with citations."""

import json

import pytest

from articles_agent import graph
from articles_agent.graph import ask, ask_stream
from articles_agent.index import run_index
from articles_agent.ingest import run_ingest
from articles_agent.retriever import HybridRetriever, tokenize
from articles_agent.store import load_chunks, parsed_cache_file
from conftest import read_json
from fakes import INVENTED_QUOTE, FakeChat, default_responder

DOE = "doe_topology_2021"


# ------------------------------------------------------------------ ingestion and index


def test_ingest_report_and_papers(built):
    report = read_json(built / "data" / "ingest_report.json")
    assert report["pdfs"] == 3 and report["parse_errors"] == {}
    papers = read_json(built / "data" / "papers.json")
    assert set(papers) >= {DOE, "muster_fehlererkennung_2019"}
    own = [p for p in papers.values() if not p["in_bib"]]
    assert [p["title"] for p in own] == ["Reinforcement Learning for Motor Control: A Survey"]
    assert papers[DOE]["short_cite"] == "Doe & Roe, 2021"


def test_chunks_have_sections_pages_and_no_references(built):
    from articles_agent.config import load_config
    chunks = load_chunks(load_config())
    doe = [c for c in chunks if c["paper_id"] == DOE]
    assert {c["section"] for c in doe} >= {"Abstract", "2 Method", "3 Results"}
    assert all(not c["is_references"] for c in chunks)  # reference lists are not searchable
    results = next(c for c in doe if c["section"] == "3 Results")
    assert results["page_start"] == 2
    assert "70 percent" in results["text"]


def test_ingest_is_incremental(built, monkeypatch):
    from articles_agent import ingest
    monkeypatch.setattr(ingest, "_parse_to_cache", lambda *a: pytest.fail("re-parsed an unchanged PDF"))
    assert run_ingest(progress=lambda *a: None)["parsed"] == 0


def test_index_is_incremental_and_removes_stale(built, fake_embeddings):
    first = run_index(progress=lambda *a: None)
    assert first["added"] == 0 and fake_embeddings.calls == 0
    # drop one paper's chunks (as if its library was removed) → its vectors are deleted
    chunks_file = built / "data" / "chunks.jsonl"
    lines = [line for line in chunks_file.read_text().splitlines() if json.loads(line)["paper_id"] != DOE]
    chunks_file.write_text("\n".join(lines) + "\n")
    second = run_index(progress=lambda *a: None)
    assert second["removed"] > 0 and second["total"] == first["total"] - second["removed"]


def test_index_refuses_other_embedding_model(built, monkeypatch):
    from articles_agent.store import get_collection
    from articles_agent.config import load_config
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "another-model")
    with pytest.raises(RuntimeError, match="--rebuild"):
        get_collection(load_config(), "another-model")


def test_parsed_cache_names(app_home):
    zotero = parsed_cache_file("/z/storage/AB12CD34/paper.pdf")
    a, b = parsed_cache_file("/papers/a.pdf"), parsed_cache_file("/papers/b.pdf")
    assert zotero.parent == app_home / "shared" / "parsed"  # shared by all projects
    assert zotero.name == "AB12CD34.json"
    assert a != b and a.name.startswith("x_")  # same folder, different files


# ------------------------------------------------------------------ retrieval


def test_tokenize():
    assert tokenize("Die Topologie-Optimierung (TO) of IPMSMs!") == ["die", "topologie", "optimierung",
                                                                       "to", "of", "ipmsms"]


def test_hybrid_search_and_filters(built):
    r = HybridRetriever()
    hits = r.search("finite element evaluations reduced percent", k=3)
    assert hits[0].paper["paper_id"] == DOE
    assert hits[0].bm25_rank and hits[0].dense_rank
    assert hits[0].citation.startswith("(Doe & Roe, 2021, p")
    german = r.search("Lagerschäden Klassifikator Genauigkeit", k=2)
    assert german[0].paper["paper_id"] == "muster_fehlererkennung_2019"
    only = r.search("finite element", k=5, paper_ids=["muster_fehlererkennung_2019"])
    assert {h.paper["paper_id"] for h in only} == {"muster_fehlererkennung_2019"}
    assert r.search("finite element", k=5, year_from=2022) == []
    assert all(h.paper["year"] == "2021" for h in r.search("motor", k=10, year_from=2021, year_to=2021))


# ------------------------------------------------------------------ answer graph


def test_answer_has_verified_quotes_pages_and_references(built, fake_chat):
    res = ask("How do CNN surrogates speed up topology optimization?")
    assert res.dropped_quotes == 1  # the invented quote was removed
    assert all(INVENTED_QUOTE not in f.quote for f in res.findings)
    assert len(res.findings) == 2
    for f in res.findings:
        assert f.page >= 1 and f.citation.endswith(f"p. {f.page})")
    assert "(" in res.summary and "[S" not in res.summary  # markers became citations
    assert any(ref.startswith("Doe, J., & Roe, R. (2021).") for ref in res.references)
    assert res.queries[0] == "How do CNN surrogates speed up topology optimization?"
    assert [name for name, _ in fake_chat.calls] == ["Queries", "Relevant", "Answer"]


def test_retry_with_new_queries_when_too_few_sources(built, monkeypatch):
    def stingy(schema, payload):
        if schema is graph.Relevant:
            return graph.Relevant(source_ids=[])  # nothing relevant
        return default_responder(schema, payload)
    chat = FakeChat(stingy)
    monkeypatch.setattr(graph, "get_chat", lambda fast=False: chat)
    res = ask("Something the library does not cover")
    names = [n for n, _ in chat.calls]
    assert names == ["Queries", "Relevant", "Queries", "Relevant"]  # max_attempts = 2, then give up
    assert "too little relevant material" in chat.calls[2][1]
    assert res.findings == [] and "No passages" in res.summary


def test_grader_ids_are_validated(built, monkeypatch):
    def odd_ids(schema, payload):
        if schema is graph.Relevant:
            return graph.Relevant(source_ids=["S99", "s1", "2", "S2", "nonsense"])
        return default_responder(schema, payload)
    monkeypatch.setattr(graph, "get_chat", lambda fast=False: FakeChat(odd_ids))
    state = graph.grade({"question": "q", "candidates": HybridRetriever().search("motor", k=5)})
    assert len(state["sources"]) == 1  # 'S99' and 'nonsense' ignored, '2'/'S2' deduplicated


def test_stream_yields_every_node(built):
    nodes = [node for node, _ in ask_stream("CNN surrogate")]
    assert nodes == ["rewrite", "retrieve", "grade", "generate", "cite"]


def test_filters_reach_the_retriever(built):
    res = ask("Klassifikator", paper_ids=["muster_fehlererkennung_2019"])
    assert {f.paper_id for f in res.findings} <= {"muster_fehlererkennung_2019"}
