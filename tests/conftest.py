"""Shared fixtures.

Every test runs against a temporary installation (`workspace`): its own config, library/,
data/ and .env, fake Azure models and no network. Nothing touches the real library or index.

    .venv/bin/python -m pytest                    # all offline tests
    .venv/bin/python -m pytest -m "not integration"
    ARTICLES_AGENT_LIVE_TESTS=1 .venv/bin/python -m pytest -m live   # real Azure calls
"""

import json
import shutil
import urllib.request
from pathlib import Path

import pymupdf
import pytest

from fakes import FakeChat, FakeEmbeddings

CODE_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ sample papers

PAPERS = {
    "doe_topology_2021": {
        "title": "Topology Optimization of Electric Motors Using Convolutional Neural Networks",
        "authors": ["Doe, Jane", "Roe, Richard"], "year": "2021",
        "venue": "IEEE Transactions on Magnetics", "doi": "10.1234/TMAG.2021.0001",
        "volume": "57", "number": "6", "pages": "1--4",
        "pages_html": [
            "<h1>Topology Optimization of Electric Motors Using Convolutional Neural Networks</h1>"
            "<p>Jane Doe, Richard Roe</p><p>DOI: 10.1234/TMAG.2021.0001</p>"
            "<h2>Abstract</h2><p>We propose a convolutional neural network surrogate for the topology "
            "optimization of electric motors. The surrogate replaces most finite element evaluations.</p>"
            "<h2>1 Introduction</h2><p>Topology optimization of rotor geometries requires thousands of "
            "finite element evaluations. Genetic algorithms make this especially expensive.</p>",
            "<h2>2 Method</h2><p>A convolutional neural network is trained on cross-sectional images of "
            "the motor. It predicts the average torque and the torque ripple of each candidate design.</p>"
            "<h2>3 Results</h2><p>The CNN surrogate reduced the number of finite element evaluations by "
            "70 percent. The Pareto front was obtained four times faster than with the conventional "
            "method.</p>",
            "<h2>References</h2><p>[1] A. Author, Finite element analysis of machines, 2015.</p>"
            "<p>[2] B. Author, Genetic algorithms in engineering, 2018.</p>",
        ],
    },
    "muster_fehlererkennung_2019": {
        "title": "Fehlererkennung an Asynchronmaschinen mit maschinellem Lernen",
        "authors": ["Muster, Max"], "year": "2019", "venue": "Elektrotechnik und Informationstechnik",
        "doi": "", "volume": "", "number": "", "pages": "",
        "pages_html": [
            "<h1>Fehlererkennung an Asynchronmaschinen mit maschinellem Lernen</h1><p>Max Muster</p>"
            "<h2>Kurzfassung</h2><p>Diese Arbeit untersucht die Fehlererkennung an Asynchronmaschinen. "
            "Ein Random-Forest-Klassifikator erkennt Lagerschäden aus Stromsignalen.</p>",
            "<h2>2 Ergebnisse</h2><p>Der Klassifikator erreicht eine Genauigkeit von 97 Prozent. "
            "Stabbrüche werden zuverlässig von Lagerschäden unterschieden.</p>",
        ],
    },
    "smith_reinforcement_2020": {  # not in any export: becomes a paper of its own
        "title": "Reinforcement Learning for Motor Control: A Survey",
        "authors": ["Smith, Alice"], "year": "2020", "venue": "Control Engineering Practice",
        "doi": "10.5555/cep.2020.42", "volume": "", "number": "", "pages": "",
        "pages_html": [
            "<h1>Reinforcement Learning for Motor Control: A Survey</h1><p>Alice Smith</p>"
            "<p>https://doi.org/10.5555/cep.2020.42</p><h2>Abstract</h2><p>Reinforcement learning "
            "controllers are reviewed for field-oriented motor control. Policy gradient methods "
            "dominate recent work.</p>",
        ],
    },
}


def make_pdf(path: Path, pages_html: list[str], title: str = "", author: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for html in pages_html:
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(pymupdf.Rect(60, 60, 535, 800), html)
    doc.set_metadata({"title": title, "author": author})
    doc.save(path)
    doc.close()
    return path


def bib_entry(key: str, file_field: str | None = None, **override) -> str:
    p = {**PAPERS[key], **override}
    fields = {"title": p["title"], "author": " and ".join(p["authors"]), "date": p["year"],
              "journaltitle": p["venue"], "doi": p["doi"], "volume": p["volume"],
              "number": p["number"], "pages": p["pages"], "file": file_field}
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@article{{{key},\n{body}\n}}\n\n"


def ris_entry(key: str, link: str | None = None) -> str:
    p = PAPERS[key]
    lines = ["TY  - JOUR", f"TI  - {p['title']}", *[f"AU  - {a}" for a in p["authors"]],
             f"PY  - {p['year']}", f"T2  - {p['venue']}"]
    if p["doi"]:
        lines.append(f"DO  - https://doi.org/{p['doi']}")
    if link:
        lines.append(f"L1  - {link}")
    return "\n".join(lines + ["ER  - ", ""]) + "\n"


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory) -> dict[str, Path]:
    """The sample papers as PDFs, with file names that don't reveal the paper (as in Citavi)."""
    folder = tmp_path_factory.mktemp("pdfs")
    names = {"doe_topology_2021": "Doe2021.pdf", "muster_fehlererkennung_2019": "scan_0042.pdf",
             "smith_reinforcement_2020": "sub/download (3).pdf"}
    return {k: make_pdf(folder / names[k], PAPERS[k]["pages_html"]) for k in PAPERS}


# ------------------------------------------------------------------ isolation


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No test may reach the network (Crossref, arXiv); tests that need it patch it explicitly."""
    def blocked(*args, **kwargs):
        raise OSError("network disabled in tests")
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


FAKE_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://example.invalid/",
    "AZURE_OPENAI_API_KEY": "test-key-SECRET-0123456789",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "fake-embed",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "fake-chat",
    "AZURE_OPENAI_FAST_DEPLOYMENT": "fake-chat",
}


def patch_home(mp: pytest.MonkeyPatch, root: Path, chat: FakeChat, embeddings: FakeEmbeddings):
    """Point every module at `root` as installation home and swap in fake models."""
    from articles_agent import (config, graph, index, library, llm, pdfmeta, retriever, sources,
                                store)

    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(CODE_ROOT / "config.yaml", root / "config.yaml")
    mp.setattr(config, "PROJECT_ROOT", root)
    mp.setattr(llm, "PROJECT_ROOT", root)
    mp.setattr(library, "PROJECT_ROOT", root)
    mp.setattr(library, "LIBRARY_DIR", root / "library")
    mp.setattr(library, "REGISTRY", root / "library" / "extra_papers.json")
    mp.setattr(library, "UPLOADS", root / "library" / "uploads")
    mp.setattr(sources, "SOURCES_FILE", root / "library" / "sources.json")
    mp.setattr(sources, "EXPORTS_DIR", root / "library" / "exports")
    mp.setattr(pdfmeta, "CACHE_DIR", root / "library" / "cache")
    mp.setattr(pdfmeta, "_caches", {})
    for k, v in FAKE_ENV.items():
        mp.setenv(k, v)
    def get_embeddings():
        return embeddings

    def get_chat(fast=False):
        return chat

    for fn in (get_embeddings, get_chat):  # like the functools.cache originals
        fn.cache_clear = lambda: None
    mp.setattr(index, "get_embeddings", get_embeddings)
    mp.setattr(retriever, "get_embeddings", get_embeddings)
    mp.setattr(llm, "get_embeddings", get_embeddings)
    mp.setattr(graph, "get_chat", get_chat)
    mp.setattr(llm, "get_chat", get_chat)
    for fn in (store.parsed_pages, graph.get_retriever, graph.get_graph):
        fn.cache_clear()


@pytest.fixture
def fake_chat():
    return FakeChat()


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def workspace(tmp_path, monkeypatch, fake_chat, fake_embeddings) -> Path:
    """An empty installation: no libraries, no database."""
    root = tmp_path / "home"
    patch_home(monkeypatch, root, fake_chat, fake_embeddings)
    yield root
    from articles_agent import graph, store
    for fn in (store.parsed_pages, graph.get_retriever, graph.get_graph):
        fn.cache_clear()


def write_zotero_like_library(folder: Path, pdfs: dict[str, Path]) -> Path:
    """A BibLaTeX export: Doe linked via `file`, Muster without link (matched by title)."""
    bib = folder / "export.bib"
    bib.write_text(bib_entry("doe_topology_2021", f"Full Text PDF:{pdfs['doe_topology_2021']}:application/pdf")
                   + bib_entry("muster_fehlererkennung_2019"), encoding="utf-8")
    return bib


@pytest.fixture(scope="session")
def built_template(tmp_path_factory, sample_pdfs) -> Path:
    """One fully built installation (ingested + indexed with fake embeddings), copied per test."""
    from articles_agent.ingest import run_ingest
    from articles_agent.index import run_index
    from articles_agent.sources import add_source

    root = tmp_path_factory.mktemp("built") / "home"
    mp = pytest.MonkeyPatch()
    try:
        patch_home(mp, root, FakeChat(), FakeEmbeddings())
        bib = write_zotero_like_library(root.parent, sample_pdfs)
        add_source("zotero", str(bib), str(sample_pdfs["doe_topology_2021"].parent), use_crossref=False)
        run_ingest(progress=lambda *a: None)
        run_index(progress=lambda *a: None)
    finally:
        mp.undo()
    return root


@pytest.fixture
def built(workspace, built_template) -> Path:
    """An installation with a ready database of the three sample papers."""
    for name in ("library", "data"):
        shutil.copytree(built_template / name, workspace / name, dirs_exist_ok=True)
    return workspace


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
