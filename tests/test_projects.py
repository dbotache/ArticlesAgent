"""Projects: management, migration, isolation, shared caches, tag filters, cross-project search."""

import json
import shutil
import subprocess
import sys

import pytest

from articles_agent import config, graph, ingest
from articles_agent.config import NoProjectError, current_project, use_project
from articles_agent.graph import ask
from articles_agent.index import run_index
from articles_agent.ingest import run_ingest
from articles_agent.projects import (create_project, delete_project, get_project, list_projects,
                                     migrate_legacy, update_project)
from articles_agent.retriever import HybridRetriever
from articles_agent.sources import add_source, keyword_counts, read_source, Source
from conftest import CODE_ROOT, FAKE_ENV, PAPERS, bib_entry, ris_entry, write_zotero_like_library

DOE, MUSTER, SMITH = "doe_topology_2021", "muster_fehlererkennung_2019", "smith_reinforcement_2020"
quiet = {"progress": lambda *a: None}


# ------------------------------------------------------------------ management


def test_create_list_rename_delete(workspace):
    a = create_project("Motor design", "PhD chapter 3")
    b = create_project("Ölpumpen & Lager")
    assert (a.slug, b.slug) == ("motor-design", "olpumpen-lager")
    assert [p.slug for p in list_projects()] == ["test", "motor-design", "olpumpen-lager"]
    with pytest.raises(ValueError, match="already exists"):
        create_project("motor DESIGN")
    with pytest.raises(ValueError, match="needs a name"):
        create_project("  ")
    update_project("motor-design", name="Motor design (final)", description="")
    assert get_project("motor-design").name == "Motor design (final)"
    assert get_project("motor-design").slug == "motor-design"  # folder stays
    delete_project("olpumpen-lager")
    assert [p.slug for p in list_projects()] == ["test", "motor-design"]


def test_same_name_slug_gets_suffix(workspace):
    create_project("Motor-Design")
    assert create_project("Motor design!").slug == "motor-design-2"


def test_active_project_resolution(workspace, monkeypatch):
    create_project("Second")
    assert current_project() == "test"  # set by the fixture
    with use_project("second"):
        assert current_project() == "second"
        assert config.load_config()["data_dir"] == config.projects_dir() / "second" / "data"
    config.set_project(None)
    try:
        monkeypatch.setenv("ARTICLES_AGENT_PROJECT", "second")
        assert current_project() == "second"
        monkeypatch.delenv("ARTICLES_AGENT_PROJECT")
        assert current_project() == "test"  # first created
    finally:
        config.set_project("test")


def test_no_project_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "APP_HOME", tmp_path)
    with use_project(None):
        monkeypatch.delenv("ARTICLES_AGENT_PROJECT", raising=False)
        with pytest.raises(NoProjectError):
            current_project()


def test_projects_cli(tmp_path):
    env = {k: v for k, v in __import__("os").environ.items()} | {"ARTICLES_AGENT_HOME": str(tmp_path)}
    run = lambda *a: subprocess.run([sys.executable, "-m", "articles_agent.projects", *a], env=env,
                                    capture_output=True, text=True, check=True).stdout
    run("create", "Battery study", "--description", "Ageing")
    assert "battery-study" in run("list") and "Ageing" in run("list")
    run("delete", "battery-study")
    assert run("list") == ""


# ------------------------------------------------------------------ migration


def test_legacy_installation_becomes_first_project(built_template, tmp_path, monkeypatch):
    """The pre-projects layout (library/, data/ in the app home) is moved into a project, the caches
    to shared/, and the index's vectors seed the shared embedding cache."""
    home = tmp_path / "legacy"
    src = built_template / "projects" / "test"
    shutil.copytree(src / "library", home / "library")
    shutil.copytree(src / "data", home / "data")
    shutil.copytree(built_template / "shared" / "parsed", home / "data" / "parsed")
    shutil.copytree(built_template / "shared" / "pdfmeta", home / "library" / "cache")
    monkeypatch.setattr(config, "APP_HOME", home)
    for k, v in FAKE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("ARTICLES_AGENT_PROJECT", raising=False)
    with use_project(None):
        (project,) = list_projects()
    assert (project.slug, project.name) == ("my-library", "My library")
    assert not (home / "library").exists() and not (home / "data").exists()
    assert (project.dir / "library" / "sources.json").exists()
    assert (project.dir / "data" / "chroma").is_dir()
    assert (home / "shared" / "parsed").is_dir() and (home / "shared" / "pdfmeta").is_dir()
    from articles_agent import embcache
    with use_project(project.slug):
        n = len(HybridRetriever().chunks)
    import sqlite3
    count = sqlite3.connect(home / "shared" / "embeddings.sqlite").execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    assert count == n
    assert migrate_legacy() is None  # runs only once


# ------------------------------------------------------------------ isolation and shared caches


def second_project_from_same_library(sample_pdfs, tmp_path, keywords=()):
    project = create_project("Second")
    bib = write_zotero_like_library(tmp_path, sample_pdfs)
    with use_project(project.slug):
        add_source("zotero", str(bib), str(sample_pdfs[DOE].parent), use_crossref=False, keywords=list(keywords))
    return project


def test_second_project_reuses_parsing_and_embeddings(built, sample_pdfs, tmp_path, monkeypatch,
                                                      fake_embeddings):
    project = second_project_from_same_library(sample_pdfs, tmp_path)
    monkeypatch.setattr(ingest, "_parse_to_cache", lambda *a: pytest.fail("PDF parsed twice"))
    with use_project(project.slug):
        assert run_ingest(**quiet)["parsed"] == 0  # shared page cache
        report = run_index(**quiet)
    assert report["added"] > 0 and report["reused"] == report["added"]
    assert report["embedded"] == 0 and fake_embeddings.calls == 0  # no Azure call at all


def test_projects_are_independent(built, sample_pdfs, tmp_path):
    other = create_project("Other")
    bib = tmp_path / "other.bib"
    bib.write_text(bib_entry(MUSTER, f"PDF:{sample_pdfs[MUSTER]}:application/pdf"))
    with use_project(other.slug):
        add_source("citavi", str(bib))
        run_ingest(**quiet)
        run_index(**quiet)
        other_papers = {c["paper_id"] for c in HybridRetriever().chunks}
    with use_project("test"):
        test_papers = {c["paper_id"] for c in HybridRetriever().chunks}
    assert other_papers == {MUSTER}
    assert {DOE, MUSTER} <= test_papers
    # removing the library from "other" leaves "test" untouched
    from articles_agent.sources import remove_source
    with use_project(other.slug):
        remove_source("citavi")
        run_ingest(**quiet)
        assert run_index(**quiet)["total"] == 0
    with use_project("test"):
        assert {c["paper_id"] for c in HybridRetriever().chunks} == test_papers


# ------------------------------------------------------------------ keyword / tag filter


def test_keywords_are_read_from_every_format(tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(bib_entry(DOE, keywords="motor-design, CNN") +
                   "@article{j, title={JabRef paper}, groups={motor-design}, year={2020}}\n"
                   "@article{m, title={Mendeley paper}, mendeley-tags={PMSM;cnn}, year={2020}}\n")
    from articles_agent.metadata import read_bibtex, read_ris
    papers, _ = read_bibtex(bib)
    assert [p.keywords for p in papers] == [["motor-design", "CNN"], ["motor-design"], ["PMSM", "cnn"]]
    ris = tmp_path / "x.ris"
    ris.write_text(ris_entry(MUSTER, keywords=("Fehlerdiagnose", "motor-design")))
    assert read_ris(ris)[0].keywords == ["Fehlerdiagnose", "motor-design"]
    assert keyword_counts(str(bib)) == {"CNN": 2, "motor-design": 2, "PMSM": 1}  # "cnn" counts as "CNN"


def test_keyword_filter_keeps_only_tagged_references(workspace, sample_pdfs, tmp_path):
    bib = tmp_path / "all.bib"
    bib.write_text(bib_entry(DOE, keywords="motor-design") + bib_entry(MUSTER, keywords="diagnosis"))
    folder = sample_pdfs[DOE].parent
    src = Source("zotero", "zotero", str(bib), str(folder), use_crossref=False, keywords=["Motor-Design"])
    rep = read_source(src)
    assert [p.paper_id for p in rep.papers] == [DOE]
    assert rep.filtered_out == 1
    # the excluded reference's PDF was still matched, so it doesn't show up as a paper of its own;
    # untagged PDFs without a reference (Smith) are left out while a filter is set
    assert rep.matched == 2 and rep.standalone == 0
    assert all(p.in_bib for p in rep.papers)


def test_tag_filtered_project(built, sample_pdfs, tmp_path):
    bib = tmp_path / "tagged.bib"
    bib.write_text(bib_entry(DOE, f"PDF:{sample_pdfs[DOE]}:application/pdf", keywords="cnn")
                   + bib_entry(MUSTER, f"PDF:{sample_pdfs[MUSTER]}:application/pdf", keywords="diagnosis"))
    project = create_project("CNN only")
    with use_project(project.slug):
        add_source("zotero", str(bib), keywords=["cnn"])
        run_ingest(**quiet)
        run_index(**quiet)
        assert {c["paper_id"] for c in HybridRetriever().chunks} == {DOE}
        from articles_agent.sources import set_keywords
        set_keywords("zotero", [])  # whole library
        run_ingest(**quiet)
        run_index(**quiet)
        assert {c["paper_id"] for c in HybridRetriever().chunks} == {DOE, MUSTER}


# ------------------------------------------------------------------ cross-project search


def test_search_across_projects(built, sample_pdfs, tmp_path):
    """'test' has Doe+Muster+Smith, 'other' has Muster (again) and a paper only it contains."""
    from conftest import make_pdf
    battery = make_pdf(tmp_path / "battery.pdf", [
        "<h1>Battery Ageing in Electric Vehicles</h1><h2>Abstract</h2><p>The convolutional neural "
        "network surrogate predicts capacity fade of lithium-ion cells. Finite element evaluations "
        "of the thermal model were reduced as well.</p>"])
    other = create_project("Other")
    bib = tmp_path / "other.bib"
    bib.write_text(bib_entry(MUSTER, f"PDF:{sample_pdfs[MUSTER]}:application/pdf")
                   + "@article{battery_2022, title={Battery Ageing in Electric Vehicles}, author={Lee, Kim},"
                     f" year={{2022}}, file={{PDF:{battery}:application/pdf}}}}\n")
    with use_project(other.slug):
        add_source("zotero", str(bib))
        run_ingest(**quiet)
        run_index(**quiet)
    res = ask("convolutional neural network surrogate finite element", projects=["test", other.slug])
    projects_hit = {f.project for f in res.findings}
    assert projects_hit <= {"test", other.slug} and res.findings
    assert all("::" in f.paper_id for f in res.findings)  # namespaced in cross-project mode
    assert all(f.pdf_path and f.passage for f in res.findings)
    state = graph.retrieve({"queries": ["Klassifikator Lagerschäden"], "projects": ["test", other.slug]})
    muster_ids = {h.paper["paper_id"] for h in state["candidates"] if h.paper["title"] == PAPERS[MUSTER]["title"]}
    assert len(muster_ids) == 1  # the paper in both projects keeps one id → cited once
    texts = [h.chunk["text"] for h in state["candidates"]]
    assert len(texts) == len(set(texts))  # identical passages from both projects count once


def test_single_project_answers_stay_inside_it(built):
    create_project("Other")
    res = ask("convolutional neural network", projects=["test"])
    assert {f.project for f in res.findings} == {"test"}
    assert all("::" not in f.paper_id for f in res.findings)
