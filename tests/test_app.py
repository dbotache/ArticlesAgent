"""The Streamlit app, driven headlessly with AppTest (fake models, temporary installation)."""

import time
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from articles_agent import graph, jobs, ui_library
from articles_agent.library import update_database
from conftest import CODE_ROOT, FAKE_ENV, bib_entry
from fakes import FakeChat, default_responder

APP = str(CODE_ROOT / "app.py")


@pytest.fixture(autouse=True)
def fresh_app(monkeypatch):
    """No cached index or finished job leaks between tests; updates run in-process so the
    fake models apply (the subprocess variant is tested in test_library.py)."""
    st.cache_resource.clear()
    monkeypatch.setattr(jobs, "_current", None)
    monkeypatch.setattr(ui_library, "update_database_isolated", update_database)
    yield
    st.cache_resource.clear()


def run_app(**kw) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def wait_for_job(at: AppTest, timeout=120) -> AppTest:
    t = time.time()
    while (job := jobs.current()) and job.status == "running" and time.time() - t < timeout:
        time.sleep(0.2)
    assert jobs.current().status == "done", (jobs.current().error, jobs.current().log)
    at.run()
    at.run()  # the finished job triggers a reload of the index
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(at: AppTest) -> list[str]:
    """All visible text of the page (sidebar included)."""
    out = []
    for kind in ("markdown", "caption", "info", "success", "warning", "error", "title", "subheader"):
        out += [str(e.value) for e in getattr(at, kind)]
    out += [str(e.label) for e in at.expander] + [str(w.value) for w in at.text_input]
    return out


def by_label(widgets, label_part):
    return next(w for w in widgets if label_part in w.label)


# ------------------------------------------------------------------ setup page


def test_setup_page_asks_for_azure_first(workspace, monkeypatch):
    for key in FAKE_ENV:
        monkeypatch.delenv(key)
    at = run_app()
    assert at.title[0].value == "📚 Set up ArticlesAgent"
    assert "Set up the Azure connection first." in texts(at)
    assert by_label(at.text_input, "API key").value == ""


def test_setup_saves_and_tests_azure_settings(workspace, monkeypatch):
    for key in FAKE_ENV:
        monkeypatch.delenv(key)
    at = run_app()
    by_label(at.text_input, "Endpoint").set_value("https://res.openai.azure.com/")
    by_label(at.text_input, "API key").set_value("new-secret-key")
    by_label(at.text_input, "Embedding deployment").set_value("fake-embed")
    by_label(at.text_input, "Chat deployment").set_value("fake-chat")
    at.button[0].click().run()  # "Save and test connection"
    assert not at.exception
    assert "AZURE_OPENAI_API_KEY=new-secret-key" in (workspace / ".env").read_text()
    assert len(at.success) == 3  # embeddings, chat, fast model
    at.run()  # the form clears on submit; AppTest shows that one run later than a browser
    assert "new-secret-key" not in " ".join(texts(at))


def test_setup_adds_library_builds_database_and_opens_chat(workspace, tmp_path, sample_pdfs):
    bib = tmp_path / "export.bib"
    bib.write_text(bib_entry("doe_topology_2021") + bib_entry("muster_fehlererkennung_2019"))
    at = run_app()
    at.selectbox(key="setup_mgr").set_value("citavi").run()
    at.text_input(key="setup_ref").set_value(str(bib)).run()
    at.text_input(key="setup_pdf_citavi").set_value(str(sample_pdfs["doe_topology_2021"].parent)).run()
    at.button(key="setup_check").click().run()
    preview = " ".join(texts(at))
    assert "**2** references in the export" in preview and "**3** PDFs in the folder" in preview
    at.button(key="setup_go").click().run()
    at = wait_for_job(at)
    assert at.sidebar.title[0].value == "📚 ArticlesAgent"
    assert at.sidebar.caption[0].value.startswith("3 papers")
    assert "Adding Citavi · export.bib" in " ".join(texts(at))


def test_setup_rejects_bad_paths(workspace, tmp_path):
    at = run_app()
    at.text_input(key="setup_ref").set_value(str(tmp_path / "missing.bib")).run()
    assert any("Reference file not found" in e.value for e in at.error)
    assert at.button(key="setup_go").disabled


# ------------------------------------------------------------------ chat


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question).run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_chat_answers_with_cited_findings(built):
    at = ask(run_app(), "How do CNN surrogates speed up topology optimization?")
    captions = [c.value for c in at.caption]
    cited = [c for c in captions if c.startswith("(") and "section" in c]
    assert len(cited) == 2
    assert any("1 finding(s) omitted" in c for c in captions)
    md = " ".join(m.value for m in at.markdown)
    assert "#### References" in md and "Doe, J., & Roe, R. (2021)" in md
    assert {b.label for b in at.get("download_button")} == {"⬇ Download session (Markdown)",
                                                            "⬇ Download answer (Markdown)"}
    assert any(b.label == "Open PDF" for b in at.button)  # sample PDFs are not in Zotero storage


def test_dollar_signs_are_not_rendered_as_latex(built, monkeypatch):
    def pricey(schema, payload):
        out = default_responder(schema, payload)
        if schema is graph.Answer:
            out.summary = "It costs $5 and $10 [S1]."
        return out
    monkeypatch.setattr(graph, "get_chat", lambda fast=False: FakeChat(pricey))
    at = ask(run_app(), "price?")
    assert any(r"\$5 and \$10" in m.value for m in at.markdown)


def test_chat_shows_backend_errors(built, monkeypatch):
    def broken(schema, payload):
        raise RuntimeError("429 Too Many Requests")
    monkeypatch.setattr(graph, "get_chat", lambda fast=False: FakeChat(broken))
    at = ask(run_app(), "anything")
    assert any("429 Too Many Requests" in e.value for e in at.error)


def test_year_and_paper_filters(built):
    at = run_app()
    assert at.sidebar.slider[0].value == (2019, 2021)
    at.sidebar.multiselect[0].select("muster_fehlererkennung_2019").run()
    at = ask(at, "Klassifikator")
    assert all("Muster, 2019" in c.value for c in at.caption if c.value.startswith("(Muster") or
               c.value.startswith("(Doe")) and not any(c.value.startswith("(Doe") for c in at.caption)


# ------------------------------------------------------------------ library management


def test_add_single_papers_from_folder(built, tmp_path):
    from conftest import make_pdf
    make_pdf(tmp_path / "new" / "battery.pdf", ["<h1>Battery Ageing Models for Electric Vehicles</h1>"
                                                "<h2>Abstract</h2><p>Lithium-ion cells degrade with "
                                                "cycling and calendar ageing. We compare empirical and "
                                                "electrochemical ageing models for vehicle batteries.</p>"])
    at = run_app()
    by_label(at.sidebar.text_input, "and/or a folder").set_value(str(tmp_path / "new")).run()
    by_label(at.sidebar.button, "Add to library").click().run()
    at = wait_for_job(at)
    assert at.sidebar.caption[0].value.startswith("4 papers")
    assert any("✅ **battery.pdf**" in m.value for m in at.sidebar.markdown)


def test_remove_last_library_returns_to_setup(built):
    at = run_app()
    at.sidebar.button(key="rm_zotero").click().run()
    at = wait_for_job(at)
    assert at.title[0].value == "📚 Set up ArticlesAgent"
