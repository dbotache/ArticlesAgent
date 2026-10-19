"""Individually added PDFs, and the database update in a separate process."""

import shutil

import pytest

from articles_agent import library
from articles_agent.library import (collect_pdfs, load_registry, register_pdfs, save_upload,
                                    update_database_isolated)
from conftest import PAPERS


def test_collect_and_upload(workspace, tmp_path, sample_pdfs):
    folder = tmp_path / "in"
    (folder / "deep").mkdir(parents=True)
    shutil.copy(sample_pdfs["doe_topology_2021"], folder / "deep" / "A.PDF")
    (folder / "notes.txt").write_text("x")
    assert [p.name for p in collect_pdfs(folder)] == ["A.PDF"]
    with pytest.raises(FileNotFoundError):
        collect_pdfs(tmp_path / "missing")
    data = sample_pdfs["smith_reinforcement_2020"].read_bytes()
    first, again = save_upload("../../evil/name.pdf", data), save_upload("name.pdf", data)
    assert first == again  # same content, same place
    assert first.is_relative_to(library.uploads_dir())  # no path traversal from the browser file name


def test_register_skips_duplicates(built, tmp_path, sample_pdfs):
    folder = tmp_path / "new"
    folder.mkdir()
    shutil.copy(sample_pdfs["doe_topology_2021"], folder / "copy_of_doe.pdf")      # same title/DOI
    shutil.copy(sample_pdfs["smith_reinforcement_2020"], folder / "smith.pdf")     # already own paper
    pdfs = collect_pdfs(folder) + [sample_pdfs["doe_topology_2021"]]               # same path
    results = {r["file"]: r for r in register_pdfs(pdfs, use_crossref=False)}
    assert results["copy_of_doe.pdf"]["status"] == "skipped"
    assert results["Doe2021.pdf"]["detail"] == "already in the library"
    assert results["smith.pdf"]["status"] == "skipped"
    assert load_registry() == {}


def test_register_adds_new_paper(built, tmp_path):
    from conftest import make_pdf
    pdf = make_pdf(tmp_path / "new.pdf", ["<h1>Battery Ageing Models for Electric Vehicles</h1>"
                                          "<p>Lithium-ion cells degrade with cycling.</p>"])
    (r,) = register_pdfs([pdf], use_crossref=False)
    assert r["status"] == "added"
    reg = load_registry()
    entry = reg[r["paper_id"]]
    assert entry["title"] == "Battery Ageing Models for Electric Vehicles"
    assert entry["metadata_source"] == "first page" and entry["source"] == "import"
    (again,) = register_pdfs([pdf], use_crossref=False)
    assert again["status"] == "skipped"


def test_register_reports_unreadable_file(workspace, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf")
    (r,) = register_pdfs([bad], use_crossref=False)
    assert r["status"] == "error"


@pytest.mark.integration
def test_isolated_update_runs_in_subprocess(workspace):
    """The app runs updates in a child process (Streamlit can't host the parser's workers).
    With an empty library this needs no models, so it runs offline."""
    events = []
    report = update_database_isolated(progress=lambda *a: events.append(a))
    assert report["ingest"]["pdfs"] == 0
    assert report["index"]["total"] == 0
    assert (workspace / "data" / "chunks.jsonl").exists()  # written to the data home, not the code dir


@pytest.mark.integration
def test_isolated_update_reports_failure(workspace):
    (workspace / "library").mkdir(exist_ok=True)
    (workspace / "library" / "sources.json").write_text("{ broken json")
    with pytest.raises(RuntimeError, match="Database update failed"):
        update_database_isolated()


def test_sample_titles_are_distinct():  # guards the fixtures used above
    assert len({p["title"] for p in PAPERS.values()}) == len(PAPERS)
