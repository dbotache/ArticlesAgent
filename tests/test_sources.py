"""Library sources: PDF link resolution and matching for every supported reference manager."""

import shutil

import pytest

from articles_agent import sources
from articles_agent.sources import (Source, add_source, build_library, load_sources, read_source,
                                    remove_source, validate)
from conftest import PAPERS, bib_entry, ris_entry

DOE, MUSTER, SMITH = "doe_topology_2021", "muster_fehlererkennung_2019", "smith_reinforcement_2020"


@pytest.fixture
def lib(workspace, tmp_path, sample_pdfs):
    """A PDF folder like a reference manager's attachments folder (copied, renamed)."""
    folder = tmp_path / "attachments"
    (folder / "sub").mkdir(parents=True)
    shutil.copy(sample_pdfs[DOE], folder / "Doe2021.pdf")
    shutil.copy(sample_pdfs[MUSTER], folder / "sub" / "scan_0042.pdf")
    shutil.copy(sample_pdfs[SMITH], folder / "unrelated.pdf")
    return folder


def src(manager, ref, pdf_dir=""):
    return Source(id=manager, manager=manager, reference_file=str(ref), pdf_dir=str(pdf_dir))


def summary(report):
    by_id = {p.paper_id: p for p in report.papers}
    return {pid: [p.name for p in map(__import__("pathlib").Path, by_id[pid].pdf_paths)] for pid in by_id}


def test_zotero_absolute_links(lib, tmp_path):
    bib = tmp_path / "zotero.bib"
    bib.write_text(bib_entry(DOE, f"Full Text PDF:{lib / 'Doe2021.pdf'}:application/pdf") + bib_entry(MUSTER))
    rep = read_source(src("zotero", bib, lib))
    assert (rep.references, rep.linked, rep.matched, rep.standalone) == (2, 1, 1, 1)
    assert summary(rep)[DOE] == ["Doe2021.pdf"]
    assert summary(rep)[MUSTER] == ["scan_0042.pdf"]  # matched by the title on page 1


def test_jabref_relative_links_with_file_directory(lib, tmp_path):
    bib = tmp_path / "jabref.bib"
    bib.write_text(bib_entry(DOE, ":Doe2021.pdf:PDF") + bib_entry(MUSTER, ":sub/scan_0042.pdf:PDF")
                   + f"@Comment{{jabref-meta: fileDirectory:{lib};}}\n")
    rep = read_source(src("jabref", bib))  # no PDF folder given: the library's own setting is used
    assert (rep.linked, rep.broken_links) == (2, 0)


def test_mendeley_and_windows_paths_found_by_name(lib, tmp_path):
    bib = tmp_path / "mendeley.bib"
    bib.write_text(bib_entry(DOE, f":{str(lib / 'Doe2021.pdf')[1:]}:pdf")
                   + bib_entry(MUSTER, r":C$\backslash$:/Users/someone/Mendeley/scan_0042.pdf:pdf"))
    rep = read_source(src("mendeley", bib, lib))
    assert rep.linked == 2
    assert summary(rep)[MUSTER] == ["scan_0042.pdf"]


def test_citavi_without_links_matches_by_doi_and_title(lib, tmp_path):
    bib = tmp_path / "citavi.bib"
    bib.write_text(bib_entry(DOE) + bib_entry(MUSTER))
    rep = read_source(src("citavi", bib, lib))
    assert (rep.linked, rep.matched, rep.with_pdf) == (0, 2, 2)


def test_endnote_ris(lib, tmp_path):
    ris = tmp_path / "endnote.ris"
    ris.write_text(ris_entry(DOE, "internal-pdf://0123/Doe2021.pdf") + ris_entry(MUSTER))
    rep = read_source(src("endnote", ris, lib))
    ids = {p.paper_id for p in rep.papers}
    assert {"doe_topology_2021", "muster_fehlererkennung_2019"} <= ids  # generated keys
    assert (rep.linked, rep.matched) == (1, 1)


def test_unmatched_pdf_becomes_own_paper(lib, tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(bib_entry(DOE))
    rep = read_source(src("other", bib, lib))
    own = [p for p in rep.papers if not p.in_bib]
    assert {p.title for p in own} == {PAPERS[SMITH]["title"], PAPERS[MUSTER]["title"]}
    assert all(p.source == "other" for p in rep.papers)


def test_copy_of_linked_pdf_is_a_duplicate(lib, tmp_path):
    shutil.copy(lib / "Doe2021.pdf", lib / "Doe2021 (copy).pdf")
    bib = tmp_path / "x.bib"
    bib.write_text(bib_entry(DOE, f"PDF:{lib / 'Doe2021.pdf'}:application/pdf"))
    rep = read_source(src("other", bib, lib))
    assert rep.duplicates == 1
    assert summary(rep)[DOE] == ["Doe2021.pdf"]


def test_preview_opens_no_pdf(lib, tmp_path, monkeypatch):
    bib = tmp_path / "citavi.bib"
    bib.write_text(bib_entry(DOE) + bib_entry(MUSTER))
    monkeypatch.setattr(sources, "probe_pdf", lambda *a: pytest.fail("PDF opened in preview"))
    rep = read_source(src("citavi", bib, lib), scan_folder=False)
    assert (rep.references, rep.pdfs_in_folder, rep.unlinked_pdfs) == (2, 3, 3)


def test_add_validate_remove_sources(workspace, lib, tmp_path):
    bib = tmp_path / "a.bib"
    bib.write_text(bib_entry(DOE))
    assert validate(str(tmp_path / "missing.bib"))
    assert validate(str(bib), str(tmp_path / "nope"))
    assert validate(str(lib / "Doe2021.pdf"))  # not an export
    assert validate(str(bib), str(lib)) == []
    s1 = add_source("zotero", str(bib), str(lib))
    with pytest.raises(ValueError, match="already"):
        add_source("zotero", str(bib))
    bib2 = tmp_path / "b.ris"
    bib2.write_text(ris_entry(MUSTER))
    s2 = add_source("zotero", str(bib2))
    assert (s1.id, s2.id) == ("zotero", "zotero2")
    assert [s.format for s in load_sources()] == ["bibtex", "ris"]
    remove_source("zotero")
    assert [s.id for s in load_sources()] == ["zotero2"]


def test_build_library_merges_sources_and_duplicates(workspace, lib, tmp_path):
    a = tmp_path / "a.bib"
    a.write_text(bib_entry(DOE) + bib_entry(MUSTER))  # no links in the first library
    b = tmp_path / "b.bib"
    b.write_text(bib_entry(DOE, f"PDF:{lib / 'Doe2021.pdf'}:application/pdf", title="Different title"))
    add_source("citavi", str(a))
    add_source("zotero", str(b))
    papers = build_library()
    assert sorted(papers) == [DOE, MUSTER]  # DOE kept once (same DOI) …
    assert [p.split("/")[-1] for p in papers[DOE].pdf_paths] == ["Doe2021.pdf"]  # … with the PDF


def test_paper_id_collision_between_sources(workspace, tmp_path):
    a, b = tmp_path / "a.bib", tmp_path / "b.bib"
    a.write_text("@article{key1, title = {First paper about electric machines}, year = {2020}}")
    b.write_text("@article{key1, title = {Second unrelated paper on batteries}, year = {2021}}")
    add_source("zotero", str(a))
    add_source("jabref", str(b))
    assert sorted(build_library()) == ["key1", "key1_jabref"]
