"""Reading reference-manager exports (BibTeX/BibLaTeX, RIS)."""

import pytest

from articles_agent.metadata import (Paper, clean_doi, make_id, norm_doi, norm_title,
                                     parse_file_field, read_bibtex, read_ris)


@pytest.mark.parametrize("field, expected", [
    # Zotero (Linux)
    ("Full Text PDF:/home/u/Zotero/storage/AB12CD34/Doe 2021.pdf:application/pdf",
     ["/home/u/Zotero/storage/AB12CD34/Doe 2021.pdf"]),
    # Zotero (Windows, escaped colon and backslashes)
    (r"Full Text PDF:C\:\\Users\\u\\Zotero\\storage\\K\\x.pdf:application/pdf",
     [r"C:\Users\u\Zotero\storage\K\x.pdf"]),
    # JabRef, relative path
    (":papers/Doe2021.pdf:PDF", ["papers/Doe2021.pdf"]),
    # Mendeley Desktop (Linux drops the leading slash, Windows uses $\backslash$)
    (":home/u/Mendeley/Doe2021.pdf:pdf", ["home/u/Mendeley/Doe2021.pdf"]),
    (r":C$\backslash$:/Users/u/Doe2021.pdf:pdf", ["C:/Users/u/Doe2021.pdf"]),
    # several attachments, HTML snapshot ignored
    ("Snapshot:/s/a.html:text/html;Full Text:/s/b.pdf:application/pdf;Other:/s/c.PDF:application/pdf",
     ["/s/b.pdf", "/s/c.PDF"]),
    ("", []),
])
def test_parse_file_field(field, expected):
    assert parse_file_field(field) == expected


def test_doi_helpers():
    assert clean_doi("https://doi.org/10.1109/TMAG.2021.1") == "10.1109/TMAG.2021.1"
    assert clean_doi("doi: 10.1/x.") == "10.1/x"
    assert norm_doi("https://dx.doi.org/10.1109/TMAG.2021.1") == "10.1109/tmag.2021.1"


def test_make_id_is_unique_and_keeps_digits():
    taken = set()
    a = make_id("3D Topology Optimization", ["Banga, Saurabh"], "", taken)
    taken.add(a)
    b = make_id("3D Topology Optimization", ["Banga, Saurabh"], "", taken)
    assert a == "banga_3d_nodate"
    assert b == "banga_3d_nodate_2"
    assert make_id("The Use of AI", [], "2020", set()) == "nn_use_2020"


def test_norm_title_ignores_case_and_punctuation():
    assert norm_title("Deep Learning: A Survey!") == norm_title("deep learning a survey")


def test_short_cite():
    assert Paper("x", "T", ["Doe, J."], "2021").short_cite == "Doe, 2021"
    assert Paper("x", "T", ["Doe, J.", "Roe, R."], "2021").short_cite == "Doe & Roe, 2021"
    assert Paper("x", "T", ["Doe, J.", "Roe, R.", "Poe, P."], "").short_cite == "Doe et al., n.d."


def test_read_bibtex_biblatex_fields_and_jabref_dir(tmp_path):
    bib = tmp_path / "lib.bib"
    bib.write_text(r"""
@article{doe_2021,
  title = {Topology {Optimization} with {CNNs}},
  author = {Doe, Jane and M{\"u}ller, Hans},
  date = {2021-06},
  journaltitle = {IEEE Transactions on Magnetics},
  doi = {https://doi.org/10.1/ABC},
  pages = {1--4},
  file = {:sub/doe.pdf:PDF},
}
@inproceedings{roe_2019, title = {A Conference Paper}, author = {Roe, R.}, year = {2019},
  booktitle = {Proc. ICEM}}
@Comment{jabref-meta: fileDirectory:/data/papers;}
""", encoding="utf-8")
    papers, dirs = read_bibtex(bib)
    doe, roe = papers
    assert doe.paper_id == "doe_2021"
    assert doe.title == "Topology Optimization with CNNs"
    assert doe.authors == ["Doe, Jane", "Müller, Hans"]
    assert (doe.year, doe.venue, doe.doi, doe.pages) == ("2021", "IEEE Transactions on Magnetics", "10.1/ABC", "1–4")
    assert doe.pdf_paths == ["sub/doe.pdf"]
    assert (roe.entry_type, roe.venue, roe.year) == ("inproceedings", "Proc. ICEM", "2019")
    assert dirs == ["/data/papers"]


def test_read_bibtex_windows_encoding(tmp_path):
    bib = tmp_path / "citavi.bib"
    bib.write_bytes("@article{k, title = {Lagerschäden erkennen}, author = {Weiß, Jörg}, year = {2020}}"
                    .encode("cp1252"))
    (paper,), _ = read_bibtex(bib)
    assert paper.title == "Lagerschäden erkennen"
    assert paper.authors == ["Weiß, Jörg"]


def test_read_ris(tmp_path):
    ris = tmp_path / "endnote.ris"
    ris.write_text("""TY  - JOUR
TI  - Topology Optimization with CNNs
AU  - Doe, Jane
AU  - Roe, Richard
PY  - 2021///
T2  - IEEE Transactions on Magnetics
VL  - 57
IS  - 6
SP  - 1
EP  - 4
DO  - https://doi.org/10.1/ABC
UR  - https://example.org/paper
L1  - internal-pdf://0123456789/Doe2021.pdf
ER  -

TY  - CONF
T1  - Second
A1  - Poe, P.
Y1  - 2018
L1  - file:///home/u/papers/Second%20Paper.pdf
ER  -
""", encoding="utf-8")
    first, second = read_ris(ris)
    assert first.paper_id == ""  # assigned later by sources.py
    assert (first.title, first.year, first.venue) == ("Topology Optimization with CNNs", "2021",
                                                       "IEEE Transactions on Magnetics")
    assert first.authors == ["Doe, Jane", "Roe, Richard"]
    assert (first.volume, first.number, first.pages, first.doi) == ("57", "6", "1–4", "10.1/ABC")
    assert first.url == "https://example.org/paper"
    assert first.pdf_paths == ["Doe2021.pdf"]  # EndNote internal link → resolved by file name
    assert second.entry_type == "inproceedings"
    assert second.pdf_paths == ["/home/u/papers/Second Paper.pdf"]
