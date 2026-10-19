"""Facts read from PDFs and the Crossref / arXiv lookups (network mocked)."""

import io
import json

import pytest

from articles_agent import pdfmeta
from conftest import PAPERS, make_pdf

CROSSREF_RECORD = {
    "title": ["Reinforcement Learning for Motor Control: A Survey"], "DOI": "10.5555/CEP.2020.42",
    "author": [{"family": "Smith", "given": "Alice"}], "issued": {"date-parts": [[2020, 5]]},
    "container-title": ["Control Engineering Practice"], "type": "journal-article",
    "volume": "99", "issue": "2", "page": "10-20", "publisher": "Elsevier", "URL": "https://doi.org/x",
}


@pytest.fixture
def smith_pdf(workspace, sample_pdfs):
    return sample_pdfs["smith_reinforcement_2020"]


def fake_urlopen(responses: dict):
    """urlopen replacement answering by URL substring; counts calls."""
    calls = []

    def urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        calls.append(url)
        for key, body in responses.items():
            if key in url:
                return io.BytesIO(body.encode() if isinstance(body, str) else body)
        raise OSError("unexpected URL " + url)
    urlopen.calls = calls
    return urlopen


def test_probe_finds_doi_title_and_text(smith_pdf):
    probe = pdfmeta.probe_pdf(smith_pdf)
    assert probe["dois"] == ["10.5555/cep.2020.42"]
    assert probe["layout_title"] == PAPERS["smith_reinforcement_2020"]["title"]
    assert "policy gradient methods" in probe["head"]


def test_probe_is_cached(smith_pdf, monkeypatch):
    pdfmeta.probe_pdf(smith_pdf)
    monkeypatch.setattr(pdfmeta.pymupdf, "open", lambda *a: (_ for _ in ()).throw(AssertionError("re-opened")))
    assert pdfmeta.probe_pdf(smith_pdf)["dois"]


def test_metadata_from_crossref(smith_pdf, monkeypatch):
    urlopen = fake_urlopen({"api.crossref.org": json.dumps({"message": CROSSREF_RECORD})})
    monkeypatch.setattr(pdfmeta.urllib.request, "urlopen", urlopen)
    info, how = pdfmeta.metadata_from_pdf(smith_pdf)
    assert how == "crossref"
    assert info["authors"] == ["Smith, Alice"]
    assert (info["year"], info["venue"], info["entry_type"], info["pages"]) == (
        "2020", "Control Engineering Practice", "article", "10–20")
    assert info["doi"] == "10.5555/CEP.2020.42"  # case kept for display
    # second call answered from the cache
    pdfmeta.metadata_from_pdf(smith_pdf)
    assert len(urlopen.calls) == 1
    assert len(urlopen.calls[0]) and "10.5555" in urlopen.calls[0]  # only the DOI is sent


def test_crossref_record_of_cited_reference_is_rejected(smith_pdf, monkeypatch):
    """A DOI on page 1 that belongs to another paper must not become this paper's metadata."""
    other = {**CROSSREF_RECORD, "title": ["A Totally Different Paper on Bread Baking"]}
    monkeypatch.setattr(pdfmeta.urllib.request, "urlopen",
                        fake_urlopen({"api.crossref.org": json.dumps({"message": other})}))
    info, how = pdfmeta.metadata_from_pdf(smith_pdf)
    assert how == "first page"
    assert info["title"] == PAPERS["smith_reinforcement_2020"]["title"]
    assert info["year"] == ""


def test_offline_falls_back_without_caching(smith_pdf):
    info, how = pdfmeta.metadata_from_pdf(smith_pdf)  # network is blocked by the `offline` fixture
    assert how == "first page"
    assert pdfmeta._cache("crossref") == {}


def test_no_crossref_when_disabled(smith_pdf, monkeypatch):
    urlopen = fake_urlopen({})
    monkeypatch.setattr(pdfmeta.urllib.request, "urlopen", urlopen)
    pdfmeta.metadata_from_pdf(smith_pdf, use_crossref=False)
    assert urlopen.calls == []


def test_pdf_properties_and_file_name_fallbacks(workspace, tmp_path):
    with_props = make_pdf(tmp_path / "a.pdf", ["<p>body text only</p>"],
                          title="A Proper Embedded Title", author="Jane Doe; Richard Roe")
    info, how = pdfmeta.metadata_from_pdf(with_props)
    assert how == "pdf metadata"
    assert info["authors"] == ["Doe, Jane", "Roe, Richard"]
    junk = make_pdf(tmp_path / "My Paper Name.pdf", ["<p>x</p>"], title="Microsoft Word - draft.docx")
    info, how = pdfmeta.metadata_from_pdf(junk)
    assert (how, info["title"]) == ("file name", "My Paper Name")


ARXIV_ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/1406.2661v1</id><title>Generative Adversarial
 Networks</title><published>2014-06-10T18:58:17Z</published>
<author><name>Ian J. Goodfellow</name></author><author><name>Yoshua Bengio</name></author></entry></feed>"""


def test_arxiv_preprint(workspace, tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path / "1406.2661.pdf", [
        "<p style='font-size:30px'>arXiv:1406.2661v1 [stat.ML] 10 Jun 2014</p>"
        "<h1>Generative Adversarial Networks</h1><p>We propose a new framework.</p>"])
    monkeypatch.setattr(pdfmeta.urllib.request, "urlopen", fake_urlopen({"export.arxiv.org": ARXIV_ATOM}))
    info, how = pdfmeta.metadata_from_pdf(pdf)
    assert how == "arxiv"
    assert info["title"] == "Generative Adversarial Networks"
    assert info["authors"] == ["Goodfellow, Ian J.", "Bengio, Yoshua"]
    assert (info["year"], info["doi"]) == ("2014", "10.48550/arXiv.1406.2661")


def test_layout_title_ignores_arxiv_stamp(workspace, tmp_path):
    pdf = make_pdf(tmp_path / "x.pdf", ["<p style='font-size:40px'>arXiv:1706.02690v5 [cs.LG]</p>"
                                        "<h1>Enhancing Reliability of Detection</h1>"])
    assert pdfmeta.probe_pdf(pdf)["layout_title"] == "Enhancing Reliability of Detection"
