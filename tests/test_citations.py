"""Quote verification, page location, references and result rendering."""

import pytest

from articles_agent.citations import (cite_labels, format_reference, locate_page, match_quote,
                                      normalize)
from articles_agent.results import CitedFinding, Result

SOURCE = ("Intro sentence. The **deep learning** (DL) based on convolu-\ntional neural networks has been "
          "shown effective for fast evaluation. In order to reduce the number of evaluations by FEM, "
          "we propose an optimization method based on DL. Final sentence here.")


def test_normalize():
    assert normalize("ﬁnite  “element”­—analysis **bold** <sup>1</sup>") == 'finite "element"-analysis bold 1'
    assert normalize("convolu-\ntional") == "convolutional"


def test_exact_quote_is_completed_to_sentence():
    quote, exact = match_quote("deep learning (DL) based on convolutional neural networks", SOURCE, 90)
    assert exact
    assert quote.startswith("The deep learning") and quote.endswith("fast evaluation.")


def test_near_verbatim_quote_is_snapped_to_source():
    llm_quote = "In order to reduce the number of evaluation by FEM we propose a optimization method based on DL."
    quote, exact = match_quote(llm_quote, SOURCE, 90)
    assert not exact
    assert quote == "In order to reduce the number of evaluations by FEM, we propose an optimization method based on DL."


@pytest.mark.parametrize("quote", [
    "Completely invented statement about sourdough bread and baking.",
    "too short",
    "",
])
def test_unverifiable_quotes_are_rejected(quote):
    assert match_quote(quote, SOURCE, 90) is None


def test_locate_page():
    pages = {3: "Text on page three. The CNN surrogate reduced evaluations",
             4: "by 70 percent. More text on page four."}
    assert locate_page("The CNN surrogate reduced evaluations by 70 percent.", pages, 3, 4) == 3
    assert locate_page("More text on page four.", pages, 3, 4) == 4
    assert locate_page("anything", {}, 7, 8) == 7


ARTICLE = {"paper_id": "doe", "title": "Topology Optimization.", "authors": ["Doe, Jane Mary", "Roe, R."],
           "year": "2021", "venue": "IEEE Trans. Magn.", "entry_type": "article", "volume": "57",
           "number": "6", "pages": "1–4", "doi": "10.1/ABC", "url": "", "publisher": "",
           "short_cite": "Doe & Roe, 2021"}


def test_format_reference_article():
    assert format_reference(ARTICLE) == ("Doe, J. M., & Roe, R. (2021). Topology Optimization. "
                                         "*IEEE Trans. Magn.*, 57(6), 1–4. https://doi.org/10.1/ABC")


def test_format_reference_book_and_missing_fields():
    book = {**ARTICLE, "entry_type": "book", "venue": "", "publisher": "Springer", "doi": "",
            "url": "https://x.org", "authors": [], "year": ""}
    assert format_reference(book) == "Topology Optimization. (n.d.). Springer. https://x.org"
    chapter = {**ARTICLE, "entry_type": "incollection", "venue": "Handbook", "doi": ""}
    assert "In *Handbook* (pp. 1–4)." in format_reference(chapter)


def test_cite_labels_disambiguate():
    a = {**ARTICLE, "paper_id": "a", "title": "Alpha"}
    b = {**ARTICLE, "paper_id": "b", "title": "Beta"}
    c = {**ARTICLE, "paper_id": "c", "short_cite": "Poe, 2019"}
    assert cite_labels([b, a, c]) == {"a": "Doe & Roe, 2021a", "b": "Doe & Roe, 2021b", "c": "Poe, 2019"}


def test_result_markdown():
    res = Result(question="Q?", summary="Answer (Doe, 2021, p. 2).",
                 findings=[CitedFinding("Claim", "Quote.", True, "doe", 2, "3 Results", "doe:0001",
                                        "(Doe, 2021, p. 2)")],
                 references=["Doe, J. (2021). T."], queries=["q"], dropped_quotes=1)
    md = res.to_markdown()
    assert md.startswith("## Q?")
    assert "> “Quote.”" in md and "— (Doe, 2021, p. 2), section *3 Results*" in md
    assert "1 finding(s) omitted" in md
    assert "- Doe, J. (2021). T." in md
    assert res.to_dict()["findings"][0]["page"] == 2
