"""Section-aware chunking with page ranges."""

from articles_agent.chunking import chunk_pages, clean_page_text


def pages(*texts):
    return [{"page": i, "text": t} for i, t in enumerate(texts, 1)]


def test_sections_pages_and_references():
    chunks = chunk_pages(pages(
        "# Title\n\nAuthors\n\n## **1 Introduction**\n\nIntro text one.\n\n",
        "Intro continues on page two.\n\n## 2 Method\n\nMethod text.\n\n## References\n\n- [1] Ref.",
    ), "p", "/x.pdf", chunk_size=2500, chunk_overlap=300)
    by_section = {c.section: c for c in chunks}
    assert set(by_section) == {"Title", "1 Introduction", "2 Method", "References"}
    intro = by_section["1 Introduction"]
    assert (intro.page_start, intro.page_end) == (1, 2)  # spans the page break
    assert "Intro continues" in intro.text
    assert by_section["References"].is_references
    assert not intro.is_references
    assert [c.chunk_id for c in chunks] == [f"p:{i:04d}" for i in range(len(chunks))]


def test_chunks_respect_size_and_overlap():
    para = "Sentence number {} talks about motors. "
    text = "## 1 Long\n\n" + "\n\n".join(para.format(i) * 5 for i in range(40))
    chunks = chunk_pages(pages(text), "p", "/x.pdf", chunk_size=800, chunk_overlap=300)
    assert len(chunks) > 3
    assert all(len(c.text) <= 800 + 300 for c in chunks)
    # consecutive chunks of one section share their boundary paragraph
    assert chunks[0].text.split("\n\n")[-1] == chunks[1].text.split("\n\n")[0]


def test_long_paragraph_is_split_at_sentences():
    text = "## A\n\n" + " ".join(f"Sentence {i} is here." for i in range(300))
    chunks = chunk_pages(pages(text), "p", "/x.pdf", chunk_size=500, chunk_overlap=0)
    assert all(len(c.text) <= 500 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks[:-1])


def test_latex_umlauts_are_repaired():
    assert clean_page_text("L¨osung und ¨Ubung, Gr¨oße") == "Lösung und Übung, Größe"


def test_picture_placeholders_are_dropped():
    chunks = chunk_pages(pages("## A\n\n==> picture [10 x 10] intentionally omitted <==\n\nReal text."),
                         "p", "/x.pdf", 2500, 300)
    assert chunks[0].text == "Real text."
