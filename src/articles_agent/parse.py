"""PDF -> per-page Markdown, keeping the physical page number."""

import pymupdf4llm


def parse_pdf(path: str) -> list[dict]:
    """Return [{"page": 1-based page number, "text": markdown}, ...]."""
    pages = pymupdf4llm.to_markdown(
        path,
        page_chunks=True,
        use_ocr=False,  # Tesseract data is not installed; scanned pages come back empty
        header=False,   # drop running headers/footers so they don't pollute chunks
        footer=False,
        show_progress=False,
    )
    return [{"page": p["metadata"]["page_number"], "text": p["text"]} for p in pages]
