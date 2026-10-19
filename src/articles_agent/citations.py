"""Quote verification against the source text, page location, and APA-style references."""

import re
import unicodedata

from rapidfuzz import fuzz

QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
                             "‘": "'", "’": "'", "‚": "'", "–": "-", "—": "-", "−": "-"})


def normalize(text: str) -> str:
    """Comparable plain text: no Markdown/HTML, ligatures resolved, whitespace collapsed."""
    t = unicodedata.normalize("NFKC", text).replace("­", "")
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)  # re-join words hyphenated at line ends
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"[*_`#>]", "", t)
    t = t.translate(QUOTE_CHARS)
    return re.sub(r"\s+", " ", t).strip()


SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[ÄÖÜ])")


def _widen_to_sentences(text: str, start: int, end: int, reach: int = 200) -> str:
    """Grow a fuzzy match to whole sentences (if a boundary is within `reach` chars)."""
    starts = [m.end() for m in SENTENCE_BREAK.finditer(text, max(0, start - reach), start + 1)]
    if starts:
        start = starts[-1]
    elif start <= reach:
        start = 0
    if m := SENTENCE_BREAK.search(text, max(start, end - 1), end + reach):
        end = m.start()
    elif len(text) - end <= reach:
        end = len(text)
    return text[start:end].strip()


def match_quote(quote: str, source: str, threshold: float) -> tuple[str, bool] | None:
    """Find `quote` in `source`. Returns (quote as it appears in the source, exact?) or None.

    Near-misses (an LLM fixing a typo, dropping a hyphen) are snapped to the real source text,
    so the delivered quote is always verbatim; partial sentences are completed.
    """
    q, src = normalize(quote).strip('"\' '), normalize(source)
    if len(q) < 15:
        return None
    if (i := src.lower().find(q.lower())) >= 0:
        return _widen_to_sentences(src, i, i + len(q)), True
    al = fuzz.partial_ratio_alignment(q.lower(), src.lower())
    if al and al.score >= threshold and len(q) <= len(src):
        return _widen_to_sentences(src, al.dest_start, al.dest_end), False
    return None


def locate_page(quote: str, pages: dict[int, str], page_start: int, page_end: int) -> int:
    """Physical PDF page on which the (verified) quote starts."""
    head = normalize(quote).lower()[:60]
    candidates = [p for p in range(page_start, page_end + 1) if p in pages]
    norm = {p: normalize(pages[p]).lower() for p in candidates}
    for p in candidates:
        if head in norm[p]:
            return p
    if not candidates:
        return page_start
    return max(candidates, key=lambda p: fuzz.partial_ratio(head, norm[p]))


# ---------------------------------------------------------------- references

def _apa_name(author: str) -> str:
    if "," not in author:  # organisation / single name
        return author
    last, first = (s.strip() for s in author.split(",", 1))
    initials = " ".join(f"{part[0]}." for part in re.split(r"[\s.]+", first) if part)
    initials = re.sub(r"(\w)\.\s*-\s*", r"\1.-", initials)
    return f"{last}, {initials}" if initials else last


def _apa_authors(authors: list[str]) -> str:
    names = [_apa_name(a) for a in authors]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) > 20:
        return ", ".join(names[:19]) + ", … " + names[-1]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def format_reference(p: dict, label_year: str | None = None) -> str:
    """APA-7-like reference entry, e.g. for the bibliography of an answer."""
    year = label_year or p.get("year") or "n.d."
    title = p["title"].rstrip(".")
    who = _apa_authors(p.get("authors", []))
    head = f"{who} ({year}). {title}." if who else f"{title}. ({year})."
    venue, etype = p.get("venue", ""), p.get("entry_type", "")
    if etype == "article" and venue:
        vol = p.get("volume", "")
        vol += f"({p['number']})" if p.get("number") else ""
        body = ", ".join(x for x in (f"*{venue}*", vol, p.get("pages", "")) if x) + "."
    elif etype in ("inproceedings", "incollection") and venue:
        pages = f" (pp. {p['pages']})" if p.get("pages") else ""
        body = f"In *{venue}*{pages}." + (f" {p['publisher']}." if p.get("publisher") else "")
    elif p.get("publisher"):
        body = f"{p['publisher']}."
    else:
        body = f"*{venue}*." if venue else ""
    link = f"https://doi.org/{p['doi']}" if p.get("doi") else p.get("url", "")
    return " ".join(x for x in (head, body, link) if x)


def cite_labels(papers: list[dict]) -> dict[str, str]:
    """Author-year labels, disambiguated with a/b/... when two papers share one."""
    by_label: dict[str, list[dict]] = {}
    for p in sorted(papers, key=lambda p: p["title"]):
        by_label.setdefault(p["short_cite"], []).append(p)
    labels = {}
    for label, group in by_label.items():
        for i, p in enumerate(group):
            labels[p["paper_id"]] = label if len(group) == 1 else f"{label}{chr(97 + i)}"
    return labels
