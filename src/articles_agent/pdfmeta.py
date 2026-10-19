"""Facts read from a PDF itself: DOIs, first-page text, embedded properties, title guess.

Used to match PDFs to references (when an export has no file links) and to cite PDFs that have
no reference entry at all. Results are cached in library/cache/ (keyed by path + mtime), and
Crossref answers by DOI, so a re-sync doesn't open every PDF or query Crossref again.
"""

import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pymupdf
from rapidfuzz import fuzz

from .config import PROJECT_ROOT
from .metadata import clean_doi, norm_doi

CACHE_DIR = PROJECT_ROOT / "library" / "cache"
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>{}]+)", re.IGNORECASE)
ARXIV_RE = re.compile(r"arxiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
CROSSREF_TYPES = {"journal-article": "article", "proceedings-article": "inproceedings",
                  "book-chapter": "incollection", "book": "book", "monograph": "book",
                  "edited-book": "book", "dissertation": "thesis", "report": "report",
                  "posted-content": "misc"}

_lock = threading.Lock()
_caches: dict[str, dict] = {}


def _cache(name: str) -> dict:
    if name not in _caches:
        f = CACHE_DIR / f"{name}.json"
        _caches[name] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return _caches[name]


def save_caches() -> None:
    with _lock:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in _caches.items():
            tmp = CACHE_DIR / f"{name}.tmp"
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(CACHE_DIR / f"{name}.json")


# ------------------------------------------------------------ probing

def _title_from_layout(page) -> str:
    """The largest-font text in the upper half of the first page is usually the title."""
    def text(block):
        return " ".join(s["text"] for line in block.get("lines", []) for s in line["spans"]).strip()
    # drop arXiv's margin stamp (may be split into several spans or lines)
    blocks = [b for b in page.get_text("dict")["blocks"] if not ARXIV_RE.match(text(b))]
    spans = [s for b in blocks for line in b.get("lines", []) for s in line["spans"]
             if s["text"].strip() and s["bbox"][1] < page.rect.height / 2]
    if not spans:
        return ""
    biggest = max(s["size"] for s in spans)
    title = " ".join(s["text"].strip() for s in spans if s["size"] >= biggest - 0.5)
    return re.sub(r"\s+", " ", title).strip() if 10 <= len(title) <= 300 else ""


def probe_pdf(path: str | Path) -> dict:
    """{dois, head (normalized first-page text), meta_title, meta_author, layout_title}."""
    path = Path(path)
    key, mtime = str(path), path.stat().st_mtime
    cache = _cache("probes")
    if (hit := cache.get(key)) and hit["mtime"] == mtime and "arxiv" in hit:
        return hit
    with pymupdf.open(path) as doc:
        text = "\n".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
        meta = doc.metadata or {}
        layout_title = _title_from_layout(doc[0]) if doc.page_count else ""
    probe = {
        "mtime": mtime,
        "dois": list(dict.fromkeys(d.rstrip(".,;)]") for d in DOI_RE.findall(text)))[:3],
        "arxiv": (m.group(1) if (m := ARXIV_RE.search(text)) else ""),
        "head": " ".join(text.split()).lower()[:6000],
        "meta_title": (meta.get("title") or "").strip(),
        "meta_author": (meta.get("author") or "").strip(),
        "layout_title": layout_title,
    }
    with _lock:
        cache[key] = probe
    return probe


# ------------------------------------------------------------ Crossref

def crossref(doi: str) -> dict | None:
    """Crossref work record for a DOI (only the DOI is sent), cached; None if unknown/offline."""
    doi = norm_doi(doi)
    cache = _cache("crossref")
    if doi in cache:
        return cache[doi]
    req = urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi),
                                 headers={"User-Agent": "ArticlesAgent/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            record = json.load(resp)["message"]
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return None
        record = None  # unknown DOI: remember the miss
    except Exception:  # offline / timeout: don't cache, try again next time
        return None
    with _lock:
        cache[doi] = record
    return record


def arxiv(arxiv_id: str) -> dict | None:
    """Paper fields from the arXiv API (only the arXiv id is sent), cached."""
    cache = _cache("arxiv")
    if arxiv_id in cache:
        return cache[arxiv_id]
    import xml.etree.ElementTree as ET
    url = "https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(arxiv_id)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ArticlesAgent/0.1"}),
                                    timeout=10) as resp:
            root = ET.fromstring(resp.read())
    except Exception:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    entry = root.find("a:entry", ns)
    title = entry.findtext("a:title", "", ns) if entry is not None else ""
    record = None
    if title.strip() and entry.find("a:id", ns) is not None and "api/errors" not in entry.findtext("a:id", "", ns):
        names = [a.findtext("a:name", "", ns).strip() for a in entry.findall("a:author", ns)]
        record = {
            "title": " ".join(title.split()), "authors": [_last_first(n) for n in names if n],
            "year": entry.findtext("a:published", "", ns)[:4], "venue": "arXiv",
            "entry_type": "misc", "doi": clean_doi(entry.findtext("x:doi", "", ns) or f"10.48550/arXiv.{arxiv_id}"),
            "url": f"https://arxiv.org/abs/{arxiv_id}", "volume": "", "number": "",
            "pages": "", "publisher": "arXiv",
        }
    with _lock:
        cache[arxiv_id] = record
    return record


def _from_crossref(m: dict) -> dict:
    year = ""
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (m.get(key) or {}).get("date-parts") or [[None]]
        if parts[0] and parts[0][0]:
            year = str(parts[0][0])
            break
    return {
        "title": re.sub(r"<[^>]+>", "", (m.get("title") or [""])[0]).strip(),
        "authors": [f"{a['family']}, {a.get('given', '')}".strip(", ")
                    for a in m.get("author", []) if a.get("family")],
        "year": year,
        "venue": (m.get("container-title") or [""])[0],
        "entry_type": CROSSREF_TYPES.get(m.get("type", ""), "misc"),
        "doi": clean_doi(m.get("DOI", "")),
        "url": m.get("URL", ""),
        "volume": m.get("volume", ""),
        "number": m.get("issue", ""),
        "pages": m.get("page", "").replace("-", "–"),
        "publisher": m.get("publisher", ""),
    }


# ------------------------------------------------------------ metadata for PDFs without a reference

def _last_first(name: str) -> str:
    name = name.strip()
    if "," in name or " " not in name:
        return name
    first, last = name.rsplit(" ", 1)
    return f"{last}, {first}"


def metadata_from_pdf(path: str | Path, use_crossref: bool = True) -> tuple[dict, str]:
    """(Paper fields, source): 'crossref', 'arxiv', 'pdf metadata', 'first page' or 'file name'.

    A Crossref record is only accepted if its title appears in the PDF text, so a DOI of a cited
    reference is not taken for the paper itself."""
    path = Path(path)
    probe = probe_pdf(path)
    if use_crossref:
        for doi in probe["dois"]:
            m = crossref(doi)
            if m and m.get("title"):
                info = _from_crossref(m)
                if fuzz.partial_ratio(info["title"].lower(), probe["head"]) >= 85:
                    return info, "crossref"
        if probe.get("arxiv") and (info := arxiv(probe["arxiv"])):
            return info, "arxiv"

    empty = {k: "" for k in ("year", "venue", "entry_type", "doi", "url", "volume",
                             "number", "pages", "publisher")}
    title = probe["meta_title"]
    junk = (len(title) < 8 or title.lower().startswith(("microsoft word", "untitled"))
            or title.lower().endswith((".pdf", ".doc", ".docx", ".tex", ".dvi")))
    authors = [_last_first(a) for a in re.split(r";|\band\b|&", probe["meta_author"]) if a.strip()]
    if not junk:
        return {**empty, "title": title, "authors": authors}, "pdf metadata"
    if probe["layout_title"]:
        return {**empty, "title": probe["layout_title"], "authors": authors}, "first page"
    return {**empty, "title": path.stem, "authors": authors}, "file name"
