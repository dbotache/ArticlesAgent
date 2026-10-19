"""Add PDFs that are not in Zotero, as single files or whole folders.

Imported papers are recorded in library/extra_papers.json, which `metadata.build_library`
merges with the Zotero .bib. Re-ingestion therefore keeps them. Uploaded files are copied to
library/uploads/; PDFs from a folder are used in place.

Citation metadata: the DOI printed on the first pages is looked up at Crossref (only the DOI is
sent). The Crossref title must also appear in the PDF text, so a DOI from a cited reference is not
taken by mistake. Otherwise the PDF's own metadata or the file name is used.
"""

import hashlib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import fields
from datetime import date
from pathlib import Path

import pymupdf
from rapidfuzz import fuzz, process

from .config import PROJECT_ROOT, load_config
from .metadata import Paper, build_library

LIBRARY_DIR = PROJECT_ROOT / "library"
REGISTRY = LIBRARY_DIR / "extra_papers.json"
UPLOADS = LIBRARY_DIR / "uploads"

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>{}]+)", re.IGNORECASE)
PAPER_FIELDS = {f.name for f in fields(Paper)}
CROSSREF_TYPES = {"journal-article": "article", "proceedings-article": "inproceedings",
                  "book-chapter": "incollection", "book": "book", "monograph": "book",
                  "edited-book": "book", "dissertation": "thesis", "report": "report",
                  "posted-content": "misc"}
STOPWORDS = {"the", "and", "for", "with", "from", "into", "using", "der", "die", "das", "und",
             "von", "mit", "zur", "zum", "ein", "eine", "on", "of", "an", "a", "in", "to"}


# ------------------------------------------------------------ registry

def load_registry() -> dict[str, dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}


def registry_papers() -> dict[str, Paper]:
    return {pid: Paper(**{k: v for k, v in entry.items() if k in PAPER_FIELDS})
            for pid, entry in load_registry().items()}


def _save_registry(reg: dict) -> None:
    LIBRARY_DIR.mkdir(exist_ok=True)
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(REGISTRY)


# ------------------------------------------------------------ files

def collect_pdfs(folder: str | Path) -> list[Path]:
    """All PDFs below `folder`, recursively."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".pdf" and p.is_file())


def save_upload(name: str, data: bytes) -> Path:
    """Store an uploaded PDF under library/uploads/<content hash>/<name>."""
    target = UPLOADS / hashlib.sha256(data).hexdigest()[:12] / Path(name).name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return target


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------ metadata

def _crossref(doi: str) -> dict | None:
    req = urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi),
                                 headers={"User-Agent": "ArticlesAgent/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)["message"]
    except Exception:
        return None


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
        "doi": m.get("DOI", ""),
        "url": m.get("URL", ""),
        "volume": m.get("volume", ""),
        "number": m.get("issue", ""),
        "pages": m.get("page", "").replace("-", "–"),
        "publisher": m.get("publisher", ""),
    }


def _last_first(name: str) -> str:
    name = name.strip()
    if "," in name or " " not in name:
        return name
    first, last = name.rsplit(" ", 1)
    return f"{last}, {first}"


def _title_from_layout(page) -> str:
    """The largest-font text in the upper half of the first page is usually the title."""
    spans = [s for b in page.get_text("dict")["blocks"] for line in b.get("lines", [])
             for s in line["spans"] if s["text"].strip() and s["bbox"][1] < page.rect.height / 2]
    if not spans:
        return ""
    biggest = max(s["size"] for s in spans)
    title = " ".join(s["text"].strip() for s in spans if s["size"] >= biggest - 0.5)
    return re.sub(r"\s+", " ", title).strip() if 10 <= len(title) <= 300 else ""


def extract_metadata(pdf: Path, use_crossref: bool = True) -> tuple[dict, str]:
    """(metadata fields, source): 'crossref', 'pdf metadata', 'first page' or 'file name'."""
    with pymupdf.open(pdf) as doc:
        text = "\n".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
        meta = doc.metadata or {}
        layout_title = _title_from_layout(doc[0]) if doc.page_count else ""
    flat = " ".join(text.split()).lower()

    if use_crossref:
        dois = list(dict.fromkeys(d.rstrip(".,;)]") for d in DOI_RE.findall(text)))[:3]
        for doi in dois:
            m = _crossref(doi)
            if not (m and m.get("title")):
                continue
            info = _from_crossref(m)
            if fuzz.partial_ratio(info["title"].lower(), flat) >= 85:
                return info, "crossref"

    empty = {k: "" for k in ("year", "venue", "entry_type", "doi", "url", "volume",
                             "number", "pages", "publisher")}
    title = (meta.get("title") or "").strip()
    junk = (len(title) < 8 or title.lower().startswith(("microsoft word", "untitled"))
            or title.lower().endswith((".pdf", ".doc", ".docx", ".tex", ".dvi")))
    authors = [_last_first(a) for a in re.split(r";|\band\b|&", meta.get("author") or "") if a.strip()]
    if not junk:
        return {**empty, "title": title, "authors": authors}, "pdf metadata"
    if layout_title:
        return {**empty, "title": layout_title, "authors": authors}, "first page"
    return {**empty, "title": pdf.stem, "authors": authors}, "file name"


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", " ", title.lower()).strip()


def _make_id(info: dict, taken: set[str]) -> str:
    """Zotero/Better-BibTeX style key: lastname_firstword_year."""
    last = info["authors"][0].split(",")[0] if info["authors"] else "nn"
    last = re.sub(r"[^a-z]", "", last.lower()) or "nn"
    words = [w for w in re.findall(r"[a-zäöü]+", info["title"].lower()) if w not in STOPWORDS]
    base = f"{last}_{words[0] if words else 'paper'}_{info['year'] or 'nodate'}"
    pid, n = base, 2
    while pid in taken:
        pid, n = f"{base}_{n}", n + 1
    return pid


# ------------------------------------------------------------ public API

def register_pdfs(paths: list[str | Path], use_crossref: bool = True, progress=None) -> list[dict]:
    """Record new PDFs in the registry. Returns one status dict per file:
    {file, path, status: added|skipped|error, detail, paper_id?}. Nothing is indexed yet."""
    cfg = load_config()
    existing = {pid: p.to_dict() for pid, p in
                build_library(cfg["library"]["bib_file"], cfg["library"]["storage_dir"]).items()}
    reg = load_registry()
    known_hashes = {e.get("sha256") for e in reg.values()}
    known_paths = {path for e in existing.values() for path in e.get("pdf_paths", [])}
    known_dois = {e["doi"].lower(): pid for pid, e in existing.items() if e.get("doi")}
    known_titles = {_norm_title(e["title"]): pid for pid, e in existing.items() if e.get("title")}

    results = []
    for n, raw in enumerate(paths, 1):
        path = Path(raw).expanduser().resolve()
        status = {"file": path.name, "path": str(path)}
        try:
            digest = _sha256(path)
            if str(path) in known_paths:
                status.update(status="skipped", detail="already in the library")
            elif digest in known_hashes:
                status.update(status="skipped", detail="same file was imported before")
            else:
                info, source = extract_metadata(path, use_crossref)
                same_title = process.extractOne(_norm_title(info["title"]), list(known_titles),
                                                scorer=fuzz.ratio, score_cutoff=95)
                if info["doi"] and info["doi"].lower() in known_dois:
                    status.update(status="skipped",
                                  detail=f"DOI already in the library as {known_dois[info['doi'].lower()]}")
                elif same_title:
                    status.update(status="skipped",
                                  detail=f"same title already in the library as {known_titles[same_title[0]]}")
                else:
                    pid = _make_id(info, set(existing) | set(reg))
                    reg[pid] = {**info, "paper_id": pid, "pdf_paths": [str(path)], "in_bib": False,
                                "sha256": digest, "added": date.today().isoformat(),
                                "metadata_source": source}
                    existing[pid] = reg[pid]
                    known_hashes.add(digest)
                    known_paths.add(str(path))
                    if info["doi"]:
                        known_dois[info["doi"].lower()] = pid
                    known_titles[_norm_title(info["title"])] = pid
                    status.update(status="added", paper_id=pid,
                                  detail=f"{info['title'][:80]} (metadata: {source})")
        except Exception as exc:  # unreadable / encrypted PDF
            status.update(status="error", detail=f"{type(exc).__name__}: {exc}")
        results.append(status)
        if progress:
            progress("register", n, len(paths))
    _save_registry(reg)
    return results


def update_database(progress=None) -> dict:
    """Parse + chunk new PDFs (Zotero and imported), then embed new chunks."""
    from .index import run_index
    from .ingest import run_ingest
    ingest = run_ingest(progress=progress)
    index = run_index(progress=progress)
    return {"ingest": ingest, "index": index}
