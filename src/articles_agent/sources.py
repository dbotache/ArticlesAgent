"""Library sources: reference-manager exports (+ PDF folders) that make up the database.

A source is one export file (.bib or .ris) from Zotero, JabRef, Citavi, Mendeley, EndNote, …
plus an optional folder with the PDFs. Sources are stored in library/sources.json, so every
installation (every user) has its own.

How PDFs are found for a source:
1. links in the export (Zotero/JabRef/Mendeley `file` field, RIS L1/L4) — absolute, relative to
   the JabRef file directory / the PDF folder / the export's folder, or by file name in the PDF folder;
2. PDFs in the PDF folder that no reference links to are matched by DOI, then by title
   (file name, then first-page text) to references that have no PDF yet;
3. the remaining PDFs are added as papers of their own, cited from Crossref / PDF metadata.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rapidfuzz import fuzz, process

from .config import PROJECT_ROOT
from .metadata import Paper, make_id, norm_doi, norm_title, read_bibtex, read_ris
from .pdfmeta import metadata_from_pdf, probe_pdf, save_caches

SOURCES_FILE = PROJECT_ROOT / "library" / "sources.json"
EXPORTS_DIR = PROJECT_ROOT / "library" / "exports"

MANAGERS = {
    "zotero": "Zotero",
    "jabref": "JabRef",
    "citavi": "Citavi",
    "mendeley": "Mendeley",
    "endnote": "EndNote",
    "other": "Other (BibTeX / RIS)",
}
RIS_SUFFIXES = {".ris", ".txt", ".enw"}


@dataclass
class Source:
    id: str
    manager: str
    reference_file: str
    pdf_dir: str = ""
    use_crossref: bool = True

    @property
    def label(self) -> str:
        return f"{MANAGERS.get(self.manager, self.manager)} · {Path(self.reference_file).name}"

    @property
    def format(self) -> str:
        return "ris" if Path(self.reference_file).suffix.lower() in RIS_SUFFIXES else "bibtex"


@dataclass
class SourceReport:
    papers: list[Paper] = field(default_factory=list)
    references: int = 0          # entries in the export
    with_pdf: int = 0            # references with at least one PDF
    linked: int = 0              # … found through the export's file links
    matched: int = 0             # … found by DOI/title in the PDF folder
    pdfs_in_folder: int = 0
    unlinked_pdfs: int = 0       # PDFs in the folder no reference links to
    standalone: int = 0          # unlinked PDFs added as papers of their own
    duplicates: int = 0          # unlinked PDFs that are copies of a linked paper
    broken_links: int = 0        # file links that point to nothing


# ------------------------------------------------------------ persistence

def load_sources() -> list[Source]:
    if not SOURCES_FILE.exists():
        return []
    return [Source(**s) for s in json.loads(SOURCES_FILE.read_text(encoding="utf-8"))]


def save_sources(sources: list[Source]) -> None:
    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_FILE.write_text(json.dumps([asdict(s) for s in sources], indent=1), encoding="utf-8")


def validate(reference_file: str, pdf_dir: str = "") -> list[str]:
    """Problems that prevent using these paths (empty list = fine)."""
    problems = []
    ref = Path(reference_file).expanduser()
    if not ref.is_file():
        problems.append(f"Reference file not found: {ref}")
    elif ref.suffix.lower() not in RIS_SUFFIXES | {".bib", ".bibtex", ".biblatex"}:
        problems.append("The reference file must be a BibTeX (.bib) or RIS (.ris/.txt) export.")
    if pdf_dir and not Path(pdf_dir).expanduser().is_dir():
        problems.append(f"PDF folder not found: {pdf_dir}")
    return problems


def add_source(manager: str, reference_file: str, pdf_dir: str = "",
               use_crossref: bool = True) -> Source:
    if problems := validate(reference_file, pdf_dir):
        raise ValueError(" ".join(problems))
    sources = load_sources()
    ref = str(Path(reference_file).expanduser().resolve())
    pdfs = str(Path(pdf_dir).expanduser().resolve()) if pdf_dir else ""
    if any(s.reference_file == ref for s in sources):
        raise ValueError(f"{Path(ref).name} is already a library source.")
    n, sid = 1, manager
    while any(s.id == sid for s in sources):
        n += 1
        sid = f"{manager}{n}"
    src = Source(id=sid, manager=manager, reference_file=ref, pdf_dir=pdfs, use_crossref=use_crossref)
    save_sources(sources + [src])
    return src


def remove_source(source_id: str) -> None:
    save_sources([s for s in load_sources() if s.id != source_id])


def save_uploaded_export(name: str, data: bytes) -> Path:
    """An export uploaded through the browser (relative PDF links then rely on the PDF folder)."""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORTS_DIR / Path(name).name
    target.write_bytes(data)
    return target


# ------------------------------------------------------------ reading a source

def _resolve(link: str, bases: list[Path], by_name: dict[str, Path]) -> Path | None:
    candidates = []
    p = Path(link)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates += [b / link for b in bases]
        candidates.append(Path("/" + link))  # Mendeley drops the leading slash on Linux
    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    # Windows path on Linux, moved library, EndNote internal-pdf: fall back to the file name
    return by_name.get(re.split(r"[\\/]", link)[-1].lower())


def _significant_words(title: str) -> set[str]:
    return {w for w in title.split() if len(w) > 3}


def _match(pdf: Path, probe: dict, by_doi: dict[str, Paper], by_title: dict[str, Paper]) -> Paper | None:
    for doi in probe["dois"]:
        if (hit := by_doi.get(norm_doi(doi))):
            return hit
    if not by_title:
        return None
    stem = norm_title(pdf.stem)
    if len(stem) >= 20 and (best := process.extractOne(stem, list(by_title), scorer=fuzz.partial_ratio,
                                                         score_cutoff=92)):
        return by_title[best[0]]
    head = norm_title(probe["head"])
    head_words = set(head.split())
    plausible = [t for t in by_title if len(_significant_words(t) & head_words) >= 0.6 * len(_significant_words(t))]
    if plausible and (best := process.extractOne(head, plausible, scorer=fuzz.partial_ratio,
                                                 score_cutoff=92)):
        return by_title[best[0]]
    return None


def read_source(src: Source, scan_folder: bool = True, progress=None) -> SourceReport:
    """All papers of one source with resolved PDF paths. scan_folder=False skips steps 2-3
    (fast preview: no PDF is opened)."""
    ref = Path(src.reference_file)
    if src.format == "ris":
        entries, file_dirs = read_ris(ref), []
    else:
        entries, file_dirs = read_bibtex(ref)
    pdf_dir = Path(src.pdf_dir) if src.pdf_dir else None
    folder_pdfs = sorted(p.resolve() for p in pdf_dir.rglob("*")
                         if p.suffix.lower() == ".pdf" and p.is_file()) if pdf_dir and pdf_dir.is_dir() else []
    by_name: dict[str, Path] = {}
    for p in folder_pdfs:
        by_name.setdefault(p.name.lower(), p)
    bases = [Path(d) if Path(d).is_absolute() else ref.parent / d for d in file_dirs]
    bases += [b for b in (pdf_dir, ref.parent) if b]

    report = SourceReport(references=len(entries), pdfs_in_folder=len(folder_pdfs))
    taken: set[str] = set()
    for e in entries:
        resolved = []
        for link in e.pdf_paths:
            if p := _resolve(link, bases, by_name):
                resolved.append(str(p))
            else:
                report.broken_links += 1
        e.pdf_paths = list(dict.fromkeys(resolved))
        e.source = src.id
        if not e.paper_id:
            e.paper_id = make_id(e.title, e.authors, e.year, taken)
        taken.add(e.paper_id)
    report.linked = sum(bool(e.pdf_paths) for e in entries)

    linked = {p for e in entries for p in e.pdf_paths}
    unlinked = [p for p in folder_pdfs if str(p) not in linked]
    report.unlinked_pdfs = len(unlinked)

    if scan_folder and unlinked:
        without_pdf = [e for e in entries if not e.pdf_paths]
        by_doi = {norm_doi(e.doi): e for e in without_pdf if e.doi}
        by_title = {norm_title(e.title): e for e in without_pdf if len(norm_title(e.title)) >= 20}
        # to recognise copies of papers that already have a PDF
        have_doi = {norm_doi(e.doi) for e in entries if e.doi and e.pdf_paths}
        have_title = {norm_title(e.title) for e in entries if e.pdf_paths}
        for n, pdf in enumerate(unlinked, 1):
            try:
                probe = probe_pdf(pdf)
            except Exception:  # unreadable PDF: ingestion will report it
                continue
            if hit := _match(pdf, probe, by_doi, by_title):
                hit.pdf_paths.append(str(pdf))
                report.matched += 1
                by_doi.pop(norm_doi(hit.doi), None)
                by_title.pop(norm_title(hit.title), None)
            else:
                info, how = metadata_from_pdf(pdf, src.use_crossref)
                if (info["doi"] and norm_doi(info["doi"]) in have_doi) or norm_title(info["title"]) in have_title:
                    report.duplicates += 1
                else:
                    pid = make_id(info["title"], info["authors"], info["year"], taken)
                    taken.add(pid)
                    entries.append(Paper(paper_id=pid, **info, pdf_paths=[str(pdf)],
                                         in_bib=False, source=src.id))
                    report.standalone += 1
            if progress:
                progress("scan", n, len(unlinked))
        save_caches()

    report.with_pdf = sum(bool(e.pdf_paths) for e in entries if e.in_bib)
    report.papers = entries
    return report


# ------------------------------------------------------------ the whole library

def build_library(progress=None) -> dict[str, Paper]:
    """Papers of all sources plus individually imported PDFs (library.py), de-duplicated by
    DOI and title across sources (the first source wins)."""
    from .library import registry_papers  # library imports this module

    papers: dict[str, Paper] = {}
    by_doi, by_title = {}, {}  # -> paper_id of the entry that was kept
    for src in load_sources():
        for p in read_source(src, progress=progress).papers:
            t = norm_title(p.title)
            kept = (by_doi.get(norm_doi(p.doi)) if p.doi else None) or (by_title.get(t) if len(t) >= 20 else None)
            if kept:  # duplicate entry: keep the first, but don't lose its PDFs
                papers[kept].pdf_paths = list(dict.fromkeys(papers[kept].pdf_paths + p.pdf_paths))
                continue
            if p.paper_id in papers:
                p.paper_id = f"{p.paper_id}_{src.id}"
            papers[p.paper_id] = p
            if p.doi:
                by_doi[norm_doi(p.doi)] = p.paper_id
            if len(t) >= 20:
                by_title[t] = p.paper_id
    papers |= {pid: p for pid, p in registry_papers().items() if pid not in papers}
    return papers
