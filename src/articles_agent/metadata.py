"""Bibliographic records from reference-manager exports: BibTeX/BibLaTeX and RIS.

BibTeX covers Zotero, JabRef, Mendeley and Citavi exports; RIS covers EndNote, Citavi and most
other tools. PDF links are returned as written in the export (`Paper.pdf_paths`); `sources.py`
resolves them to files on disk.
"""

import re
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode


@dataclass
class Paper:
    paper_id: str  # citation key
    title: str
    authors: list[str] = field(default_factory=list)  # "Last, First"
    year: str = ""
    venue: str = ""
    entry_type: str = ""
    doi: str = ""
    url: str = ""
    volume: str = ""
    number: str = ""
    pages: str = ""
    publisher: str = ""
    pdf_paths: list[str] = field(default_factory=list)
    in_bib: bool = True  # False: no reference entry, metadata taken from the PDF itself
    source: str = ""     # id of the library source it came from

    @property
    def short_cite(self) -> str:
        """Author-year label, e.g. 'Kallioras et al., 2020'."""
        if not self.authors:
            who = self.title[:40]
        else:
            last = [a.split(",")[0].strip() for a in self.authors]
            who = last[0] if len(last) == 1 else (
                f"{last[0]} & {last[1]}" if len(last) == 2 else f"{last[0]} et al.")
        return f"{who}, {self.year or 'n.d.'}"

    def to_dict(self) -> dict:
        return asdict(self) | {"short_cite": self.short_cite}


# ------------------------------------------------------------ helpers shared by all readers

STOPWORDS = {"the", "and", "for", "with", "from", "into", "using", "der", "die", "das", "und",
             "von", "mit", "zur", "zum", "ein", "eine", "on", "of", "an", "a", "in", "to"}


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", " ", title.lower()).strip()


def clean_doi(doi: str) -> str:
    """Bare DOI as written ('https://doi.org/10.1/AbC' -> '10.1/AbC')."""
    return re.sub(r"^(https?://)?(dx\.)?doi\.org/|^doi:\s*", "", doi.strip(), flags=re.I).rstrip(".,;")


def norm_doi(doi: str) -> str:
    """DOI for comparisons (DOIs are case-insensitive)."""
    return clean_doi(doi).lower()


def make_id(title: str, authors: list[str], year: str, taken: set[str]) -> str:
    """Better-BibTeX-style citation key lastname_firstword_year, unique within `taken`."""
    last = authors[0].split(",")[0] if authors else "nn"
    last = re.sub(r"[^a-z]", "", last.lower()) or "nn"
    words = [w for w in re.findall(r"[a-z0-9äöü]+", title.lower())
             if w not in STOPWORDS and not w.isdigit()]
    base = f"{last}_{words[0] if words else 'paper'}_{year or 'nodate'}"
    pid, n = base, 2
    while pid in taken:
        pid, n = f"{base}_{n}", n + 1
    return pid


def read_text(path: str | Path) -> str:
    """Exports from Windows tools are not always UTF-8."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()


# ------------------------------------------------------------ BibTeX / BibLaTeX

# JabRef stores the library's PDF base folder in a comment block
JABREF_FILE_DIR = re.compile(r"jabref-meta:\s*fileDirectory(?:-[^:]*)?:(.*?);", re.IGNORECASE)


def parse_file_field(value: str) -> list[str]:
    """PDF paths from a `file` field. Handles
    Zotero    {Full Text PDF:/home/u/Zotero/storage/KEY/x.pdf:application/pdf}
    Zotero/Win{Full Text PDF:C\\:\\\\Users\\\\u\\\\x.pdf:application/pdf}
    JabRef    {:papers/x.pdf:PDF}            (relative to the file directory)
    Mendeley  {:home/u/x.pdf:pdf} / {:C$\\backslash$:/Users/u/x.pdf:pdf}
    and several entries separated by ';'."""
    value = value.replace(r"$\backslash$", "\\")
    paths = []
    for part in re.split(r"(?<!\\);", value):
        fields = re.split(r"(?<!\\):", part.strip())
        path = fields[1] if len(fields) >= 2 else fields[0]
        path = path.replace("\\:", ":").replace("\\\\", "\\").strip()
        if path.lower().endswith(".pdf"):
            paths.append(path)
    return paths


def _keep_raw_file(record: dict) -> dict:
    raw = record.get("file")
    record = convert_to_unicode(record)  # LaTeX -> Unicode would mangle Windows paths
    if raw is not None:
        record["file"] = raw
    return record


def _paper_from_bib(e: dict) -> Paper:
    year = e.get("year") or e.get("date", "")[:4]
    return Paper(
        paper_id=e["ID"],
        title=_clean(e.get("title", "")),
        authors=[_clean(a) for a in e.get("author", e.get("editor", "")).split(" and ") if a.strip()],
        year=year if year.isdigit() else "",
        venue=_clean(e.get("journaltitle") or e.get("journal") or e.get("booktitle")
                     or e.get("eventtitle") or e.get("institution") or e.get("school") or ""),
        entry_type=e.get("ENTRYTYPE", ""),
        doi=clean_doi(e.get("doi", "")),
        url=e.get("url", ""),
        volume=e.get("volume", ""),
        number=e.get("number", ""),
        pages=e.get("pages", "").replace("--", "–"),
        publisher=_clean(e.get("publisher", "")),
        pdf_paths=parse_file_field(e.get("file", "")),
    )


def read_bibtex(path: str | Path) -> tuple[list[Paper], list[str]]:
    """(entries, JabRef file directories declared in the file)."""
    text = read_text(path)
    parser = BibTexParser(common_strings=True, ignore_nonstandard_types=False)
    parser.customization = _keep_raw_file
    db = bibtexparser.loads(text, parser=parser)
    file_dirs = [d.replace("\\:", ":").replace("\\\\", "\\").strip()
                 for d in JABREF_FILE_DIR.findall(text)]
    return [_paper_from_bib(e) for e in db.entries], [d for d in file_dirs if d]


# ------------------------------------------------------------ RIS

RIS_LINE = re.compile(r"^([A-Z][A-Z0-9])  -(?: (.*))?$")
RIS_TYPES = {"JOUR": "article", "JFULL": "article", "EJOUR": "article", "MGZN": "article",
             "CONF": "inproceedings", "CPAPER": "inproceedings", "CHAP": "incollection",
             "BOOK": "book", "EBOOK": "book", "EDBOOK": "book", "THES": "thesis",
             "RPRT": "report", "STAND": "report"}


def _ris_link(value: str) -> str | None:
    v = value.strip()
    if v.lower().startswith("file:"):
        return urllib.parse.unquote(urllib.parse.urlparse(v).path)
    if v.lower().startswith("internal-pdf:"):  # EndNote: <library>.Data/PDF/<id>/<name>
        return urllib.parse.unquote(v.rsplit("/", 1)[-1])
    if v.lower().endswith(".pdf") and not v.lower().startswith("http"):
        return v
    return None


def _paper_from_ris(r: dict[str, list[str]]) -> Paper:
    def first(*tags: str) -> str:
        return next((r[t][0].strip() for t in tags if r.get(t) and r[t][0].strip()), "")

    year = re.search(r"\d{4}", first("PY", "Y1", "DA"))
    title = first("TI", "T1", "CT", "BT")
    venue = first("T2", "JF", "JO", "JA", "BT", "T3")
    sp, ep = first("SP"), first("EP")
    links = [link for t in ("L1", "L4", "LK", "UR") for v in r.get(t, []) if (link := _ris_link(v))]
    authors = list(dict.fromkeys(a.strip() for t in ("AU", "A1") for a in r.get(t, []) if a.strip()))
    return Paper(
        paper_id="",  # RIS has no citation key; sources.py assigns one
        title=title,
        authors=authors,
        year=year.group(0) if year else "",
        venue=venue if venue != title else "",
        entry_type=RIS_TYPES.get(first("TY"), "misc"),
        doi=clean_doi(first("DO")),
        url=next((u for u in r.get("UR", []) if u.startswith("http")), ""),
        volume=first("VL"),
        number=first("IS"),
        pages=f"{sp}–{ep}" if sp and ep else sp,
        publisher=first("PB"),
        pdf_paths=links,
    )


def read_ris(path: str | Path) -> list[Paper]:
    records, cur = [], None
    for line in read_text(path).splitlines():
        m = RIS_LINE.match(line.rstrip())
        if not m:
            continue
        tag, value = m.group(1), (m.group(2) or "").strip()
        if tag == "TY":
            cur = {"TY": [value]}
        elif tag == "ER":
            if cur:
                records.append(cur)
            cur = None
        elif cur is not None:
            cur.setdefault(tag, []).append(value)
    return [_paper_from_ris(r) for r in records]
