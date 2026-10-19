"""Bibliographic metadata from the Zotero BibLaTeX export, keyed by citation key."""

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

# Zotero `file` field: "Description:/abs/path.pdf:application/pdf;Description:..."
PDF_IN_FILE_FIELD = re.compile(r":(/[^:;]+?\.pdf):application/pdf", re.IGNORECASE)


@dataclass
class Paper:
    paper_id: str  # BibTeX citation key
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
    in_bib: bool = True

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


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()


def _paper_from_entry(e: dict) -> Paper:
    year = e.get("year") or e.get("date", "")[:4]
    return Paper(
        paper_id=e["ID"],
        title=_clean(e.get("title", "")),
        authors=[_clean(a) for a in e.get("author", e.get("editor", "")).split(" and ") if a.strip()],
        year=year,
        venue=_clean(e.get("journaltitle") or e.get("journal") or e.get("booktitle")
                     or e.get("eventtitle") or e.get("institution") or ""),
        entry_type=e.get("ENTRYTYPE", ""),
        doi=e.get("doi", ""),
        url=e.get("url", ""),
        volume=e.get("volume", ""),
        number=e.get("number", ""),
        pages=e.get("pages", "").replace("--", "–"),
        publisher=_clean(e.get("publisher", "")),
        pdf_paths=PDF_IN_FILE_FIELD.findall(e.get("file", "")),
    )


def load_bib(bib_file: str) -> dict[str, Paper]:
    parser = BibTexParser(common_strings=True, ignore_nonstandard_types=False)
    parser.customization = convert_to_unicode
    with open(bib_file, encoding="utf-8") as fh:
        db = bibtexparser.load(fh, parser=parser)
    return {e["ID"]: _paper_from_entry(e) for e in db.entries}


def build_library(bib_file: str, storage_dir: str) -> dict[str, Paper]:
    """All bib entries, plus PDFs in Zotero storage that no bib entry links to."""
    papers = load_bib(bib_file)
    linked = {p for paper in papers.values() for p in paper.pdf_paths}
    for pdf in sorted(Path(storage_dir).glob("*/*.pdf")):
        if str(pdf) in linked:
            continue
        pid = f"zotero_{pdf.parent.name}"
        papers[pid] = Paper(paper_id=pid, title=pdf.stem, pdf_paths=[str(pdf)], in_bib=False)
    return papers
