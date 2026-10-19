"""Section-aware chunking that keeps the page range of every chunk."""

import re
import unicodedata
from dataclasses import asdict, dataclass

MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
BOLD_HEADING = re.compile(r"^\*\*((?:\d+(?:\.\d+)*\.?\s+)?[A-Z][^*]{1,80})\*\*$")
REFERENCES = re.compile(
    r"^(\d+\.?\s*)?(references|bibliography|literature cited|literatur(verzeichnis)?|quellen(verzeichnis)?)\b",
    re.IGNORECASE)
# LaTeX PDFs often emit a spacing diaeresis before the vowel: "L¨osung" -> "Lösung"
SPACING_UMLAUT = re.compile(r"¨\s?([AOUaeiouy])")
NOISE = re.compile(r"^(==> picture .* <==|\*\*----- (Start|End) of picture text -----\*\*.*)$")


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    pdf_path: str
    section: str
    page_start: int
    page_end: int
    is_references: bool
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _heading(line: str) -> str | None:
    m = MD_HEADING.match(line) or BOLD_HEADING.match(line)
    return re.sub(r"[*_]", "", m.group(1)).strip() if m else None


def clean_page_text(text: str) -> str:
    return SPACING_UMLAUT.sub(lambda m: unicodedata.normalize("NFC", m.group(1) + "\u0308"), text)


def _paragraphs(pages: list[dict]):
    """Yield (page, section, paragraph) in reading order."""
    section = "Front matter"
    for page in pages:
        for block in re.split(r"\n\s*\n", clean_page_text(page["text"])):
            lines = [ln for ln in block.strip().splitlines() if not NOISE.match(ln.strip())]
            if not lines:
                continue
            if heading := _heading(lines[0].strip()):
                section = heading
                lines = lines[1:]
            if text := "\n".join(lines).strip():
                yield page["page"], section, text


def _split_long(text: str, size: int) -> list[str]:
    """Split a paragraph longer than `size` at sentence boundaries."""
    parts, buf = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if buf and len(buf) + len(sentence) + 1 > size:
            parts.append(buf)
            buf = ""
        buf = f"{buf} {sentence}".strip()
        while len(buf) > size:  # single sentence too long (tables, formulas)
            parts.append(buf[:size])
            buf = buf[size:]
    if buf:
        parts.append(buf)
    return parts


def chunk_pages(pages: list[dict], paper_id: str, pdf_path: str,
                chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[tuple[int, str]] = []  # (page, paragraph)
    buf_section = None

    def flush(carry_overlap: bool):
        nonlocal buf
        if not buf:
            return
        chunks.append(Chunk(
            chunk_id=f"{paper_id}:{len(chunks):04d}",
            paper_id=paper_id,
            pdf_path=pdf_path,
            section=buf_section,
            page_start=min(p for p, _ in buf),
            page_end=max(p for p, _ in buf),
            is_references=bool(REFERENCES.match(buf_section)),
            text="\n\n".join(t for _, t in buf),
        ))
        carried, size = [], 0
        if carry_overlap:
            for p, t in reversed(buf):
                if size + len(t) > chunk_overlap:
                    break
                carried.insert(0, (p, t))
                size += len(t)
        buf = carried

    for page, section, para in _paragraphs(pages):
        if section != buf_section:
            flush(carry_overlap=False)
            buf_section = section
        for piece in _split_long(para, chunk_size):
            if buf and sum(len(t) for _, t in buf) + len(piece) > chunk_size:
                flush(carry_overlap=True)
            buf.append((page, piece))
    flush(carry_overlap=False)
    return chunks
