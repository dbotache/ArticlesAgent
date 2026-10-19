"""The answer object shared by the CLI, the chat UI and batch reports."""

from dataclasses import asdict, dataclass, field


@dataclass
class CitedFinding:
    claim: str
    quote: str            # verbatim text from the PDF
    exact: bool           # False: LLM quote was slightly off and snapped to the source text
    paper_id: str
    page: int             # physical PDF page
    section: str
    chunk_id: str
    citation: str         # "(Kallioras et al., 2020, p. 6)"


@dataclass
class Result:
    question: str
    summary: str
    findings: list[CitedFinding] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    dropped_quotes: int = 0  # findings removed because the quote was not in the source

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self, heading_level: int = 2) -> str:
        h = "#" * heading_level
        out = [f"{h} {self.question}", "", self.summary, ""]
        if self.findings:
            out += [f"{h}# Findings", ""]
            for i, f in enumerate(self.findings, 1):
                out += [f"**{i}. {f.claim}**", "", f"> “{f.quote}”", ">",
                        f"> — {f.citation}, section *{f.section}*", ""]
        if self.dropped_quotes:
            out += [f"_{self.dropped_quotes} finding(s) omitted: quote not found in the source._", ""]
        if self.references:
            out += [f"{h}# References", ""] + [f"- {r}" for r in self.references] + [""]
        return "\n".join(out)
