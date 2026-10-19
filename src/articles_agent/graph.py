"""LangGraph RAG pipeline that answers with verbatim, page-cited quotes.

rewrite → retrieve → grade ─┬→ generate → cite → END
   ↑                        │
   └── too few sources ─────┘ (at most `max_attempts` rounds)
"""

import re
from functools import cache
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .citations import cite_labels, format_reference, locate_page, match_quote
from .llm import get_chat
from .results import CitedFinding, Result
from .retriever import RRF_K, Hit, HybridRetriever
from .store import parsed_pages

# ------------------------------------------------------------ LLM output schemas


class Queries(BaseModel):
    queries: list[str] = Field(description="3-5 search queries")


class Relevant(BaseModel):
    source_ids: list[str] = Field(description="ids like 'S3' of useful sources, most useful first")


class Finding(BaseModel):
    claim: str = Field(description="what this passage shows, in the language of the question")
    quote: str = Field(description="1-3 sentences copied character-for-character from the source")
    source_id: str = Field(description="id of the single source the quote comes from, e.g. 'S2'")


class Answer(BaseModel):
    summary: str = Field(description="2-6 sentence synthesis citing sources inline as [S1], [S2]")
    findings: list[Finding]


# ------------------------------------------------------------ prompts

REWRITE_PROMPT = """You prepare search queries for a library of scientific papers \
(engineering, electrical machines, optimisation, machine learning; English and German).
Question: {question}

Write 3-5 diverse search queries that would find passages answering it. Include at least one \
English and one German formulation, the key technical terms, synonyms and common acronyms."""

GRADE_PROMPT = """Question: {question}

Below are numbered passages from scientific papers. Return the ids of the passages that contain \
information that directly helps answer the question, most useful first (at most {max_sources}). \
Leave out passages that only mention the topic in passing.

{sources}"""

ANSWER_SYSTEM = """You are a research assistant analysing the user's own library of scientific \
papers. Use ONLY the numbered sources provided.

Rules:
- Every finding cites exactly one source id and contains a quote copied verbatim, character for \
character, from that source: at most 3 consecutive sentences (about 60 words) — pick the \
sentences that carry the point, no ellipses, no paraphrasing, no merging of separate passages. \
Keep quotes in the source's language.
- Prefer passages with concrete statements (methods, results, numbers, conclusions) over general \
introductions. Give one finding per distinct point: never two findings that make the same point \
(e.g. a paper's abstract and its introduction saying the same). Up to 8 findings.
- Write the summary and the claims in the language of the question. Put the citation right after \
the sentence it supports, as [S1] or [S1, S3] — not all at the end.
- If the sources do not answer the question, say so in the summary and return no findings."""


# ------------------------------------------------------------ graph


class State(TypedDict, total=False):
    question: str
    paper_ids: list[str] | None
    year_from: int | None
    year_to: int | None
    attempt: int
    queries: list[str]
    candidates: list[Hit]
    sources: list[Hit]
    answer: Answer
    result: Result


@cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


def _cfg() -> dict:
    return get_retriever().cfg["answer"]


def _format_sources(hits: list[Hit], max_chars: int | None = None) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        text = h.chunk["text"] if max_chars is None else h.chunk["text"][:max_chars]
        blocks.append(f"[S{i}] {h.paper['short_cite']}: \"{h.paper['title']}\" "
                      f"— section: {h.chunk['section']} — {h.citation}\n{text}")
    return "\n\n---\n\n".join(blocks)


def rewrite(state: State) -> State:
    prompt = REWRITE_PROMPT.format(question=state["question"])
    if prev := state.get("queries"):
        prompt += (f"\n\nThese queries found too little relevant material: {prev}. "
                   "Use broader or alternative terminology.")
    llm = get_chat(fast=True).with_structured_output(Queries, method="json_schema", strict=True)
    queries = [state["question"]] + [q.strip() for q in llm.invoke(prompt).queries if q.strip()]
    return {"queries": list(dict.fromkeys(queries))[:6], "attempt": state.get("attempt", 0) + 1}


def retrieve(state: State) -> State:
    """Search with every query and fuse the ranked lists (reciprocal rank fusion)."""
    cfg, fused, scores = _cfg(), {}, {}
    for q in state["queries"]:
        hits = get_retriever().search(q, k=cfg["per_query_k"], paper_ids=state.get("paper_ids"),
                                   year_from=state.get("year_from"), year_to=state.get("year_to"))
        for rank, h in enumerate(hits, 1):
            cid = h.chunk["chunk_id"]
            fused.setdefault(cid, h)
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
    top = sorted(fused, key=lambda cid: -scores[cid])[:cfg["grade_candidates"]]
    return {"candidates": [fused[cid] for cid in top]}


def grade(state: State) -> State:
    cands = state["candidates"]
    if not cands:
        return {"sources": []}
    llm = get_chat(fast=True).with_structured_output(Relevant, method="json_schema", strict=True)
    out = llm.invoke(GRADE_PROMPT.format(question=state["question"], max_sources=_cfg()["max_sources"],
                                         sources=_format_sources(cands, max_chars=1500)))
    picked = []
    for sid in out.source_ids:
        m = re.fullmatch(r"S?(\d+)", sid.strip())
        if m and 1 <= int(m.group(1)) <= len(cands) and cands[int(m.group(1)) - 1] not in picked:
            picked.append(cands[int(m.group(1)) - 1])
    return {"sources": picked[:_cfg()["max_sources"]]}


def route_after_grade(state: State) -> str:
    if len(state["sources"]) < _cfg()["min_sources"] and state["attempt"] < _cfg()["max_attempts"]:
        return "rewrite"
    return "generate"


def generate(state: State) -> State:
    if not state["sources"]:
        return {"answer": Answer(summary="No passages in the library address this question.",
                                 findings=[])}
    llm = get_chat().with_structured_output(Answer, method="json_schema", strict=True)
    answer = llm.invoke([
        SystemMessage(ANSWER_SYSTEM),
        HumanMessage(f"Question: {state['question']}\n\nSources:\n\n{_format_sources(state['sources'])}"),
    ])
    return {"answer": answer}


def cite(state: State) -> State:
    """Verify every quote against the source chunk, find its page, build references."""
    sources, answer = state["sources"], state["answer"]
    data_dir = get_retriever().cfg["data_dir"]
    threshold = _cfg()["fuzzy_threshold"]

    def source(sid: str) -> Hit | None:
        m = re.fullmatch(r"S?(\d+)", sid.strip())
        return sources[int(m.group(1)) - 1] if m and 1 <= int(m.group(1)) <= len(sources) else None

    verified, dropped = [], 0
    for f in answer.findings:
        hit = source(f.source_id)
        match = hit and match_quote(f.quote, hit.chunk["text"], threshold)
        if not match:
            dropped += 1
            continue
        quote, exact = match
        c = hit.chunk
        page = locate_page(quote, parsed_pages(data_dir, c["pdf_path"]), c["page_start"], c["page_end"])
        verified.append((f, hit, quote, exact, page))

    cited_ids = set(re.findall(r"S(\d+)", answer.summary))
    cited_papers = {h.paper["paper_id"]: h.paper for _, h, *_ in verified}
    cited_papers |= {sources[int(i) - 1].paper["paper_id"]: sources[int(i) - 1].paper
                     for i in cited_ids if 1 <= int(i) <= len(sources)}
    labels = cite_labels(list(cited_papers.values()))

    def pages_of(hit: Hit) -> list[int]:
        # prefer the page of a verified quote from this chunk
        if page := next((p for _, h, _, _, p in verified if h is hit), None):
            return [page]
        return list(range(hit.chunk["page_start"], hit.chunk["page_end"] + 1))

    def summary_marker(m: re.Match) -> str:
        """[S1, S4][S2] -> (Doi et al., 2019, pp. 1, 4; Sasaki et al., 2021, p. 2)"""
        by_paper: dict[str, set[int]] = {}
        for sid in re.findall(r"S\d+", m.group(0)):
            if h := source(sid):
                by_paper.setdefault(h.paper["paper_id"], set()).update(pages_of(h))
        cites = [f"{labels[pid]}, {'p.' if len(pages) == 1 else 'pp.'} "
                 f"{', '.join(map(str, sorted(pages)))}" for pid, pages in by_paper.items()]
        return f" ({'; '.join(cites)})" if cites else ""

    # a run of adjacent markers like "[S1][S3], [S1]" becomes one de-duplicated citation
    marker = r"\[S\d+(?:\s*[,;]\s*S\d+)*\]"
    summary = re.sub(rf"\s*{marker}(?:\s*[,;]?\s*{marker})*", summary_marker, answer.summary)
    findings = [CitedFinding(claim=f.claim, quote=quote, exact=exact, paper_id=h.paper["paper_id"],
                             page=page, section=h.chunk["section"], chunk_id=h.chunk["chunk_id"],
                             citation=f"({labels[h.paper['paper_id']]}, p. {page})")
                for f, h, quote, exact, page in verified]
    refs = sorted(format_reference(p, labels[pid].rsplit(", ", 1)[-1])
                  for pid, p in cited_papers.items())
    return {"result": Result(question=state["question"], summary=summary, findings=findings,
                             references=refs, queries=state["queries"], dropped_quotes=dropped)}


@cache
def get_graph():
    g = StateGraph(State)
    for name, fn in (("rewrite", rewrite), ("retrieve", retrieve), ("grade", grade),
                     ("generate", generate), ("cite", cite)):
        g.add_node(name, fn)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, ["rewrite", "generate"])
    g.add_edge("generate", "cite")
    g.add_edge("cite", END)
    return g.compile()


def ask(question: str, paper_ids: list[str] | None = None,
        year_from: int | None = None, year_to: int | None = None) -> Result:
    state = get_graph().invoke({"question": question, "paper_ids": paper_ids,
                                "year_from": year_from, "year_to": year_to})
    return state["result"]



def ask_stream(question: str, paper_ids: list[str] | None = None,
               year_from: int | None = None, year_to: int | None = None):
    """Yield (node_name, state_update) while the graph runs; the 'cite' update holds the Result."""
    inputs = {"question": question, "paper_ids": paper_ids, "year_from": year_from, "year_to": year_to}
    for update in get_graph().stream(inputs, stream_mode="updates"):
        yield from update.items()
