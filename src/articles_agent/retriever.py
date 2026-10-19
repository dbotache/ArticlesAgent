"""Hybrid retrieval: dense (Chroma) + keyword (BM25), merged with reciprocal rank fusion."""

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .config import current_project, load_config
from .llm import embedding_deployment, get_embeddings
from .store import embed_text, get_collection, load_chunks, load_papers

TOKEN = re.compile(r"\w+", re.UNICODE)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1]


@dataclass
class Hit:
    chunk: dict
    paper: dict
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    project: str = ""

    @property
    def citation(self) -> str:
        c = self.chunk
        pages = f"p. {c['page_start']}" if c["page_start"] == c["page_end"] \
            else f"pp. {c['page_start']}–{c['page_end']}"
        return f"({self.paper['short_cite']}, {pages})"


class HybridRetriever:
    def __init__(self, cfg: dict | None = None):
        self.project = current_project()
        self.cfg = cfg or load_config()
        self.papers = load_papers(self.cfg)
        self.chunks = load_chunks(self.cfg)
        self.by_id = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25Okapi([tokenize(embed_text(c, self.papers[c["paper_id"]])) for c in self.chunks])
        self.collection = get_collection(self.cfg, embedding_deployment())
        self.embeddings = get_embeddings()

    def _allowed(self, paper_ids, year_from, year_to) -> np.ndarray | None:
        if not (paper_ids or year_from or year_to):
            return None
        ids = set(paper_ids or [])

        def ok(c):
            y = self.papers[c["paper_id"]].get("year", "")
            y = int(y) if y.isdigit() else 0
            return ((not ids or c["paper_id"] in ids)
                    and (not year_from or y >= year_from) and (not year_to or y <= year_to))
        return np.array([ok(c) for c in self.chunks])

    def _where(self, paper_ids, year_from, year_to) -> dict | None:
        conds = []
        if paper_ids:
            conds.append({"paper_id": {"$in": list(paper_ids)}})
        if year_from:
            conds.append({"year": {"$gte": year_from}})
        if year_to:
            conds.append({"year": {"$lte": year_to}})
        return None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}

    def search(self, query: str, k: int | None = None, candidates: int | None = None,
               paper_ids: list[str] | None = None, year_from: int | None = None,
               year_to: int | None = None) -> list[Hit]:
        k = k or self.cfg["retrieval"]["k"]
        n = candidates or self.cfg["retrieval"]["candidates"]

        dense = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)], n_results=n,
            where=self._where(paper_ids, year_from, year_to), include=[])["ids"][0]

        scores = self.bm25.get_scores(tokenize(query))
        if (mask := self._allowed(paper_ids, year_from, year_to)) is not None:
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:n]
        sparse = [self.chunks[i]["chunk_id"] for i in top if scores[i] > 0]

        fused: dict[str, Hit] = {}
        for ranks, attr in ((dense, "dense_rank"), (sparse, "bm25_rank")):
            for rank, cid in enumerate(ranks, start=1):
                if cid not in fused:
                    c = self.chunks[self.by_id[cid]]
                    fused[cid] = Hit(chunk=c, paper=self.papers[c["paper_id"]], score=0.0,
                                     project=self.project)
                fused[cid].score += 1 / (RRF_K + rank)
                setattr(fused[cid], attr, rank)
        return sorted(fused.values(), key=lambda h: -h.score)[:k]
