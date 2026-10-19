"""Measure retrieval quality against questions with known answers (eval/questions.yaml).

    python -m articles_agent.eval_retrieval
"""

import yaml

from .config import PROJECT_ROOT
from .retriever import HybridRetriever


def main():
    cases = yaml.safe_load((PROJECT_ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    k = 10
    hits_at = {1: 0, 5: 0, 10: 0}
    page_hits, mrr = 0, 0.0

    for case in cases:
        expected = set(case["papers"])
        results = retriever.search(case["question"], k=k)
        rank = next((i for i, h in enumerate(results, 1) if h.paper["paper_id"] in expected), None)
        page_ok = "pages" in case and any(
            h.paper["paper_id"] in expected
            and any(h.chunk["page_start"] <= p <= h.chunk["page_end"] for p in case["pages"])
            for h in results)
        for n in hits_at:
            hits_at[n] += bool(rank and rank <= n)
        mrr += 1 / rank if rank else 0
        page_hits += page_ok
        mark = f"rank {rank}" if rank else "MISS"
        print(f"[{mark:>7}] {case['question']}" + ("  (page ✓)" if page_ok else ""))

    n = len(cases)
    print(f"\n{n} questions | " + " | ".join(f"hit@{a}: {b / n:.0%}" for a, b in hits_at.items())
          + f" | MRR: {mrr / n:.2f} | right page in top {k}: {page_hits}/{sum('pages' in c for c in cases)}")


if __name__ == "__main__":
    main()
