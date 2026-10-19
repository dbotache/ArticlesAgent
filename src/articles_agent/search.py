"""Try retrieval from the command line (no LLM answer yet).

    python -m articles_agent.search "surrogate models for electric motor design" -k 8
    python -m articles_agent.search "SIMP acceleration" --from 2019 --paper kallioras_accelerated_2020
"""

import argparse

from .retriever import HybridRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--from", dest="year_from", type=int)
    ap.add_argument("--to", dest="year_to", type=int)
    ap.add_argument("--paper", action="append", help="restrict to citation key (repeatable)")
    args = ap.parse_args()

    hits = HybridRetriever().search(args.query, k=args.k, paper_ids=args.paper,
                                    year_from=args.year_from, year_to=args.year_to)
    for i, h in enumerate(hits, 1):
        src = f"dense #{h.dense_rank or '-'} / bm25 #{h.bm25_rank or '-'}"
        print(f"{i}. {h.citation}  [{h.chunk['section']}]  {src}")
        print(f"   {h.paper['title']}")
        print(f"   {h.chunk['text'][:300].replace(chr(10), ' ')} …\n")


if __name__ == "__main__":
    main()
