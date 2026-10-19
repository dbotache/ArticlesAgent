"""Eyeball the ingestion result.

    python -m articles_agent.inspect_chunks                 # library-wide stats
    python -m articles_agent.inspect_chunks kallioras_accelerated_2020
"""

import json
import sys
from collections import Counter
from statistics import mean

from .config import load_config


def main():
    data_dir = load_config()["data_dir"]
    papers = json.loads((data_dir / "papers.json").read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in open(data_dir / "chunks.jsonl", encoding="utf-8")]

    if len(sys.argv) > 1:
        pid = sys.argv[1]
        p = papers[pid]
        print(f"{p['short_cite']} — {p['title']}  ({p['venue']}, doi:{p['doi']})\n")
        for c in (c for c in chunks if c["paper_id"] == pid):
            ref = " [REFERENCES]" if c["is_references"] else ""
            print(f"=== {c['chunk_id']} | {c['section']} | p.{c['page_start']}-{c['page_end']}"
                  f" | {len(c['text'])} chars{ref}")
            print(c["text"][:400].replace("\n", " ") + " …\n")
        return

    lengths = [len(c["text"]) for c in chunks]
    per_paper = Counter(c["paper_id"] for c in chunks)
    print(f"papers in bib: {sum(p['in_bib'] for p in papers.values())}, "
          f"with chunks: {len(per_paper)}, chunks: {len(chunks)}")
    print(f"chunk length: mean {mean(lengths):.0f}, min {min(lengths)}, max {max(lengths)} chars")
    print("most common sections:", Counter(c["section"] for c in chunks).most_common(15))
    print("papers with fewest chunks:", per_paper.most_common()[-5:])


if __name__ == "__main__":
    main()
