"""Ask the library a question and get cited findings.

    python -m articles_agent.ask "How is deep learning used to accelerate topology optimization?"
    python -m articles_agent.ask "..." --from 2020 --paper sasaki_explainable_2021 --json
    python -m articles_agent.ask "..." -o answer.md
"""

import argparse
import json
from pathlib import Path

from .graph import ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--from", dest="year_from", type=int)
    ap.add_argument("--to", dest="year_to", type=int)
    ap.add_argument("--paper", action="append", help="restrict to citation key (repeatable)")
    ap.add_argument("--json", action="store_true", help="print the raw result as JSON")
    ap.add_argument("-o", "--output", type=Path, help="also write the Markdown answer to a file")
    args = ap.parse_args()

    result = ask(args.question, paper_ids=args.paper, year_from=args.year_from, year_to=args.year_to)
    md = result.to_markdown()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=1) if args.json else md)
    if args.output:
        args.output.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
