"""Ask the library a question and get cited findings.

    python -m articles_agent.ask "How is deep learning used to accelerate topology optimization?"
    python -m articles_agent.ask "..." --from 2020 --paper sasaki_explainable_2021 --json
    python -m articles_agent.ask "..." -o answer.md
"""

import argparse
import json
from pathlib import Path

from .config import add_project_option, apply_project_option
from .graph import ask
from .projects import list_projects


def main():
    ap = argparse.ArgumentParser()
    add_project_option(ap)
    ap.add_argument("question")
    ap.add_argument("--from", dest="year_from", type=int)
    ap.add_argument("--to", dest="year_to", type=int)
    ap.add_argument("--paper", action="append", help="restrict to citation key (repeatable)")
    ap.add_argument("--all-projects", action="store_true", help="search every project at once")
    ap.add_argument("--json", action="store_true", help="print the raw result as JSON")
    ap.add_argument("-o", "--output", type=Path, help="also write the Markdown answer to a file")
    args = ap.parse_args()
    apply_project_option(args)

    projects = [p.slug for p in list_projects()] if args.all_projects else None
    result = ask(args.question, paper_ids=args.paper, year_from=args.year_from, year_to=args.year_to,
                 projects=projects)
    md = result.to_markdown()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=1) if args.json else md)
    if args.output:
        args.output.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
