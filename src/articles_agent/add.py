"""Add PDFs from outside Zotero to the library and index them.

    python -m articles_agent.add paper.pdf other.pdf      # single files
    python -m articles_agent.add ~/Papers/new_batch       # folder (recursive)
    python -m articles_agent.add ~/Papers --no-crossref   # don't look up DOIs online
"""

import argparse
from pathlib import Path

from .config import add_project_option, apply_project_option
from .library import collect_pdfs, register_pdfs, update_database


def main():
    ap = argparse.ArgumentParser()
    add_project_option(ap)
    ap.add_argument("paths", nargs="+", help="PDF files and/or folders")
    ap.add_argument("--no-crossref", action="store_true", help="skip the Crossref DOI lookup")
    args = ap.parse_args()
    apply_project_option(args)

    pdfs = []
    for p in map(Path, args.paths):
        pdfs += collect_pdfs(p) if p.is_dir() else [p]
    results = register_pdfs(pdfs, use_crossref=not args.no_crossref)
    for r in results:
        print(f"[{r['status']:>7}] {r['file']}: {r['detail']}")

    if any(r["status"] == "added" for r in results):
        report = update_database()
        per_paper = report["ingest"]["chunks_per_paper"]
        for r in results:
            if r["status"] == "added" and not per_paper.get(r["paper_id"]):
                print(f"warning: no text extracted from {r['file']} (scanned PDF?)")
        print(f"indexed {report['index']['added']} new chunks; "
              f"index holds {report['index']['total']} chunks")


if __name__ == "__main__":
    main()
