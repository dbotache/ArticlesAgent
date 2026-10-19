"""Manage library sources from the command line (the app's setup page does the same).

    python -m articles_agent.libraries list
    python -m articles_agent.libraries check  zotero ~/Zotero/export.bib --pdfs ~/Zotero/storage
    python -m articles_agent.libraries add    citavi ~/Citavi/export.bib --pdfs "~/Citavi/Projekt/Citavi Attachments"
    python -m articles_agent.libraries remove citavi
    python -m articles_agent.libraries sync          # re-read all sources, index new PDFs

Managers: zotero, jabref, citavi, mendeley, endnote, other.
"""

import argparse
import json

from .library import update_database
from .sources import MANAGERS, Source, add_source, load_sources, read_source, remove_source, validate


def _print_report(src: Source, scan: bool):
    rep = read_source(src, scan_folder=scan)
    print(f"{src.label}\n  references: {rep.references}, with PDF: {rep.with_pdf} "
          f"(linked {rep.linked}, matched {rep.matched})")
    if src.pdf_dir:
        print(f"  PDFs in folder: {rep.pdfs_in_folder}, not linked: {rep.unlinked_pdfs}"
              + (f", added as own papers: {rep.standalone}, duplicates: {rep.duplicates}" if scan else ""))
    if rep.broken_links:
        print(f"  broken file links: {rep.broken_links}")


def _sync(progress_json: bool = False):
    if progress_json:  # machine-readable progress for the app (library.update_database_isolated)
        def emit(msg):
            print("@@" + json.dumps(msg, ensure_ascii=False), flush=True)
        report = update_database(progress=lambda s, d, t: emit({"stage": s, "done": d, "total": t}))
        emit({"report": report})
        return
    report = update_database()
    ing, idx = report["ingest"], report["index"]
    print(f"{ing['pdfs']} PDFs ({ing['parsed']} parsed now), {idx['embedded']} passages embedded, "
          f"index holds {idx['total']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("sync").add_argument("--progress-json", action="store_true", help=argparse.SUPPRESS)
    for name in ("check", "add"):
        p = sub.add_parser(name)
        p.add_argument("manager", choices=list(MANAGERS))
        p.add_argument("reference_file")
        p.add_argument("--pdfs", default="", help="folder with the PDFs (searched recursively)")
        p.add_argument("--no-crossref", action="store_true")
        p.add_argument("--no-build", action="store_true", help="add only, don't build the database")
    p = sub.add_parser("remove")
    p.add_argument("id")
    args = ap.parse_args()

    if args.cmd == "list":
        for s in load_sources():
            print(f"{s.id:10} {s.label}\n           export: {s.reference_file}\n           pdfs:   {s.pdf_dir or '-'}")
    elif args.cmd == "check":
        if problems := validate(args.reference_file, args.pdfs):
            raise SystemExit(" ".join(problems))
        _print_report(Source("check", args.manager, args.reference_file, args.pdfs,
                             not args.no_crossref), scan=True)
    elif args.cmd == "add":
        src = add_source(args.manager, args.reference_file, args.pdfs, not args.no_crossref)
        print(f"added source '{src.id}'")
        if not args.no_build:
            _sync()
    elif args.cmd == "remove":
        remove_source(args.id)
        _sync()
    elif args.cmd == "sync":
        _sync(args.progress_json)


if __name__ == "__main__":
    main()
