"""Manage library sources from the command line (the app's setup page does the same).

    python -m articles_agent.libraries list
    python -m articles_agent.libraries check  zotero ~/Zotero/export.bib --pdfs ~/Zotero/storage
    python -m articles_agent.libraries add    citavi ~/Citavi/export.bib --pdfs "~/Citavi/Projekt/Citavi Attachments"
    python -m articles_agent.libraries add    zotero ~/Zotero/all.bib --pdfs ~/Zotero/storage --keywords motor-design
    python -m articles_agent.libraries keywords ~/Zotero/all.bib        # tags in an export, with counts
    python -m articles_agent.libraries filter zotero motor-design PMSM  # change a library's tag filter
    python -m articles_agent.libraries remove citavi
    python -m articles_agent.libraries sync          # re-read all sources, index new PDFs

All commands work on the active project; choose another with --project <slug>.

Managers: zotero, jabref, citavi, mendeley, endnote, other.
"""

import argparse
import json

from .config import add_project_option, apply_project_option
from .library import update_database
from .sources import (MANAGERS, Source, add_source, keyword_counts, load_sources, read_source,
                      remove_source, set_keywords, validate)


def _print_report(src: Source, scan: bool):
    rep = read_source(src, scan_folder=scan)
    print(f"{src.label}  [{src.filter_label}]\n  references: {rep.references}, with PDF: {rep.with_pdf} "
          f"(linked {rep.linked}, matched {rep.matched})"
          + (f", excluded by the keyword filter: {rep.filtered_out}" if src.keywords else ""))
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
    print(f"{ing['pdfs']} PDFs ({ing['parsed']} parsed now), {idx['added']} passages added "
          f"({idx['reused']} from the shared cache), index holds {idx['total']}")


def main():
    ap = argparse.ArgumentParser()
    add_project_option(ap)
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
        p.add_argument("--keywords", nargs="*", default=[],
                       help="only references tagged with one of these (Zotero tags, Citavi keywords, …)")
    p = sub.add_parser("remove")
    p.add_argument("id")
    p = sub.add_parser("keywords")
    p.add_argument("reference_file")
    p = sub.add_parser("filter")
    p.add_argument("id")
    p.add_argument("keywords", nargs="*", help="none = no filter")
    args = ap.parse_args()
    apply_project_option(args)

    if args.cmd == "list":
        for s in load_sources():
            print(f"{s.id:10} {s.label}\n           export: {s.reference_file}\n           pdfs:   {s.pdf_dir or '-'}"
                  f"\n           filter: {s.filter_label}")
    elif args.cmd == "check":
        if problems := validate(args.reference_file, args.pdfs):
            raise SystemExit(" ".join(problems))
        _print_report(Source("check", args.manager, args.reference_file, args.pdfs,
                             not args.no_crossref, args.keywords), scan=True)
    elif args.cmd == "add":
        src = add_source(args.manager, args.reference_file, args.pdfs, not args.no_crossref, args.keywords)
        print(f"added source '{src.id}'")
        if not args.no_build:
            _sync()
    elif args.cmd == "remove":
        remove_source(args.id)
        _sync()
    elif args.cmd == "sync":
        _sync(args.progress_json)
    elif args.cmd == "keywords":
        for kw, n in keyword_counts(args.reference_file).items():
            print(f"{n:6}  {kw}")
    elif args.cmd == "filter":
        set_keywords(args.id, args.keywords)
        _sync()


if __name__ == "__main__":
    main()
