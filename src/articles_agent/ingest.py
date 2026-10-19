"""Parse every PDF of the library into cited, page-aware chunks.

The library is defined by the sources in library/sources.json (reference-manager exports +
PDF folders) plus papers imported via `articles_agent.add` / the UI.

    python -m articles_agent.ingest            # incremental (re-uses parsed cache)
    python -m articles_agent.ingest --limit 5  # quick trial run
"""

import argparse
import json
import multiprocessing
import os
import sys
import types
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_pages
from .config import add_project_option, apply_project_option, load_config, shared_dir
from .parse import parse_pdf
from .sources import build_library
from .store import parsed_cache_file


def _parse_to_cache(pdf: str, cache: Path) -> tuple[str, str | None]:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        pages = parse_pdf(pdf)
    except Exception as exc:  # corrupt / encrypted PDFs must not stop the run
        return pdf, f"{type(exc).__name__}: {exc}"
    cache.write_text(json.dumps({"pdf": pdf, "pages": pages}, ensure_ascii=False), encoding="utf-8")
    return pdf, None


@contextmanager
def _workers_skip_main():
    """Spawned workers re-import the parent's __main__ module. If the host is a script such as
    the Streamlit app (or a test running it), each worker would re-run that script. The workers
    only need this module, so hide __main__ from them while they start."""
    main = sys.modules["__main__"]
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        sys.modules["__main__"] = main


def run_ingest(reparse: bool = False, limit: int | None = None, progress=None) -> dict:
    """Parse new/changed PDFs, re-chunk everything. progress(stage, done, total) is optional."""
    cfg = load_config()
    data_dir: Path = cfg["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir() / "parsed").mkdir(parents=True, exist_ok=True)

    papers = build_library(progress=progress)
    jobs = [(pid, pdf) for pid, p in papers.items() for pdf in p.pdf_paths if Path(pdf).exists()]
    if limit:
        jobs = jobs[:limit]

    def stale(pdf: str) -> bool:
        cache = parsed_cache_file(pdf)
        return reparse or not cache.exists() or cache.stat().st_mtime < Path(pdf).stat().st_mtime

    todo = [pdf for _, pdf in jobs if stale(pdf)]
    errors = {}
    if todo:
        # spawn: safe to start from threaded hosts such as the Streamlit app
        ctx = multiprocessing.get_context("spawn")
        with _workers_skip_main(), ProcessPoolExecutor(
                max_workers=min(cfg["ingest"]["workers"], len(todo)), mp_context=ctx) as pool:
            futures = [pool.submit(_parse_to_cache, pdf, parsed_cache_file(pdf))
                       for pdf in todo]
            done = tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs",
                        disable=progress is not None)
            for n, fut in enumerate(done, 1):
                pdf, err = fut.result()
                if err:
                    errors[pdf] = err
                if progress:
                    progress("parse", n, len(todo))

    ccfg = cfg["chunking"]
    all_chunks, empty_pages = [], {}
    pdf_count = {}
    for pid, pdf in jobs:
        cache = parsed_cache_file(pdf)
        if not cache.exists():
            continue
        pages = json.loads(cache.read_text(encoding="utf-8"))["pages"]
        blank = [p["page"] for p in pages if len(p["text"].strip()) < 50]
        if blank:
            empty_pages[pdf] = blank
        # entries with several PDFs get distinct chunk-id prefixes
        n = pdf_count[pid] = pdf_count.get(pid, 0) + 1
        prefix = pid if n == 1 else f"{pid}#{n}"
        chunks = chunk_pages(pages, prefix, pdf, ccfg["chunk_size"], ccfg["chunk_overlap"])
        for c in chunks:
            c.paper_id = pid
        all_chunks.extend(chunks)

    with open(data_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    (data_dir / "papers.json").write_text(
        json.dumps({pid: p.to_dict() for pid, p in papers.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    report = {
        "pdfs": len(jobs),
        "parsed": len(todo),
        "chunks": len(all_chunks),
        "reference_chunks": sum(c.is_references for c in all_chunks),
        "parse_errors": errors,
        "pdfs_with_blank_pages": {k: len(v) for k, v in empty_pages.items()},
        "chunks_per_paper": dict(Counter(c.paper_id for c in all_chunks)),
    }
    (data_dir / "ingest_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def main():
    ap = argparse.ArgumentParser()
    add_project_option(ap)
    ap.add_argument("--limit", type=int, help="only process the first N PDFs")
    ap.add_argument("--reparse", action="store_true", help="ignore the parsed-page cache")
    args = ap.parse_args()
    apply_project_option(args)

    report = run_ingest(reparse=args.reparse, limit=args.limit)
    print(f"{report['pdfs']} PDFs ({report['parsed']} parsed now) -> {report['chunks']} chunks "
          f"({report['reference_chunks']} reference-list chunks)")
    print(f"parse errors: {len(report['parse_errors'])}, "
          f"PDFs with blank (scanned?) pages: {len(report['pdfs_with_blank_pages'])}")
    print(f"written to {load_config()['data_dir']}")


if __name__ == "__main__":
    main()
