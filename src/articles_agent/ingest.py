"""Parse every PDF of the Zotero library into cited, page-aware chunks.

    python -m articles_agent.ingest            # incremental (re-uses parsed cache)
    python -m articles_agent.ingest --limit 5  # quick trial run
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_pages
from .config import load_config
from .metadata import build_library
from .parse import parse_pdf
from .store import parsed_cache_file


def _cache_file(cache_dir: Path, pdf: str) -> Path:
    return parsed_cache_file(cache_dir.parent, pdf)


def _parse_to_cache(pdf: str, cache: Path) -> tuple[str, str | None]:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        pages = parse_pdf(pdf)
    except Exception as exc:  # corrupt / encrypted PDFs must not stop the run
        return pdf, f"{type(exc).__name__}: {exc}"
    cache.write_text(json.dumps({"pdf": pdf, "pages": pages}, ensure_ascii=False), encoding="utf-8")
    return pdf, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only process the first N PDFs")
    ap.add_argument("--reparse", action="store_true", help="ignore the parsed-page cache")
    args = ap.parse_args()

    cfg = load_config()
    data_dir: Path = cfg["data_dir"]
    cache_dir = data_dir / "parsed"
    cache_dir.mkdir(parents=True, exist_ok=True)

    papers = build_library(cfg["library"]["bib_file"], cfg["library"]["storage_dir"])
    jobs = [(pid, pdf) for pid, p in papers.items() for pdf in p.pdf_paths if Path(pdf).exists()]
    if args.limit:
        jobs = jobs[: args.limit]

    todo = [pdf for _, pdf in jobs
            if args.reparse or not _cache_file(cache_dir, pdf).exists()
            or _cache_file(cache_dir, pdf).stat().st_mtime < Path(pdf).stat().st_mtime]
    errors = {}
    if todo:
        with ProcessPoolExecutor(max_workers=cfg["ingest"]["workers"]) as pool:
            futures = [pool.submit(_parse_to_cache, pdf, _cache_file(cache_dir, pdf)) for pdf in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs"):
                pdf, err = fut.result()
                if err:
                    errors[pdf] = err

    ccfg = cfg["chunking"]
    all_chunks, empty_pages = [], {}
    pdf_count = {}
    for pid, pdf in jobs:
        cache = _cache_file(cache_dir, pdf)
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
        "chunks": len(all_chunks),
        "reference_chunks": sum(c.is_references for c in all_chunks),
        "parse_errors": errors,
        "pdfs_with_blank_pages": {k: len(v) for k, v in empty_pages.items()},
    }
    (data_dir / "ingest_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print(f"{report['pdfs']} PDFs -> {report['chunks']} chunks "
          f"({report['reference_chunks']} reference-list chunks)")
    print(f"parse errors: {len(errors)}, PDFs with blank (scanned?) pages: {len(empty_pages)}")
    print(f"written to {data_dir}")


if __name__ == "__main__":
    main()
