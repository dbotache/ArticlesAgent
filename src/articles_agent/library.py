"""Add individual PDFs that belong to no library source, as single files or whole folders.

Imported papers are recorded in the project's library/extra_papers.json, which `sources.build_library`
merges with the library sources, so re-ingestion keeps them. Uploaded files are copied to
library/uploads/; PDFs from a folder are used in place.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import fields
from datetime import date
from pathlib import Path

from rapidfuzz import fuzz, process

from .config import APP_HOME, current_project, project_dir
from .metadata import Paper, make_id, norm_doi, norm_title
from .pdfmeta import metadata_from_pdf, save_caches
from .sources import build_library



def library_dir() -> Path:
    return project_dir() / "library"


def registry_file() -> Path:
    return library_dir() / "extra_papers.json"


def uploads_dir() -> Path:
    return library_dir() / "uploads"

PAPER_FIELDS = {f.name for f in fields(Paper)}


# ------------------------------------------------------------ registry

def load_registry() -> dict[str, dict]:
    return json.loads(registry_file().read_text(encoding="utf-8")) if registry_file().exists() else {}


def registry_papers() -> dict[str, Paper]:
    return {pid: Paper(**{k: v for k, v in entry.items() if k in PAPER_FIELDS})
            for pid, entry in load_registry().items()}


def _save_registry(reg: dict) -> None:
    library_dir().mkdir(parents=True, exist_ok=True)
    tmp = registry_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(registry_file())


# ------------------------------------------------------------ files

def collect_pdfs(folder: str | Path) -> list[Path]:
    """All PDFs below `folder`, recursively."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".pdf" and p.is_file())


def save_upload(name: str, data: bytes) -> Path:
    """Store an uploaded PDF under library/uploads/<content hash>/<name>."""
    target = uploads_dir() / hashlib.sha256(data).hexdigest()[:12] / Path(name).name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return target


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------ public API

def register_pdfs(paths: list[str | Path], use_crossref: bool = True, progress=None) -> list[dict]:
    """Record new PDFs in the registry. Returns one status dict per file:
    {file, path, status: added|skipped|error, detail, paper_id?}. Nothing is indexed yet."""
    existing = {pid: p.to_dict() for pid, p in build_library().items()}
    reg = load_registry()
    known_hashes = {e.get("sha256") for e in reg.values()}
    known_paths = {path for e in existing.values() for path in e.get("pdf_paths", [])}
    known_dois = {norm_doi(e["doi"]): pid for pid, e in existing.items() if e.get("doi")}
    known_titles = {norm_title(e["title"]): pid for pid, e in existing.items() if e.get("title")}

    results = []
    for n, raw in enumerate(paths, 1):
        path = Path(raw).expanduser().resolve()
        status = {"file": path.name, "path": str(path)}
        try:
            digest = _sha256(path)
            if str(path) in known_paths:
                status.update(status="skipped", detail="already in the library")
            elif digest in known_hashes:
                status.update(status="skipped", detail="same file was imported before")
            else:
                info, source = metadata_from_pdf(path, use_crossref)
                same_title = process.extractOne(norm_title(info["title"]), list(known_titles),
                                                scorer=fuzz.ratio, score_cutoff=95)
                if info["doi"] and norm_doi(info["doi"]) in known_dois:
                    status.update(status="skipped",
                                  detail=f"DOI already in the library as {known_dois[norm_doi(info['doi'])]}")
                elif same_title:
                    status.update(status="skipped",
                                  detail=f"same title already in the library as {known_titles[same_title[0]]}")
                else:
                    pid = make_id(info["title"], info["authors"], info["year"], set(existing) | set(reg))
                    reg[pid] = {**info, "paper_id": pid, "pdf_paths": [str(path)], "in_bib": False,
                                "source": "import", "sha256": digest,
                                "added": date.today().isoformat(), "metadata_source": source}
                    existing[pid] = reg[pid]
                    known_hashes.add(digest)
                    known_paths.add(str(path))
                    if info["doi"]:
                        known_dois[norm_doi(info["doi"])] = pid
                    known_titles[norm_title(info["title"])] = pid
                    status.update(status="added", paper_id=pid,
                                  detail=f"{info['title'][:80]} (metadata: {source})")
        except Exception as exc:  # unreadable / encrypted PDF
            status.update(status="error", detail=f"{type(exc).__name__}: {exc}")
        results.append(status)
        if progress:
            progress("register", n, len(paths))
    _save_registry(reg)
    save_caches()
    return results


def update_database(progress=None) -> dict:
    """Read all sources, parse + chunk new PDFs, then embed new chunks."""
    from .index import run_index
    from .ingest import run_ingest
    ingest = run_ingest(progress=progress)
    index = run_index(progress=progress)
    return {"ingest": ingest, "index": index}


def update_database_isolated(progress=None) -> dict:
    """update_database() in a separate Python process, reporting progress back.

    Used by the Streamlit app: there, the PDF parser's worker processes would re-run the app
    script (Streamlit's __main__) instead of starting cleanly."""
    src_dir = str(Path(__file__).resolve().parents[1])
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [src_dir, os.environ.get("PYTHONPATH")])),
                        "ARTICLES_AGENT_HOME": str(APP_HOME), "ARTICLES_AGENT_PROJECT": current_project()}
    report = None
    with tempfile.TemporaryFile("w+", encoding="utf-8") as err:
        proc = subprocess.Popen([sys.executable, "-m", "articles_agent.libraries", "sync", "--progress-json"],
                                cwd=APP_HOME, env=env, stdout=subprocess.PIPE, stderr=err, text=True)
        for line in proc.stdout:
            if not line.startswith("@@"):
                continue  # library chatter (e.g. pymupdf notices)
            msg = json.loads(line[2:])
            if "report" in msg:
                report = msg["report"]
            elif progress:
                progress(msg["stage"], msg["done"], msg["total"])
        if proc.wait() != 0 or report is None:
            err.seek(0)
            tail = err.read().strip().splitlines()[-3:]
            raise RuntimeError("Database update failed: " + (" | ".join(tail) or f"exit code {proc.returncode}"))
    return report
