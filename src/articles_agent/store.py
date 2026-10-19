"""Shared access to ingested papers/chunks and the Chroma collection."""

import json
import re
from functools import cache
from pathlib import Path

import chromadb

from .chunking import clean_page_text

COLLECTION = "chunks"


def load_papers(cfg: dict) -> dict[str, dict]:
    return json.loads((cfg["data_dir"] / "papers.json").read_text(encoding="utf-8"))


@cache
def _exclude_re(patterns: tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


def load_chunks(cfg: dict) -> list[dict]:
    """Searchable chunks: reference lists and index/contents sections are dropped."""
    exclude = _exclude_re(tuple(cfg["index"]["exclude_sections"]))
    skip_refs = cfg["chunking"]["exclude_references"]
    chunks = []
    with open(cfg["data_dir"] / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            if (skip_refs and c["is_references"]) or exclude.search(c["section"].strip()):
                continue
            if len(c["text"]) < cfg["index"]["min_chars"]:
                continue
            chunks.append(c)
    return chunks


def parsed_cache_file(data_dir: Path, pdf: str) -> Path:
    # Zotero storage key (parent folder) is unique per attachment
    return data_dir / "parsed" / f"{Path(pdf).parent.name}.json"


@cache
def parsed_pages(data_dir: Path, pdf: str) -> dict[int, str]:
    """Cleaned full text of every page of one PDF, from the ingestion cache."""
    pages = json.loads(parsed_cache_file(data_dir, pdf).read_text(encoding="utf-8"))["pages"]
    return {p["page"]: clean_page_text(p["text"]) for p in pages}


def embed_text(chunk: dict, paper: dict) -> str:
    """Text that gets embedded / BM25-indexed: title and section give each chunk context."""
    return f"{paper['title']}\n{chunk['section']}\n\n{chunk['text']}"


def get_collection(cfg: dict, embedding_model: str, create: bool = False):
    client = chromadb.PersistentClient(path=str(cfg["data_dir"] / "chroma"))
    if create:
        return client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine", "embedding_model": embedding_model})
    col = client.get_collection(COLLECTION)
    built_with = (col.metadata or {}).get("embedding_model")
    if built_with != embedding_model:
        raise RuntimeError(
            f"Index was built with '{built_with}' but .env uses '{embedding_model}'. "
            "Run `python -m articles_agent.index --rebuild`.")
    return col
