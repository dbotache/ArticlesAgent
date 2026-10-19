"""Shared embedding cache: one vector per (embedding model, passage text), for all projects.

A paper that is part of several projects is embedded once; building a second project from the
same library costs no Azure calls for the passages already known.
"""

import hashlib
import sqlite3
from array import array
from contextlib import closing

from .config import shared_dir


def _db() -> sqlite3.Connection:
    shared_dir().mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(shared_dir() / "embeddings.sqlite", timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
    return con


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()


def get_many(keys: list[str]) -> dict[str, list[float]]:
    found = {}
    with closing(_db()) as con:
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            rows = con.execute(f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(part))})", part)
            for key, blob in rows:
                found[key] = array("f", blob).tolist()
    return found


def put_many(items: dict[str, list[float]]) -> None:
    with closing(_db()) as con, con:
        con.executemany("INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
                        [(k, array("f", v).tobytes()) for k, v in items.items()])


def seed_from_index() -> int:
    """Copy the active project's indexed vectors into the cache (used by the migration)."""
    from .config import load_config
    from .llm import embedding_deployment
    from .store import embed_text, get_collection, load_chunks, load_papers

    cfg = load_config()
    model = embedding_deployment()
    col = get_collection(cfg, model)
    papers = load_papers(cfg)
    chunks = {c["chunk_id"]: c for c in load_chunks(cfg)}
    seeded = 0
    total = col.count()
    for offset in range(0, total, 2000):
        got = col.get(include=["embeddings"], limit=2000, offset=offset)
        items = {cache_key(model, embed_text(chunks[cid], papers[chunks[cid]["paper_id"]])): list(vec)
                 for cid, vec in zip(got["ids"], got["embeddings"]) if cid in chunks}
        put_many(items)
        seeded += len(items)
    return seeded
