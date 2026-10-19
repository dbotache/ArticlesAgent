"""Embed searchable chunks into the persistent Chroma index.

    python -m articles_agent.index            # incremental: only new chunks
    python -m articles_agent.index --rebuild  # e.g. after changing the embedding model
"""

import argparse

import chromadb
from tqdm import tqdm

from .config import load_config
from .llm import embedding_deployment, get_embeddings
from .store import COLLECTION, embed_text, get_collection, load_chunks, load_papers


def _metadata(chunk: dict, paper: dict) -> dict:
    year = paper.get("year", "")
    return {
        "paper_id": chunk["paper_id"],
        "section": chunk["section"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "year": int(year) if year.isdigit() else 0,
        "short_cite": paper["short_cite"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild the index")
    args = ap.parse_args()

    cfg = load_config()
    model = embedding_deployment()
    if args.rebuild:
        client = chromadb.PersistentClient(path=str(cfg["data_dir"] / "chroma"))
        if COLLECTION in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION)
    col = get_collection(cfg, model, create=True)
    if (col.metadata or {}).get("embedding_model") != model:
        raise SystemExit(f"Index uses {col.metadata.get('embedding_model')}, not {model}: use --rebuild")

    papers = load_papers(cfg)
    chunks = load_chunks(cfg)
    wanted = {c["chunk_id"] for c in chunks}
    existing = set(col.get(include=[])["ids"])
    if stale := list(existing - wanted):  # chunks that disappeared after re-ingestion
        col.delete(ids=stale)
    todo = [c for c in chunks if c["chunk_id"] not in existing]
    print(f"{len(chunks)} searchable chunks, {len(existing & wanted)} already indexed, "
          f"{len(todo)} to embed with {model}")

    embeddings = get_embeddings()
    batch = cfg["index"]["batch_size"]
    for i in tqdm(range(0, len(todo), batch), desc="Embedding", unit="batch"):
        part = todo[i:i + batch]
        texts = [embed_text(c, papers[c["paper_id"]]) for c in part]
        col.add(
            ids=[c["chunk_id"] for c in part],
            embeddings=embeddings.embed_documents(texts),
            documents=[c["text"] for c in part],
            metadatas=[_metadata(c, papers[c["paper_id"]]) for c in part],
        )
    print(f"index now holds {col.count()} chunks")


if __name__ == "__main__":
    main()
