"""Embed the active project's searchable chunks into its Chroma index.

Vectors come from the shared embedding cache when another project (or an earlier build)
already embedded the same passage; only the rest is sent to Azure.

    python -m articles_agent.index            # incremental: only new chunks
    python -m articles_agent.index --rebuild  # e.g. after changing the embedding model
"""

import argparse

import chromadb
from tqdm import tqdm

from . import embcache
from .config import add_project_option, apply_project_option, load_config
from .config import current_project
from .llm import get_embeddings
from .projects import embedding_pin, embedding_spec, pin_embedding
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
    add_project_option(ap)
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild the index")
    args = ap.parse_args()
    apply_project_option(args)
    report = run_index(rebuild=args.rebuild)
    print(f"{report['searchable']} searchable chunks, {report['added']} added "
          f"({report['embedded']} embedded with {report['model']}, {report['reused']} from the shared cache); "
          f"index now holds {report['total']} chunks")


def run_index(rebuild: bool = False, progress=None) -> dict:
    """Embed all searchable chunks not yet in the index. progress(stage, done, total) is optional."""
    cfg = load_config()
    spec = embedding_spec()  # pinned for this project (or the default before its first build)
    model = spec.identity
    if rebuild:
        client = chromadb.PersistentClient(path=str(cfg["data_dir"] / "chroma"))
        if COLLECTION in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION)
    col = get_collection(cfg, model, create=True)
    if (col.metadata or {}).get("embedding_model") != model:
        # only reachable after the project's embedding model was changed on purpose (re-embedding):
        # the pin decides, so replace the index built with the old model
        chromadb.PersistentClient(path=str(cfg["data_dir"] / "chroma")).delete_collection(COLLECTION)
        col = get_collection(cfg, model, create=True)

    papers = load_papers(cfg)
    chunks = load_chunks(cfg)
    wanted = {c["chunk_id"] for c in chunks}
    existing = set(col.get(include=[])["ids"])
    if stale := list(existing - wanted):  # chunks that disappeared after re-ingestion
        col.delete(ids=stale)
    todo = [c for c in chunks if c["chunk_id"] not in existing]

    embeddings = get_embeddings(spec)
    batch = cfg["index"]["batch_size"]
    starts = range(0, len(todo), batch)
    embedded = reused = 0
    dims = None
    for n, i in enumerate(tqdm(starts, desc="Embedding", unit="batch", disable=progress is not None), 1):
        part = todo[i:i + batch]
        texts = [embed_text(c, papers[c["paper_id"]]) for c in part]
        keys = [embcache.cache_key(model, t) for t in texts]
        vectors = embcache.get_many(keys)
        if missing := [j for j, k in enumerate(keys) if k not in vectors]:
            new = embeddings.embed_documents([texts[j] for j in missing])
            fresh = {keys[j]: vec for j, vec in zip(missing, new)}
            embcache.put_many(fresh)
            vectors |= fresh
        embedded += len(missing)
        dims = dims or len(next(iter(vectors.values())))
        reused += len(part) - len(missing)
        col.add(
            ids=[c["chunk_id"] for c in part],
            embeddings=[vectors[k] for k in keys],
            documents=[c["text"] for c in part],
            metadatas=[_metadata(c, papers[c["paper_id"]]) for c in part],
        )
        if progress:
            progress("embed", n, len(starts))
    pin = embedding_pin()
    if not pin or pin.get("identity") != model or (dims and not pin.get("dimensions")):
        pin_embedding(current_project(), spec, dims or (pin or {}).get("dimensions"))
    return {"searchable": len(chunks), "added": len(todo), "embedded": embedded, "reused": reused,
            "removed": len(stale), "total": col.count(), "model": model}


def reembed_estimate(spec) -> dict:
    """What re-embedding the active project with `spec` would cost: passages, how many are
    already in the shared cache for that model, and the tokens to embed (≈ 4 characters each)."""
    cfg = load_config()
    papers = load_papers(cfg)
    chunks = load_chunks(cfg)
    texts = [embed_text(c, papers[c["paper_id"]]) for c in chunks]
    keys = [embcache.cache_key(spec.identity, t) for t in texts]
    cached = embcache.known(keys)
    missing_chars = sum(len(t) for t, k in zip(texts, keys) if k not in cached)
    return {"passages": len(chunks), "cached": len(cached), "tokens": missing_chars // 4}


if __name__ == "__main__":
    main()
