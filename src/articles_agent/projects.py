"""Projects: independent workspaces, each with its own libraries, index and chat.

    python -m articles_agent.projects list
    python -m articles_agent.projects create "Motor design" --description "PhD chapter 3"
    python -m articles_agent.projects rename motor-design "Motor design (final)"
    python -m articles_agent.projects delete battery-study
"""

import argparse
import json
import os
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import config

LEGACY_NAME = "My library"


@dataclass
class Project:
    slug: str
    name: str
    description: str = ""
    created: str = ""

    @property
    def dir(self) -> Path:
        return config.projects_dir() / self.slug


def _slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:40] or "project"


FIELDS = {"name", "description", "created"}


def _meta(slug: str) -> dict:
    meta = config.projects_dir() / slug / "project.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}


def _read(folder: Path) -> Project | None:
    if not (folder / "project.json").exists():
        return None
    data = _meta(folder.name)
    return Project(slug=folder.name, **{k: v for k, v in data.items() if k in FIELDS})


def _write(project: Project, **extra) -> None:
    """Save the project's fields, keeping other keys (e.g. the embedding pin)."""
    project.dir.mkdir(parents=True, exist_ok=True)
    data = _meta(project.slug) | {k: v for k, v in asdict(project).items() if k != "slug"} | extra
    tmp = project.dir / "project.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(project.dir / "project.json")


# ------------------------------------------------------------ embedding model (pinned per project)

def embedding_pin(slug: str | None = None) -> dict | None:
    """{provider, model, base_url, identity, dimensions, pinned} or None (not built yet)."""
    slug = slug or config.current_project()
    if pin := _meta(slug).get("embedding"):
        return pin
    return _pin_from_index(slug)


def embedding_spec(slug: str | None = None):
    """The embedding model of a project: the one its database was built with, or, before the
    first build, the default for new projects."""
    from .llm import EmbeddingSpec, default_embedding_spec
    pin = embedding_pin(slug)
    if not pin:
        return default_embedding_spec()
    return EmbeddingSpec(pin["provider"], pin["model"], pin.get("base_url", ""))


def pin_embedding(slug: str, spec, dimensions: int | None = None) -> None:
    pin = {**spec.to_dict(), "identity": spec.identity, "dimensions": dimensions,
           "pinned": datetime.now().isoformat(timespec="seconds")}
    _write(get_project(slug), embedding=pin)


def _pin_from_index(slug: str) -> dict | None:
    """Projects built before pins existed: take the model from the index's own record."""
    chroma = config.projects_dir() / slug / "data" / "chroma"
    if not chroma.is_dir():
        return None
    import chromadb
    from .llm import EmbeddingSpec
    try:
        col = chromadb.PersistentClient(path=str(chroma)).get_collection("chunks")
    except Exception:
        return None
    identity = (col.metadata or {}).get("embedding_model")
    if not identity:
        return None
    provider, model = identity.split(":", 1) if ":" in identity else ("azure", identity)
    base_url = os.environ.get("COMPATIBLE_BASE_URL", "") if provider == "compatible" else ""
    sample = col.get(limit=1, include=["embeddings"])["embeddings"]
    dims = len(sample[0]) if len(sample) else None
    pin_embedding(slug, EmbeddingSpec(provider, model, base_url), dims)
    return _meta(slug)["embedding"]


def list_projects() -> list[Project]:
    migrate_legacy()
    root = config.projects_dir()
    found = [p for d in sorted(root.iterdir()) if d.is_dir() and (p := _read(d))] if root.is_dir() else []
    return sorted(found, key=lambda p: (p.created, p.slug))


def get_project(slug: str) -> Project:
    if not (p := _read(config.projects_dir() / slug)):
        raise KeyError(f"No project '{slug}'")
    return p


def create_project(name: str, description: str = "") -> Project:
    name = name.strip()
    if not name:
        raise ValueError("A project needs a name.")
    if any(p.name.lower() == name.lower() for p in list_projects()):
        raise ValueError(f"A project named '{name}' already exists.")
    base = slug = _slugify(name)
    n = 1
    while (config.projects_dir() / slug).exists():
        n += 1
        slug = f"{base}-{n}"
    project = Project(slug=slug, name=name, description=description.strip(),
                      created=datetime.now().isoformat(timespec="seconds"))
    _write(project)
    return project


def update_project(slug: str, name: str | None = None, description: str | None = None) -> Project:
    project = get_project(slug)
    if name is not None and name.strip():
        if any(p.name.lower() == name.strip().lower() and p.slug != slug for p in list_projects()):
            raise ValueError(f"A project named '{name}' already exists.")
        project.name = name.strip()
    if description is not None:
        project.description = description.strip()
    _write(project)
    return project


def delete_project(slug: str) -> None:
    """Deletes the project's libraries config, index and imported uploads. Your PDFs and
    reference-manager exports elsewhere are not touched; shared caches stay."""
    folder = get_project(slug).dir
    shutil.rmtree(folder)


# ------------------------------------------------------------ migration of the single-library layout

def migrate_legacy() -> Project | None:
    """Before projects existed, library/ and data/ lived directly in APP_HOME. Move them into a
    first project, move the caches to shared/, and seed the shared embedding cache from the
    existing index, so nothing has to be parsed or embedded again."""
    home = config.APP_HOME
    old_lib, old_data = home / "library", home / "data"
    if not ((old_lib / "sources.json").exists() or (old_data / "chunks.jsonl").exists()):
        return None
    if config.projects_dir().is_dir() and any(config.projects_dir().iterdir()):
        return None
    project = Project(slug=_slugify(LEGACY_NAME), name=LEGACY_NAME,
                      description="Migrated from the single-library setup",
                      created=datetime.now().isoformat(timespec="seconds"))
    project.dir.mkdir(parents=True, exist_ok=True)
    shared = config.shared_dir()
    shared.mkdir(parents=True, exist_ok=True)
    if (old_data / "parsed").is_dir():
        shutil.move(str(old_data / "parsed"), str(shared / "parsed"))
    if (old_lib / "cache").is_dir():
        shutil.move(str(old_lib / "cache"), str(shared / "pdfmeta"))
    for folder in (old_lib, old_data):
        if folder.is_dir():
            shutil.move(str(folder), str(project.dir / folder.name))
    _write(project)
    try:
        from .embcache import seed_from_index
        with config.use_project(project.slug):
            seed_from_index()
    except Exception:  # the cache is an optimisation; migration must not fail on it
        pass
    return project


# ------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("create")
    p.add_argument("name")
    p.add_argument("--description", default="")
    p = sub.add_parser("rename")
    p.add_argument("slug")
    p.add_argument("name")
    p = sub.add_parser("delete")
    p.add_argument("slug")
    args = ap.parse_args()

    if args.cmd == "list":
        for p in list_projects():
            print(f"{p.slug:24} {p.name}" + (f" — {p.description}" if p.description else ""))
    elif args.cmd == "create":
        print(f"created project '{create_project(args.name, args.description).slug}'")
    elif args.cmd == "rename":
        update_project(args.slug, name=args.name)
    elif args.cmd == "delete":
        delete_project(args.slug)


if __name__ == "__main__":
    main()
