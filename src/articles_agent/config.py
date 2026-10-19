"""Paths and settings.

    <APP_HOME>/                 ARTICLES_AGENT_HOME, default: the code folder
    ├── .env                    Azure settings, shared by all projects
    ├── projects/<slug>/        one folder per project: project.json, library/, data/
    └── shared/                 caches shared by all projects (parsed PDFs, PDF metadata,
                                embeddings) – a paper used in two projects is processed once

The active project is per thread/session (`use_project`, `set_project`); processes started by
the app get it through ARTICLES_AGENT_PROJECT.
"""

import contextvars
import os
from contextlib import contextmanager
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
APP_HOME = Path(os.environ["ARTICLES_AGENT_HOME"]).resolve() if os.environ.get("ARTICLES_AGENT_HOME") \
    else CODE_ROOT

_project: contextvars.ContextVar[str | None] = contextvars.ContextVar("articles_agent_project", default=None)


class NoProjectError(RuntimeError):
    pass


def projects_dir() -> Path:
    return APP_HOME / "projects"


def shared_dir() -> Path:
    return APP_HOME / "shared"


def current_project() -> str:
    """Slug of the active project: set for this thread, else $ARTICLES_AGENT_PROJECT, else the
    first project (by creation)."""
    slug = _project.get() or os.environ.get("ARTICLES_AGENT_PROJECT")
    if slug:
        return slug
    from .projects import list_projects  # projects imports this module
    projects = list_projects()
    if not projects:
        raise NoProjectError("No project yet. Create one in the app or with "
                             "`python -m articles_agent.projects create <name>`.")
    return projects[0].slug


def set_project(slug: str | None) -> None:
    _project.set(slug)


@contextmanager
def use_project(slug: str):
    token = _project.set(slug)
    try:
        yield
    finally:
        _project.reset(token)


def add_project_option(parser) -> None:
    """`--project <slug>` for the command-line tools (default: $ARTICLES_AGENT_PROJECT / first)."""
    parser.add_argument("--project", help="project slug (see `python -m articles_agent.projects list`)")


def apply_project_option(args) -> None:
    if getattr(args, "project", None):
        from .projects import get_project
        get_project(args.project)  # fail early on a typo
        set_project(args.project)


def project_dir(slug: str | None = None) -> Path:
    return projects_dir() / (slug or current_project())


def settings() -> dict:
    """config.yaml: APP_HOME's own, else the shipped one (same for all projects)."""
    path = APP_HOME / "config.yaml"
    if not path.exists():
        path = CODE_ROOT / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_config(path: Path | None = None) -> dict:
    """settings() + the active project's paths (project_dir, data_dir)."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path else settings()
    cfg["project_dir"] = project_dir()
    cfg["data_dir"] = cfg["project_dir"] / cfg["data_dir"]
    return cfg
