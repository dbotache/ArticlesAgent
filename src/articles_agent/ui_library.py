"""Streamlit widgets for setting up and maintaining the library (used by app.py)."""

import os
from pathlib import Path

import streamlit as st

from . import jobs
from .config import load_config
from .library import collect_pdfs, register_pdfs, save_upload, update_database_isolated
from .llm import SETTINGS, check_connection, embedding_deployment, missing_settings, save_settings
from .sources import (MANAGERS, Source, add_source, keyword_counts, load_sources, read_source,
                      remove_source, save_uploaded_export, set_keywords, validate)
from .store import get_collection
from .graph import get_retriever
from .projects import Project, create_project, delete_project, update_project

ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"

MANAGER_HELP = {
    "zotero": "In Zotero: **File → Export Library…** → format **BibLaTeX** (or *Better BibLaTeX*), "
              "*Export Files* unchecked. PDF folder: Zotero's `storage` folder "
              "(Zotero → Settings → Advanced → Files and Folders → *Data Directory* + `/storage`).",
    "jabref": "Use the JabRef library file (`.bib`) itself. PDF links are read from its `file` field; "
              "relative links are resolved against the library's *General file directory* "
              "(Library → Library properties). Give the PDF folder if links are missing or broken.",
    "citavi": "In Citavi: **File → Export → Export references…** → **BibTeX** (or RIS). Citavi doesn't "
              "export PDF links, so also give the project's **Citavi Attachments** folder; PDFs are "
              "matched to references by DOI and title.",
    "mendeley": "Mendeley Desktop: **File → Export…** → *BibTeX (.bib)*. Mendeley Reference Manager: "
                "select all references → **Export** → *BibTeX*. Give the folder with your PDFs "
                "(Mendeley Desktop: Tools → Options → File Organizer) if the export has no file links.",
    "endnote": "In EndNote: **File → Export…**, output style **RefMan (RIS) Export**, save as `.ris` or "
               "`.txt`. PDF folder: `<Library>.Data/PDF`.",
    "other": "Any BibTeX/BibLaTeX (`.bib`) or RIS (`.ris`) export, plus the folder with the PDFs.",
}


# ------------------------------------------------------------ state

def index_size() -> int:
    """Number of indexed passages (0 if there is no usable index yet)."""
    if missing_settings() or not (load_config()["data_dir"] / "chunks.jsonl").exists():
        return 0
    try:
        return get_collection(load_config(), embedding_deployment()).count()
    except Exception:
        return 0


def _summary(report: dict) -> list[str]:
    ing, idx = report["ingest"], report["index"]
    lines = [f"{ing['pdfs']:,} PDFs in the library, {ing['parsed']:,} parsed now.",
             f"{idx['added']:,} passages added ({idx['reused']:,} from the shared cache, "
             f"{idx['embedded']:,} embedded now); the index holds {idx['total']:,}."]
    if ing["parse_errors"]:
        lines.append(f"⚠️ {len(ing['parse_errors'])} PDFs could not be read (see data/ingest_report.json).")
    return lines


def start_update(label: str, before=None) -> None:
    """Run `before(job)` (optional) and then the incremental database update in the background."""
    def work(job: jobs.Job):
        if before and before(job) is False:
            return
        job.log += _summary(update_database_isolated(progress=job.progress))

    if not jobs.start(label, work):
        st.warning("Another update is still running.")
    st.rerun()


# ------------------------------------------------------------ job progress

def _job_body(on_finished):
    job = jobs.current()
    if not job:
        return
    if job.status == "running":
        st.info(f"**{job.label}**")
        st.progress(job.fraction, text=f"{job.stage_label}: {job.done:,}/{job.total:,}"
                    if job.total else f"{job.stage_label}…")
        st.caption("Runs in the background. You can keep using the app, but keep it running.")
        return
    if not job.seen:
        job.seen = True
        on_finished()
        st.rerun(scope="app")
    (st.success if job.status == "done" else st.error)(
        f"**{job.label}**: " + ("finished" if job.status == "done" else f"failed — {job.error}"))
    if job.log:
        st.markdown("\n\n".join(line for line in job.log if not line.startswith("Traceback")))


_job_live = st.fragment(run_every=2)(_job_body)


def render_job(on_finished) -> bool:
    """Progress of the background job; True while one is running."""
    job = jobs.current()
    if job and job.status == "running":
        _job_live(on_finished)
        return True
    _job_body(on_finished)
    return False


# ------------------------------------------------------------ Azure connection

def render_connection():
    st.caption("Stored in the project's `.env` file on this computer.")
    # cleared after submit, so the key doesn't stay in the browser's form field
    with st.form("azure_settings", clear_on_submit=True):
        values = {}
        for key, label in SETTINGS.items():
            current = os.environ.get(key, "")
            if key == "AZURE_OPENAI_API_KEY":
                values[key] = st.text_input(label, type="password",
                                            placeholder="•••• saved — leave empty to keep" if current else "")
            else:
                values[key] = st.text_input(label, value=current)
        submitted = st.form_submit_button("Save and test connection", type="primary")
    if submitted:
        values = {k: v for k, v in values.items() if v.strip() or k != "AZURE_OPENAI_API_KEY"}
        if missing := [SETTINGS[k] for k in SETTINGS if k in values and not values[k].strip()
                       and k != "AZURE_OPENAI_FAST_DEPLOYMENT"]:
            st.error("Please fill in: " + ", ".join(missing))
            return
        save_settings(values)
        with st.spinner("Testing the deployments…"):
            st.session_state.connection_checks = check_connection()
    for label, ok, msg in st.session_state.get("connection_checks", []):
        (st.success if ok else st.error)(f"{label}: {msg}")


# ------------------------------------------------------------ library sources

def _preview(src: Source):
    rep = read_source(src, scan_folder=False)
    kept = rep.references - rep.filtered_out
    st.markdown(f"- **{rep.references:,}** references in the export\n"
                + (f"- **{kept:,}** of them carry one of the tags {', '.join(src.keywords)}"
                   " — only these go into the project\n" if src.keywords else "")
                + f"- **{rep.linked:,}** of {'these' if src.keywords else 'them'} with a PDF found "
                  "through the export's file links\n"
                + (f"- **{rep.pdfs_in_folder:,}** PDFs in the folder, **{rep.unlinked_pdfs:,}** not linked "
                   "— they will be matched to references by DOI/title, or added as papers of their own\n"
                   if src.pdf_dir else "")
                + (f"- ⚠️ {rep.broken_links:,} file links point to files that don't exist "
                   "— give the PDF folder so they can be found by file name\n" if rep.broken_links else ""))
    if not rep.references:
        st.warning("No references found. Is this a BibTeX or RIS export?")


def _keyword_picker(reference_file: str, key: str, default: list[str] | None = None) -> list[str]:
    """Multiselect of the export's tags/keywords (with counts); empty = whole library."""
    try:
        counts = keyword_counts(reference_file)
    except Exception:
        return []
    if not counts:
        st.caption("This export has no keywords/tags, so the whole library is used.")
        return []
    options = list(dict.fromkeys(list(default or []) + list(counts)))
    return st.multiselect(
        "Only references with one of these keywords/tags (optional)", options, default=default or [],
        key=key, format_func=lambda kw: f"{kw} ({counts.get(kw, 0)})",
        placeholder="All references", help="Zotero tags, Citavi keywords, JabRef groups, Mendeley tags, "
                                           "RIS KW. Use this for a project-specific part of a big library.")


def render_add_source(key: str, first: bool = False):
    """Form for adding a reference-manager library; builds/updates the database on submit."""
    manager = st.selectbox("Reference manager", list(MANAGERS), format_func=MANAGERS.get, key=f"{key}_mgr")
    st.info(MANAGER_HELP[manager], icon="ℹ️")
    ref_path = st.text_input("Export file (.bib / .ris): full path", key=f"{key}_ref",
                             placeholder="/home/you/Documents/My Library.bib")
    upload = st.file_uploader("…or upload the export file", type=["bib", "ris", "txt"], key=f"{key}_up")
    default_dir = str(ZOTERO_STORAGE) if manager == "zotero" and ZOTERO_STORAGE.is_dir() else ""
    pdf_dir = st.text_input("PDF folder (searched recursively)" + ("" if manager == "citavi" else ", optional"),
                            value=default_dir, key=f"{key}_pdf_{manager}")
    crossref = st.checkbox("Look up citation data of unmatched PDFs via DOI (Crossref)", value=True,
                           key=f"{key}_cr", help="Only the DOI printed in a PDF is sent to api.crossref.org.")

    if upload is not None:
        ref_path = str(save_uploaded_export(upload.name, upload.getvalue()))
    problems = validate(ref_path, pdf_dir) if ref_path.strip() else ["Give the export file."]
    keywords = [] if problems else _keyword_picker(ref_path.strip(), key=f"{key}_kw")
    left, right = st.columns(2)
    check = left.button("Check", key=f"{key}_check", disabled=bool(problems), use_container_width=True)
    go = right.button("Build database" if first else "Add library", type="primary", key=f"{key}_go",
                      disabled=bool(problems) or missing_settings() != [], use_container_width=True)
    if ref_path.strip() and problems:
        st.error(" ".join(problems))
    if check:
        _preview(Source(id="preview", manager=manager, reference_file=ref_path.strip(),
                        pdf_dir=pdf_dir.strip(), use_crossref=crossref, keywords=keywords))
    if go:
        try:
            src = add_source(manager, ref_path.strip(), pdf_dir.strip(), crossref, keywords)
        except ValueError as exc:
            st.error(str(exc))
            return
        start_update(f"Adding {src.label}")


def render_sources():
    sources = load_sources()
    if not sources:
        st.caption("No library yet.")
    for src in sources:
        with st.container(border=True):
            st.markdown(f"**{src.label}**")
            st.caption(f"Export: `{src.reference_file}`" + (f"  \nPDFs: `{src.pdf_dir}`" if src.pdf_dir else "")
                       + f"  \nFilter: {src.filter_label}")
            if not Path(src.reference_file).exists():
                st.error("Export file not found.")
                continue
            new_kw = _keyword_picker(src.reference_file, key=f"kw_{src.id}", default=src.keywords)
            left, right = st.columns(2)
            if left.button("Apply filter", key=f"apply_{src.id}", disabled=new_kw == src.keywords,
                           use_container_width=True):
                set_keywords(src.id, new_kw)
                start_update(f"Changing the filter of {src.label}")
            if right.button("Remove", key=f"rm_{src.id}", use_container_width=True,
                            help="Removes this library and its papers from this project's database"):
                remove_source(src.id)
                start_update(f"Removing {src.label}")
    if sources and st.button("🔄 Re-sync all libraries", use_container_width=True,
                             help="Re-read the exports and index new or changed PDFs. "
                                  "Export the libraries again first."):
        start_update("Re-syncing libraries")


def render_add_papers():
    """Single PDF files and/or a folder of PDFs that belong to no library."""
    st.session_state.setdefault("upload_key", 0)
    uploads = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True,
                               key=f"upload_{st.session_state.upload_key}")
    folder = st.text_input("…and/or a folder (searched recursively)", placeholder="/home/you/Papers/new")
    crossref = st.checkbox("Look up citation data via DOI (Crossref)", value=True, key="papers_cr",
                           help="Only the DOI printed in the PDF is sent to api.crossref.org.")
    if st.button("Add to library", type="primary", use_container_width=True,
                 disabled=not (uploads or folder.strip())):
        try:
            paths = [save_upload(f.name, f.getvalue()) for f in uploads]
            paths += collect_pdfs(folder.strip()) if folder.strip() else []
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        st.session_state.upload_key += 1  # empties the uploader

        def register(job: jobs.Job):
            results = register_pdfs(paths, use_crossref=crossref, progress=job.progress)
            icons = {"added": "✅", "skipped": "⏭️", "error": "❌"}
            job.log += [f"{icons[x['status']]} **{x['file']}**: {x['detail']}" for x in results]
            return any(x["status"] == "added" for x in results)  # False: nothing to index

        start_update(f"Adding {len(paths)} PDF(s)", before=register)


# ------------------------------------------------------------ projects

def render_new_project(key: str, first: bool = False) -> None:
    with st.form(f"{key}_new_project", clear_on_submit=True):
        name = st.text_input("Project name", placeholder="e.g. Motor design", key=f"{key}_name")
        description = st.text_area("Description (optional)", key=f"{key}_desc", height=68,
                                   placeholder="What is this project about?")
        submitted = st.form_submit_button("Create project", type="primary" if first else "secondary")
    if submitted:
        try:
            project = create_project(name, description)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.project = project.slug
        st.rerun()


def render_project_switcher(projects: list[Project]) -> None:
    """Selectbox of all projects (sidebar) plus a form for a new one."""
    slugs = [p.slug for p in projects]
    names = {p.slug: p.name for p in projects}
    # no widget key: the selection follows st.session_state.project (e.g. after creating a project)
    chosen = st.selectbox("Project", slugs, index=slugs.index(st.session_state.project),
                          format_func=names.get)
    if chosen != st.session_state.project:
        st.session_state.project = chosen
        st.rerun()
    if desc := next(p.description for p in projects if p.slug == chosen):
        st.caption(desc)
    with st.expander("➕ New project"):
        render_new_project("sidebar")


def render_project_settings(project: Project) -> None:
    with st.form(f"settings_{project.slug}"):
        name = st.text_input("Name", value=project.name)
        description = st.text_area("Description", value=project.description, height=68)
        if st.form_submit_button("Save"):
            try:
                update_project(project.slug, name=name, description=description)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    st.caption(f"Folder: `{project.dir}`")
    sure = st.checkbox("Delete this project's libraries settings, index and uploads", key=f"del_ok_{project.slug}",
                       help="Your PDFs and reference-manager exports are not touched; shared caches stay, "
                            "so re-creating the project later is fast.")
    if st.button("🗑 Delete project", disabled=not sure, key=f"del_{project.slug}"):
        job = jobs.current()
        if job and job.status == "running" and job.project == project.slug:
            st.error("An update of this project is still running.")
            return
        delete_project(project.slug)
        get_retriever.cache_clear()
        st.session_state.pop("project", None)
        st.rerun()
