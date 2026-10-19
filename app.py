"""Chat UI for the library.

    .venv/bin/streamlit run app.py
"""

import subprocess
from pathlib import Path

import streamlit as st

from articles_agent.graph import ask_stream, get_retriever
from articles_agent.library import collect_pdfs, register_pdfs, save_upload, update_database
from articles_agent.results import Result
from articles_agent.store import ZOTERO_KEY

st.set_page_config(page_title="ArticlesAgent", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the library index…")
def retriever():
    return get_retriever()


def md_safe(text: str) -> str:
    """Streamlit renders $…$ as LaTeX; formulas in PDF text would break the layout."""
    return text.replace("$", r"\$")


def zotero_url(pdf_path: str, page: int) -> str:
    # the Zotero storage folder name is the attachment key
    return f"zotero://open-pdf/library/items/{Path(pdf_path).parent.name}?page={page}"


def run_update(uploads, folder: str, use_crossref: bool):
    """Register uploaded files / a folder's PDFs, then parse, chunk and embed what is new."""
    log = []
    with st.status("Updating the library…", expanded=True) as status:
        try:
            paths = [save_upload(f.name, f.getvalue()) for f in uploads]
            if folder.strip():
                found = collect_pdfs(folder.strip())
                status.write(f"{len(found)} PDFs found in {folder.strip()}")
                paths += found
            bar = st.progress(0.0)
            labels = {"register": "Reading metadata", "parse": "Parsing PDFs",
                      "embed": "Embedding passages"}

            def progress(stage, done, total):
                bar.progress(done / max(total, 1), text=f"{labels[stage]}: {done}/{total}")

            added = []
            if paths:
                results = register_pdfs(paths, use_crossref=use_crossref, progress=progress)
                icons = {"added": "✅", "skipped": "⏭️", "error": "❌"}
                log += [f"{icons[x['status']]} **{x['file']}**: {x['detail']}" for x in results]
                added = [x for x in results if x["status"] == "added"]
            if added or not paths:  # no paths = Zotero re-sync
                status.write("Parsing new PDFs and embedding new passages…")
                report = update_database(progress=progress)
                per_paper = report["ingest"]["chunks_per_paper"]
                log += [f"⚠️ **{x['file']}**: no text extracted (scanned PDF?)"
                        for x in added if not per_paper.get(x["paper_id"])]
                log.append(f"Indexed {report['index']['embedded']:,} new passages; "
                           f"the index now holds {report['index']['total']:,}.")
            status.update(label="Library updated", state="complete")
        except Exception as exc:
            status.update(label="Update failed", state="error")
            log.append(f"❌ {type(exc).__name__}: {exc}")
    retriever.clear()
    get_retriever.cache_clear()
    st.session_state.update_log = log
    st.session_state.upload_key += 1  # empties the file uploader
    st.rerun()


r = retriever()
papers = r.papers
chunk_by_id = {c["chunk_id"]: c for c in r.chunks}
st.session_state.setdefault("history", [])
history: list[dict] = st.session_state.history

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("📚 ArticlesAgent")
    indexed = sorted({c["paper_id"] for c in r.chunks}, key=lambda pid: papers[pid]["short_cite"])
    st.caption(f"{len(indexed)} papers · {len(r.chunks):,} searchable passages")

    st.subheader("Filters")
    years = sorted({int(p["year"]) for p in papers.values() if str(p.get("year", "")).isdigit()})
    year_range = st.slider("Publication year", years[0], years[-1], (years[0], years[-1]))
    year_filtered = year_range != (years[0], years[-1])
    if year_filtered:
        st.caption("Papers without a year are excluded while this filter is set.")
    selected = st.multiselect(
        "Restrict to papers", indexed, placeholder="All papers",
        format_func=lambda pid: f"{papers[pid]['short_cite']} — {papers[pid]['title'][:70]}")

    st.divider()
    st.session_state.setdefault("upload_key", 0)
    with st.expander("➕ Add papers", expanded=bool(st.session_state.get("update_log"))):
        uploads = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True,
                                   key=f"upload_{st.session_state.upload_key}")
        folder = st.text_input("…and/or a folder (searched recursively)",
                               placeholder="/home/diego/Papers/new")
        crossref = st.checkbox("Look up citation data via DOI (Crossref)", value=True,
                               help="Only the DOI printed in the PDF is sent to api.crossref.org.")
        add_clicked = st.button("Add to library", type="primary", use_container_width=True,
                                disabled=not (uploads or folder.strip()))
        sync_clicked = st.button(
            "🔄 Re-sync Zotero library", use_container_width=True,
            help="Re-read My_Library.bib and index PDFs added in Zotero since the last run. "
                 "Export the .bib from Zotero first.")
        st.caption("Parsing takes a few seconds per PDF, embedding about 1 minute per "
                   "10 papers. Keep this tab open until it is done.")
        if add_clicked:
            run_update(uploads, folder, crossref)
        elif sync_clicked:
            run_update([], "", crossref)
        if log := st.session_state.get("update_log"):
            st.markdown("\n\n".join(log))
            if st.button("Dismiss", key="dismiss_log"):
                del st.session_state.update_log
                st.rerun()

    st.divider()
    if history:
        session_md = "\n\n---\n\n".join(t["result"].to_markdown() for t in history)
        st.download_button("⬇ Download session (Markdown)", session_md,
                           file_name="articles_agent_session.md", mime="text/markdown",
                           use_container_width=True)
        if st.button("🗑 Clear chat", use_container_width=True):
            history.clear()
            st.rerun()
    st.caption("Each question is answered independently from the library; "
               "follow-up questions don't see earlier answers.")

# ------------------------------------------------------------------ rendering


def render_result(res: Result, idx: int):
    st.markdown(md_safe(res.summary))

    if res.findings:
        st.markdown("#### Findings")
        for i, f in enumerate(res.findings, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {md_safe(f.claim)}**")
                st.markdown(f"> “{md_safe(f.quote)}”")
                note = "" if f.exact else " · quote aligned to the source text"
                st.caption(f"{f.citation} · section *{md_safe(f.section)}*{note}")
                chunk = chunk_by_id.get(f.chunk_id)
                if chunk:
                    left, right = st.columns([1, 4])
                    if ZOTERO_KEY.match(Path(chunk["pdf_path"]).parent.name):
                        left.link_button("Open in Zotero", zotero_url(chunk["pdf_path"], f.page))
                    elif left.button("Open PDF", key=f"open_{idx}_{i}",
                                     help=f"Opens the file in your PDF viewer (go to page {f.page})"):
                        subprocess.Popen(["xdg-open", chunk["pdf_path"]])  # app runs locally
                    with right.expander("Show surrounding passage"):
                        st.markdown(md_safe(chunk["text"]))

    if res.dropped_quotes:
        st.caption(f"{res.dropped_quotes} finding(s) omitted: their quote was not found in the source.")
    if res.references:
        st.markdown("#### References")
        st.markdown("\n".join(f"- {md_safe(ref)}" for ref in res.references))
    with st.expander("Search details"):
        st.markdown("\n".join(f"- {md_safe(q)}" for q in res.queries))
    st.download_button("⬇ Download answer (Markdown)", res.to_markdown(),
                       file_name=f"answer_{idx + 1}.md", mime="text/markdown", key=f"dl_{idx}")


# ------------------------------------------------------------------ chat

if not history:
    st.markdown("### Ask a question about your library")
    st.markdown("English or German. Every finding comes with a verbatim quote, the PDF page "
                "and a reference list.")

for i, turn in enumerate(history):
    with st.chat_message("user"):
        st.markdown(md_safe(turn["question"]))
    with st.chat_message("assistant"):
        render_result(turn["result"], i)

if question := st.chat_input("e.g. How is deep learning used to accelerate topology optimization?"):
    with st.chat_message("user"):
        st.markdown(md_safe(question))
    with st.chat_message("assistant"):
        result = None
        with st.status("Searching the library…", expanded=False) as status:
            try:
                for node, update in ask_stream(
                        question, paper_ids=selected or None,
                        year_from=year_range[0] if year_filtered else None,
                        year_to=year_range[1] if year_filtered else None):
                    if node == "rewrite":
                        status.update(label="Searching with rewritten queries…")
                        status.write("**Queries:** " + " · ".join(update["queries"][1:]))
                    elif node == "retrieve":
                        status.write(f"{len(update['candidates'])} candidate passages found")
                    elif node == "grade":
                        status.update(label="Writing the answer…")
                        status.write(f"{len(update['sources'])} relevant passages selected")
                    elif node == "generate":
                        status.update(label="Verifying quotes…")
                    elif node == "cite":
                        result = update["result"]
                status.update(label=f"Done: {len(result.findings)} cited findings", state="complete")
            except Exception as exc:  # show Azure / index errors in the UI instead of a traceback
                status.update(label="Failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
        if result:
            history.append({"question": question, "result": result})
            st.rerun()  # redraw from history so the sidebar offers the session download
