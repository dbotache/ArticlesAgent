"""Chat UI for the library.

    .venv/bin/streamlit run app.py --server.address localhost

Without a database yet, the app shows a setup page: Azure connection → reference library
(Zotero, JabRef, Citavi, Mendeley, EndNote, any BibTeX/RIS) → build the database.
"""

import subprocess
from pathlib import Path

import streamlit as st

from articles_agent.graph import ask_stream, get_retriever
from articles_agent.llm import missing_settings
from articles_agent.results import Result
from articles_agent.sources import load_sources
from articles_agent.store import ZOTERO_KEY
from articles_agent.ui_library import (index_size, render_add_papers, render_add_source,
                                       render_connection, render_job, render_sources,
                                       start_update)

st.set_page_config(page_title="ArticlesAgent", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the library index…")
def retriever():
    return get_retriever()


def reload_index():
    """After a database update: rebuild BM25 and reload chunks on the next access."""
    retriever.clear()
    get_retriever.cache_clear()


def md_safe(text: str) -> str:
    """Streamlit renders $…$ as LaTeX; formulas in PDF text would break the layout."""
    return text.replace("$", r"\$")


def zotero_url(pdf_path: str, page: int) -> str:
    # the Zotero storage folder name is the attachment key
    return f"zotero://open-pdf/library/items/{Path(pdf_path).parent.name}?page={page}"


# ------------------------------------------------------------------ setup (no database yet)


def setup_page():
    st.title("📚 Set up ArticlesAgent")
    st.markdown("ArticlesAgent answers questions from **your own library of papers**, with "
                "verbatim quotes, page numbers and a reference list. Setting it up takes three steps; "
                "building the database then runs in the background (roughly 1–2 minutes per 10 PDFs, "
                "depending on your Azure rate limit).")
    running = render_job(reload_index)
    missing = missing_settings()
    sources = load_sources()

    with st.expander(("✅ " if not missing else "") + "**1 · Azure OpenAI connection**", expanded=bool(missing)):
        render_connection()
    with st.expander(("✅ " if sources else "") + "**2 · Reference library**",
                     expanded=not missing and not sources):
        if missing:
            st.caption("Set up the Azure connection first.")
        elif sources:
            render_sources()
            st.markdown("**Add another library**")
            render_add_source("setup_more")
        else:
            render_add_source("setup", first=True)
    if sources and not missing and not running:
        st.markdown("**3 · Build the database**")
        if st.button("🚀 Build database", type="primary"):
            start_update("Building the database")


if index_size() == 0:
    setup_page()
    st.stop()

# ------------------------------------------------------------------ chat app

r = retriever()
papers = r.papers
chunk_by_id = {c["chunk_id"]: c for c in r.chunks}
st.session_state.setdefault("history", [])
history: list[dict] = st.session_state.history

with st.sidebar:
    st.title("📚 ArticlesAgent")
    indexed = sorted({c["paper_id"] for c in r.chunks}, key=lambda pid: papers[pid]["short_cite"])
    st.caption(f"{len(indexed)} papers · {len(r.chunks):,} searchable passages")
    render_job(reload_index)

    st.subheader("Filters")
    years = sorted({int(p["year"]) for p in papers.values() if str(p.get("year", "")).isdigit()})
    if len(years) > 1:
        year_range = st.slider("Publication year", years[0], years[-1], (years[0], years[-1]))
        year_filtered = year_range != (years[0], years[-1])
    else:
        year_range, year_filtered = (None, None), False
    if year_filtered:
        st.caption("Papers without a year are excluded while this filter is set.")
    selected = st.multiselect(
        "Restrict to papers", indexed, placeholder="All papers",
        format_func=lambda pid: f"{papers[pid]['short_cite']} — {papers[pid]['title'][:70]}")

    st.divider()
    with st.expander("📚 Libraries"):
        render_sources()
        st.markdown("**Add a library**")
        render_add_source("sidebar")
    with st.expander("➕ Add single papers"):
        render_add_papers()
    with st.expander("⚙️ Azure connection"):
        render_connection()

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
