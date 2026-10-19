"""Checks for running the app as a hosted web application (e.g. on your own website).

Part 1 must pass today. Part 2 lists what is still missing for *public* hosting; those tests are
marked xfail(strict=True): they are expected to fail now, and turn into an error ("XPASS(strict)")
as soon as a gap is closed, so the marker gets removed and the check is enforced from then on.
"""

import ast
import os
import re
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request

import pytest
from streamlit.testing.v1 import AppTest

from conftest import CODE_ROOT, FAKE_ENV

APP = str(CODE_ROOT / "app.py")
PYTHON = sys.executable
SRC_FILES = sorted((CODE_ROOT / "src" / "articles_agent").glob("*.py")) + [CODE_ROOT / "app.py"]


def child_env(home, **extra) -> dict:
    """Environment of a fresh process: no Azure settings unless given, data in `home`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("AZURE_OPENAI_")}
    return env | {"ARTICLES_AGENT_HOME": str(home)} | extra


def run_python(code: str, env: dict) -> str:
    out = subprocess.run([PYTHON, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


# ================================================================== part 1: must pass


@pytest.fixture
def real_network(monkeypatch):
    monkeypatch.undo()  # lifts the suite-wide `offline` patch: this test talks to its own server


@pytest.mark.integration
@pytest.mark.usefixtures("real_network")
def test_server_starts_and_answers_health_checks(tmp_path):
    """What a reverse proxy / container orchestrator checks."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [PYTHON, "-m", "streamlit", "run", APP, "--server.headless", "true", "--server.port", str(port),
         "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"],
        env=child_env(tmp_path / "home", **FAKE_ENV), cwd=tmp_path,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
                    assert r.read().decode() == "ok"
                    break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("server did not become healthy within 30 s")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert r.status == 200 and b"<html" in r.read().lower()
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def test_data_home_is_separate_from_code(tmp_path):
    """On a server, .env/library/data live on a volume, the code in the image."""
    home = tmp_path / "volume"
    out = run_python("from articles_agent.config import PROJECT_ROOT, load_config, CODE_ROOT;"
                     "from articles_agent import sources, library, pdfmeta;"
                     "print(PROJECT_ROOT, load_config()['data_dir'], sources.SOURCES_FILE.parent,"
                     " library.UPLOADS.parent, pdfmeta.CACHE_DIR.parent, CODE_ROOT)", child_env(home))
    root, data, lib1, lib2, lib3, code = out.split()
    assert root == str(home) and data == str(home / "data")
    assert lib1 == lib2 == lib3 == str(home / "library")
    assert code == str(CODE_ROOT)  # config.yaml falls back to the shipped one


def test_configuration_through_environment_only(tmp_path):
    """Containers pass secrets as environment variables; no .env file needed, and the
    developer's .env next to the code must not be picked up."""
    home = tmp_path / "home"
    home.mkdir()
    code = "from articles_agent.llm import missing_settings; print(missing_settings())"
    assert run_python(code, child_env(home)) == str(["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                                                     "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                                                     "AZURE_OPENAI_CHAT_DEPLOYMENT"])
    assert run_python(code, child_env(home, **FAKE_ENV)) == "[]"


def all_text(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "info", "success", "warning", "error", "title", "text", "code"):
        parts += [str(e.value) for e in getattr(at, kind)]
    parts += [str(w.value) for w in at.text_input] + [str(w.placeholder) for w in at.text_input]
    return "\n".join(parts)


@pytest.mark.parametrize("page", ["setup", "chat"])
def test_api_key_is_never_sent_to_the_browser(page, request, workspace):
    if page == "chat":
        request.getfixturevalue("built")
    import streamlit as st
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    for exp in at.expander:  # open every section, including the Azure settings
        exp.proto.expanded = True
    assert not at.exception
    assert FAKE_ENV["AZURE_OPENAI_API_KEY"] not in all_text(at)


def test_dependencies_are_declared():
    """A clean image installs only pyproject dependencies; every import must be covered."""
    declared = {re.split(r"[<>=!\[ ]", d)[0].lower()
                for d in tomllib.loads((CODE_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]}
    dist = {"yaml": "pyyaml", "dotenv": "python-dotenv", "rank_bm25": "rank-bm25",
            "langchain_openai": "langchain-openai", "langchain_core": "langchain-core"}
    missing = set()
    for f in SRC_FILES:
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top == "articles_agent":
                    continue
                if dist.get(top, top).lower() not in declared:
                    missing.add(f"{top} ({f.name})")
    assert not missing, f"imported but not in pyproject.toml: {sorted(missing)}"


def test_no_personal_paths_in_code_or_config():
    offenders = [f"{f.name}:{i}" for f in SRC_FILES + [CODE_ROOT / "config.yaml"]
                 for i, line in enumerate(f.read_text().splitlines(), 1)
                 if re.search(r"/home/(?!u/|you/)\w+|[A-Z]:\\\\Users\\\\(?!u\\\\)", line)]
    assert not offenders


def test_secrets_and_data_are_git_ignored():
    ignored = set((CODE_ROOT / ".gitignore").read_text().split())
    assert {".env", "data/", "library/"} <= ignored


# ================================================================== part 2: gaps for public hosting

PUBLIC = "Not ready for public hosting yet: "


@pytest.fixture
def visitor_page(built):
    """The chat page as an anonymous visitor of the website would see it."""
    import streamlit as st
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


@pytest.mark.xfail(strict=True, reason=PUBLIC + "visitors can type server folder/file paths "
                                         "(library export, PDF folder) and so read files on the server")
def test_visitors_cannot_enter_server_paths(visitor_page):
    path_inputs = [w.label for w in visitor_page.text_input if re.search(r"folder|path|file", w.label, re.I)]
    assert path_inputs == []


@pytest.mark.xfail(strict=True, reason=PUBLIC + "visitors can change the Azure endpoint/key "
                                         "and manage libraries (needs an admin-only area)")
def test_visitors_cannot_change_settings_or_libraries(visitor_page):
    labels = [e.label for e in visitor_page.expander]
    assert not any(x in " ".join(labels) for x in ("Azure connection", "Libraries", "Add single papers"))


@pytest.mark.xfail(strict=True, reason=PUBLIC + "no login: anyone with the URL can use the "
                                         "app and spend the Azure quota")
def test_access_requires_login(visitor_page):
    assert len(visitor_page.chat_input) == 0  # a login form instead of the chat


@pytest.mark.xfail(strict=True, reason=PUBLIC + "'Open PDF' runs xdg-open on the server; on a "
                                         "website it should send the PDF to the browser instead")
def test_no_os_commands_from_the_ui():
    assert "xdg-open" not in (CODE_ROOT / "app.py").read_text()


@pytest.mark.xfail(strict=True, reason=PUBLIC + "no .streamlit/config.toml limiting upload size "
                                         "(Streamlit default: 200 MB per file)")
def test_upload_size_is_limited():
    cfg = CODE_ROOT / ".streamlit" / "config.toml"
    assert cfg.exists() and tomllib.loads(cfg.read_text()).get("server", {}).get("maxUploadSize", 200) <= 50
