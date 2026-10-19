"""Azure settings (.env) and the background job runner."""

import os
import stat
import threading
import time

from articles_agent import jobs, llm


def test_missing_settings(workspace, monkeypatch):
    assert llm.missing_settings() == []
    monkeypatch.delenv("AZURE_OPENAI_API_KEY")
    monkeypatch.delenv("AZURE_OPENAI_FAST_DEPLOYMENT")  # optional
    assert llm.missing_settings() == ["AZURE_OPENAI_API_KEY"]


def test_save_settings_keeps_other_lines_and_is_private(app_home, monkeypatch):
    env = app_home / ".env"  # shared by all projects
    env.write_text("# comment\nOTHER=1\nAZURE_OPENAI_ENDPOINT=https://old/\n")
    cleared = []
    for name in ("get_embeddings", "get_chat"):  # the fakes have no lru cache; record the clear
        fake = lambda *a, **k: None
        fake.cache_clear = lambda n=name: cleared.append(n)
        monkeypatch.setattr(llm, name, fake)
    llm.save_settings({"AZURE_OPENAI_ENDPOINT": " https://new.example/ ", "AZURE_OPENAI_API_KEY": "k"})
    assert env.read_text() == "# comment\nOTHER=1\nAZURE_OPENAI_ENDPOINT=https://new.example/\nAZURE_OPENAI_API_KEY=k\n"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://new.example/"
    assert cleared == ["get_embeddings", "get_chat"]  # new settings take effect immediately


def test_check_connection_reports_each_deployment(workspace):
    checks = llm.check_connection()
    assert [ok for _, ok, _ in checks] == [True, True, True]
    assert checks[0][2] == "64 dimensions"


def test_check_connection_reports_failures(workspace, monkeypatch):
    class Broken:
        model_name = "broken"
        def embed_query(self, _):
            raise RuntimeError("401 Unauthorized")
        def invoke(self, _):
            raise RuntimeError("DeploymentNotFound")
    monkeypatch.setattr(llm, "get_embeddings", lambda: Broken())
    monkeypatch.setattr(llm, "get_chat", lambda fast=False: Broken())
    checks = llm.check_connection()
    assert [ok for _, ok, _ in checks] == [False, False, False]
    assert "401" in checks[0][2]


def test_base_url_uses_v1_api(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
    assert llm._base_url() == "https://res.openai.azure.com/openai/v1/"


def wait(job, timeout=5):
    t = time.time()
    while job.status == "running" and time.time() - t < timeout:
        time.sleep(0.01)


def test_job_runs_in_background_and_reports_progress():
    gate = threading.Event()

    def work(job):
        job.progress("parse", 1, 4)
        gate.wait(2)
        job.log.append("done")

    assert jobs.start("Test", work)
    job = jobs.current()
    time.sleep(0.05)
    assert job.status == "running" and job.fraction == 0.25 and job.stage_label == "Parsing PDFs"
    assert not jobs.start("Second", lambda j: None)  # only one job at a time
    gate.set()
    wait(job)
    assert job.status == "done" and job.log == ["done"]
    assert jobs.start("Next", lambda j: None)  # free again
    wait(jobs.current())


def test_job_errors_are_captured():
    jobs.start("Failing", lambda j: 1 / 0)
    job = jobs.current()
    wait(job)
    assert job.status == "error" and job.error.startswith("ZeroDivisionError")
