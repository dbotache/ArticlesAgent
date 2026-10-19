"""One background job at a time (building/updating the database).

Building can take an hour; running it in a thread keeps it alive while the user keeps using
the UI (a Streamlit rerun would otherwise interrupt it). The UI polls `current()`.
"""

import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable

STAGES = {"scan": "Matching PDFs to references", "register": "Reading PDF metadata",
          "parse": "Parsing PDFs", "embed": "Embedding passages"}


@dataclass
class Job:
    label: str
    status: str = "running"  # running | done | error
    stage: str = ""
    done: int = 0
    total: int = 0
    log: list[str] = field(default_factory=list)
    error: str = ""
    seen: bool = False  # the UI has reacted to the finished job

    def progress(self, stage: str, done: int, total: int) -> None:
        self.stage, self.done, self.total = stage, done, total

    @property
    def stage_label(self) -> str:
        return STAGES.get(self.stage, self.stage or "Starting")

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


_current: Job | None = None
_lock = threading.Lock()


def current() -> Job | None:
    return _current


def start(label: str, fn: Callable[[Job], None]) -> bool:
    """Run fn(job) in a thread. False if another job is still running."""
    global _current
    with _lock:
        if _current and _current.status == "running":
            return False
        job = _current = Job(label)

    def run():
        try:
            fn(job)
            job.status = "done"
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.append(traceback.format_exc(limit=3))
            job.status = "error"

    threading.Thread(target=run, daemon=True, name="articles-agent-job").start()
    return True
