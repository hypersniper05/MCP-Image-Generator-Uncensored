"""Background jobs, so long generations survive client timeouts and disconnects.

A tool call starts a job and waits for it for ``wait_seconds``. If the job is
still running when the wait ends, the call returns the job id and the client
fetches the result later with ``get_job``.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger("imagegen.jobs")


def release_memory() -> None:
    """Return freed heap memory to the OS. glibc keeps large freed blocks (an 8192 px upscale frees ~1 GB of
    image buffers), so without this the process grows to the sum of past jobs' peaks."""
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

Progress = Callable[[float, str], Awaitable[None]]


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"  # running | completed | failed | cancelled
    progress: float = 0.0  # 0..1
    message: str = "starting"
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None

    def summary(self) -> dict:
        return {"job_id": self.id, "kind": self.kind, "status": self.status,
                "progress_percent": round(self.progress * 100, 1), "message": self.message,
                "elapsed_s": round((self.finished or time.time()) - self.created, 1), "error": self.error}


class JobManager:
    def __init__(self, ttl_seconds: float = 24 * 3600, max_jobs: int = 500,
                 format_error: Callable[[BaseException], str] | None = None):
        self.jobs: dict[str, Job] = {}
        self.format_error = format_error or (lambda exc: str(exc) or exc.__class__.__name__)
        self.ttl = ttl_seconds
        self.max_jobs = max_jobs

    def _prune(self) -> None:
        now = time.time()
        for jid, job in list(self.jobs.items()):
            if job.finished and now - job.finished > self.ttl:
                del self.jobs[jid]
        if len(self.jobs) > self.max_jobs:
            finished = sorted((j for j in self.jobs.values() if j.finished), key=lambda j: j.finished)
            for job in finished[: len(self.jobs) - self.max_jobs]:
                del self.jobs[job.id]

    def running(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == "running")

    def submit(self, kind: str, factory: Callable[[Progress], Awaitable[Any]]) -> Job:
        self._prune()
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)

        async def progress(frac: float, message: str) -> None:
            # progress only moves forward, so clients never see it jump back
            job.progress = max(job.progress, min(1.0, float(frac)))
            job.message = message

        async def runner() -> None:
            try:
                job.result = await factory(progress)
                job.status = "completed"
                job.progress = 1.0
                job.message = "done"
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.error = "cancelled"
            except Exception as exc:  # noqa: BLE001 - reported to the client
                job.status = "failed"
                job.error = self.format_error(exc)
                log.warning("job %s (%s) failed: %s", job.id, job.kind, job.error[:500])
                if not isinstance(exc, (ValueError, RuntimeError)):
                    log.debug("job %s traceback", job.id, exc_info=True)
            finally:
                job.finished = time.time()
                job.done.set()
                await asyncio.to_thread(release_memory)

        job.task = asyncio.create_task(runner(), name=f"job-{job.id}")
        self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id.strip())

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job and job.status == "running" and job.task:
            job.task.cancel()
        return job

    async def wait(self, job: Job, seconds: float, on_progress: Progress | None = None) -> bool:
        """Wait up to ``seconds`` for the job; forward progress about once per second."""
        deadline = time.monotonic() + max(0.0, seconds)
        last = (-1.0, "")
        last_sent = 0.0
        while not job.done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(job.done.wait(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            if on_progress and ((job.progress, job.message) != last or now - last_sent > 10):
                last, last_sent = (job.progress, job.message), now
                await on_progress(job.progress, job.message)
        return True
