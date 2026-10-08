"""Background jobs, so long generations survive client timeouts and disconnects.

A tool call starts a job and waits for it for a short, server-chosen time. If the
job is still running when the wait ends, the call returns the job id and the client
fetches the result later with ``get_job``. A repeated identical request (e.g. a
model retrying after its client timed out) attaches to the job that is already
running instead of rendering the same image again.
"""

from __future__ import annotations

import asyncio
import contextvars
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

# The job whose task is running: lets the GPU queue tell a job its place in line without passing it around.
CURRENT: contextvars.ContextVar["Job | None"] = contextvars.ContextVar("imagegen_current_job", default=None)

REUSE_MIN_AGE = 5.0      # identical calls closer together than this are parallel requests (e.g. 4 versions), not retries
REUSE_UNDELIVERED = 1800  # a finished job whose image never reached a client is reused for this long


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
    key: str | None = None      # fingerprint of the request (tool + arguments + caller), for reusing a running job
    label: str = ""             # short description of the request (start of the prompt), for listings
    delivered: bool = False     # a client received the finished image(s)
    announced: bool = False     # a reply gave a client this job's id (or its result): not reused after that
    attached: int = 0           # retries that were attached to this job
    viewer_previews: list | None = None  # small previews for the in-chat viewer, built once
    polls: int = 0              # get_job calls so far
    turn: Any = None            # this job's current turn in the GPU queue (sdserver.Turn) ...
    queue: Any = None           # ... and that queue (sdserver.GpuQueue)

    def place(self) -> tuple[int | None, int | None]:
        """(place in line, 1 = next; None once on the GPU or when not queued), rough seconds until done."""
        q, t = self.queue, self.turn
        if self.status != "running" or q is None or t is None or t not in q.turns:
            return None, None
        now = time.monotonic()
        if t.started is None:
            place, start = q.wait_for(t)
            return place, int(start + t.estimate)
        return None, int(q.remaining(t, now))

    def summary(self) -> dict:
        place, eta = self.place()
        return {"job_id": self.id, "kind": self.kind, "status": self.status,
                "progress_percent": round(self.progress * 100, 1), "message": self.message,
                "elapsed_s": round((self.finished or time.time()) - self.created, 1), "error": self.error,
                "queue_position": place, "eta_seconds": eta, "request": self.label}


def note_turn(turn: Any, queue: Any) -> None:
    """Called by the GPU queue when the running job takes a place in line."""
    job = CURRENT.get()
    if job is not None:
        job.turn, job.queue = turn, queue


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

    def find_reusable(self, key: str) -> Job | None:
        """A job a retry should attach to: same request from the same client, whose id never reached the client (its
        call was cut off: client time limit, disconnect), and which is still running (not started in the last few
        seconds, which would be a parallel request) or finished but never fetched. Once a client has been told a job's
        id, an identical call is a new request (e.g. "another version"). Several retries spread over several such
        jobs, oldest first."""
        now = time.time()
        found = [job for job in self.jobs.values() if job.key == key and not job.announced and (
            (job.status == "running" and now - job.created >= REUSE_MIN_AGE)
            or (job.status == "completed" and not job.delivered and job.finished is not None
                and now - job.finished <= REUSE_UNDELIVERED))]
        if not found:
            return None
        best = min(found, key=lambda j: (j.attached, j.created))
        best.attached += 1
        return best

    def recent(self, seconds: float = 1800) -> list[Job]:
        """Running jobs and jobs that finished in the last ``seconds``, newest first."""
        now = time.time()
        jobs = [j for j in self.jobs.values() if j.status == "running" or (j.finished and now - j.finished <= seconds)]
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def submit(self, kind: str, factory: Callable[[Progress], Awaitable[Any]], key: str | None = None,
               label: str = "") -> Job:
        self._prune()
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, key=key, label=label)

        async def progress(frac: float, message: str) -> None:
            # progress only moves forward, so clients never see it jump back
            job.progress = max(job.progress, min(1.0, float(frac)))
            job.message = message

        async def runner() -> None:
            CURRENT.set(job)
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
