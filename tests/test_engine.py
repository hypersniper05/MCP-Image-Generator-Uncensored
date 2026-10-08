import asyncio
import base64
from types import SimpleNamespace

import httpx
import pytest

from imagegen_mcp import sdserver
from imagegen_mcp.config import Config
from imagegen_mcp.devices import build_device_plan
from imagegen_mcp.sdserver import EngineError, SdServer

TICK = 10.0  # seconds of fake time per poll


def _engine(monkeypatch, statuses, *, ahead_progresses=True):
    """An SdServer whose sd-server reports ``statuses`` in turn, one per poll, on a fake clock."""
    cfg = Config.model_validate({})
    e = SdServer(cfg, build_device_plan(cfg), {})
    clock = SimpleNamespace(t=1000.0)
    monkeypatch.setattr(sdserver, "time", SimpleNamespace(monotonic=lambda: clock.t))

    async def no_wait(_seconds):
        clock.t += TICK

    monkeypatch.setattr(sdserver.asyncio, "sleep", no_wait)
    e.progress.updated = clock.t
    e.proc = SimpleNamespace(returncode=None)
    e.cancelled = False

    async def running():
        return None

    e.ensure_running = running
    polls = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sdcpp/v1/img_gen":
            return httpx.Response(200, json={"id": "j1"})
        if request.url.path.endswith("/cancel"):
            e.cancelled = True
            return httpx.Response(200, json={})
        status = next(polls)
        if status == "queued" and ahead_progresses:
            e.progress.updated = clock.t  # the job ahead keeps logging denoising steps
        if status == "completed":
            png = base64.b64encode(b"png-bytes").decode()
            return httpx.Response(200, json={"status": status, "result": {"images": [{"b64_json": png}]}})
        return httpx.Response(200, json={"status": status, "queue_position": 1})

    e._client = httpx.AsyncClient(base_url=e.base_url, transport=httpx.MockTransport(handler))
    return e


async def test_queue_wait_does_not_count_toward_timeout(monkeypatch):
    # 200 s behind other jobs, then 50 s of work: fine with a 100 s timeout
    e = _engine(monkeypatch, ["queued"] * 20 + ["generating"] * 5 + ["completed"])
    assert await e.generate({}, timeout=100) == [b"png-bytes"]


async def test_running_time_is_limited(monkeypatch):
    e = _engine(monkeypatch, ["queued"] * 3 + ["generating"] * 30)
    with pytest.raises(EngineError, match="timed out after 100 s"):
        await e.generate({}, timeout=100)
    await asyncio.gather(*e.queue._settling)  # the card stays taken until sd-server lets go of the job
    assert e.cancelled and not e.queue.turns


async def test_idle_engine_does_not_fail_a_briefly_queued_job(monkeypatch):
    # the engine's last progress is long ago (it was idle), but this job only waits a moment in the queue
    e = _engine(monkeypatch, ["queued"] * 2 + ["generating"] * 2 + ["completed"], ahead_progresses=False)
    e.progress.updated -= 500
    assert await e.generate({}, timeout=100) == [b"png-bytes"]


async def test_stuck_engine_fails_queued_job(monkeypatch):
    e = _engine(monkeypatch, ["queued"] * 30, ahead_progresses=False)
    with pytest.raises(EngineError, match="no progress"):
        await e.generate({}, timeout=100)
    await asyncio.gather(*e.queue._settling)
