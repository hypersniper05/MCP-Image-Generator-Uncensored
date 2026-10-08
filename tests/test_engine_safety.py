"""Engine start, cancel and error paths never strand the engine or let two jobs share the card."""
import asyncio
import io
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from imagegen_mcp import sdserver
from imagegen_mcp.config import Config
from imagegen_mcp.devices import build_device_plan
from imagegen_mcp.sdserver import SdServer
from test_queue import Card


def _engine():
    cfg = Config.model_validate({})
    return SdServer(cfg, build_device_plan(cfg), {})


def _fake_start(e, starts, seconds=0.5):
    async def start():
        starts.append(time.monotonic())
        e.state = "starting"
        await asyncio.sleep(seconds)
        e.proc = SimpleNamespace(returncode=None)
        e.state = "ready"
    return start


async def test_cancel_during_reload_does_not_strand_the_engine():
    """A job cancelled while the engine reloads used to leave it in 'starting', and every later job failed."""
    e = _engine()
    e.state, starts = "unloaded", []
    e.start = _fake_start(e, starts)
    first = asyncio.create_task(e.ensure_running())
    await asyncio.sleep(0.1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(e.ensure_running(), 5)  # the next job waits for the same start instead of failing
    assert e.state == "ready" and len(starts) == 1


async def test_dead_engine_still_marked_ready_is_restarted():
    e = _engine()
    e.state, e.proc, starts = "ready", SimpleNamespace(returncode=1), []
    e._restarts = -1  # the first restart then waits 1 s instead of 2
    e.start = _fake_start(e, starts, 0.0)
    await asyncio.wait_for(e.ensure_running(), 10)
    assert e.state == "ready" and e.proc.returncode is None and len(starts) == 1


async def test_failed_start_is_retried_by_the_next_job():
    e = _engine()
    e.state, starts = "unloaded", []

    async def broken_start():
        starts.append(1)
        e.state = "starting"
        e.proc = SimpleNamespace(returncode=1)
        raise sdserver.EngineError("sd-server exited during startup")

    e.start = broken_start
    with pytest.raises(sdserver.EngineError, match="during startup"):
        await e.ensure_running()
    e._restarts = -1
    e.start = _fake_start(e, starts, 0.0)
    await asyncio.wait_for(e.ensure_running(), 10)
    assert e.state == "ready" and len(starts) == 2


class SlowCli:
    """sd-cli that runs until killed and takes 0.3 s to exit after the kill."""

    def __init__(self, events):
        self.events, self.returncode, self.killed = events, None, False
        self._exit = None

    async def communicate(self):
        await asyncio.sleep(30)

    def kill(self):
        self.killed = True

        async def die():
            await asyncio.sleep(0.3)
            self.returncode = -9
            self.events.append("sd-cli exited")
        self._exit = asyncio.ensure_future(die())

    async def wait(self):
        if self.returncode is None:
            await self._exit
        return self.returncode


async def test_cancelled_upscale_stops_sd_cli_and_keeps_the_card_until_it_exits(monkeypatch):
    e, events, procs = _engine(), [], []

    async def fake_exec(*cmd, **kw):
        procs.append(SlowCli(events))
        return procs[-1]

    monkeypatch.setattr(sdserver.asyncio, "create_subprocess_exec", fake_exec)
    png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(png, "PNG")
    up = asyncio.create_task(e.upscale(png.getvalue(), model=Path("m.safetensors"), tile_size=128, timeout=60))
    await asyncio.sleep(0.2)
    up.cancel()

    async def next_job():
        async with e.queue.turn("generate", 1.0):
            events.append("next job started")

    nxt = asyncio.create_task(next_job())
    with pytest.raises(asyncio.CancelledError):
        await up
    await asyncio.wait_for(nxt, 5)
    assert procs[0].killed and events == ["sd-cli exited", "next job started"]


def _wrap_transport(card, handler):
    card.e._client = httpx.AsyncClient(base_url=card.e.base_url, transport=httpx.MockTransport(handler))


async def test_cancel_while_the_request_is_on_its_way_still_drops_the_job(monkeypatch):
    """sd-server already has the job when the cancel lands mid-request: it must be cancelled there, and the next
    job waits until it has stopped."""
    card = Card(monkeypatch)
    inner = card.e._client._transport.handler

    async def slow_answer(request):
        resp = inner(request)  # sd-server takes (and starts) the job at once ...
        if request.url.path == "/sdcpp/v1/img_gen":
            await asyncio.sleep(0.5)  # ... but its answer is still on the way
        return resp

    _wrap_transport(card, slow_answer)
    g1 = card.generate("g1", 1.0)
    await asyncio.sleep(0.2)
    g1.cancel()
    up = card.upscale()
    await asyncio.wait_for(up, 10)
    assert card.jobs["0"]["cancelled"]
    assert card.order() == ["g1", "upscale"]
    card.assert_never_shared()


async def test_error_while_polling_keeps_the_card_until_the_engine_stops(monkeypatch):
    card = Card(monkeypatch)
    inner = card.e._client._transport.handler
    bad = {"left": 1}

    def garbled_once(request):
        path = request.url.path
        if "/jobs/" in path and not path.endswith("/cancel") and bad["left"]:
            bad["left"] -= 1
            return httpx.Response(200, content=b"not json")
        return inner(request)

    _wrap_transport(card, garbled_once)
    g1 = card.generate("g1", 1.0)
    await asyncio.sleep(0.1)
    up = card.upscale()
    with pytest.raises(ValueError):
        await g1
    await asyncio.wait_for(up, 10)
    assert card.jobs["0"]["cancelled"]
    assert card.order() == ["g1", "upscale"]
    card.assert_never_shared()


async def test_a_good_job_resets_the_crash_back_off(monkeypatch):
    card = Card(monkeypatch)
    card.e._restarts = 4
    await asyncio.wait_for(card.generate("g1", 0.1), 10)
    assert card.e._restarts == 0


async def test_second_cancel_while_dropping_the_job_still_holds_the_card(monkeypatch):
    """cancel_job twice in a row: the second cancel used to land while the first one's cancel request was on its
    way, skip the hand-over, and let the next job share the card with the still-running generation."""
    card = Card(monkeypatch)
    inner = card.e._client._transport.handler

    async def slow_cancel(request):
        if request.url.path.endswith("/cancel"):
            await asyncio.sleep(0.3)
        return inner(request)

    _wrap_transport(card, slow_cancel)
    g1 = card.generate("g1", 1.5)
    await asyncio.sleep(0.3)
    up = card.upscale()
    g1.cancel()
    await asyncio.sleep(0.05)
    g1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await g1
    await asyncio.wait_for(up, 10)
    assert card.jobs["0"]["cancelled"]
    assert card.order() == ["g1", "upscale"]
    card.assert_never_shared()


async def test_job_cancelled_during_a_reload_keeps_the_card_until_the_load_ends(monkeypatch):
    card = Card(monkeypatch)
    e, starts, loaded = card.e, [], {}
    del e.ensure_running  # use the real one, with a fake 0.8 s load
    e.state, e.proc = "unloaded", None
    fake = _fake_start(e, starts, 0.8)

    async def start():
        await fake()
        loaded["at"] = time.monotonic()

    e.start = start
    g1 = card.generate("g1", 0.3)
    await asyncio.sleep(0.2)
    up = card.upscale()
    g1.cancel()
    await asyncio.wait_for(up, 10)
    up_start = next(s for label, s, _ in card.spans if label == "upscale")
    assert up_start >= loaded["at"] and e.state == "ready" and len(starts) == 1
