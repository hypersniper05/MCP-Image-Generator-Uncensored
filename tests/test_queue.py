"""The engine queue: generations and upscales take turns on the card in arrival order."""
import asyncio
import base64
import io
import json
import time
from types import SimpleNamespace

import httpx
from PIL import Image

from imagegen_mcp import sdserver
from imagegen_mcp.config import Config
from imagegen_mcp.devices import build_device_plan
from imagegen_mcp.sdserver import GpuQueue, SdServer


def _png(w=4, h=4):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, "PNG")
    return buf.getvalue()


class Card:
    """A fake sd-server plus sd-cli that records when each job really used the card."""

    def __init__(self, monkeypatch, up_s=0.3):
        cfg = Config.model_validate({})
        self.e = e = SdServer(cfg, build_device_plan(cfg), {})
        e.proc = SimpleNamespace(returncode=None)

        async def running():
            return None

        e.ensure_running = running
        self.spans: list[tuple[str, float, float]] = []  # (label, start, end) on the card
        self.jobs: dict[str, dict] = {}
        self.up_s = up_s
        b64 = base64.b64encode(_png()).decode()

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/sdcpp/v1/img_gen":
                body = json.loads(request.content)
                jid = str(len(self.jobs))
                now = time.monotonic()
                self.jobs[jid] = {"label": body["prompt"], "start": now, "end": now + body["dur"], "cancelled": False}
                self.spans.append((body["prompt"], now, now + body["dur"]))
                return httpx.Response(200, json={"id": jid})
            jid = path.split("/")[4]
            job = self.jobs[jid]
            if path.endswith("/cancel"):
                job["cancelled"] = True  # like sd-server: a started job still runs to the end
                return httpx.Response(200, json={})
            if time.monotonic() < job["end"]:
                return httpx.Response(200, json={"status": "generating"})
            if job["cancelled"]:
                return httpx.Response(200, json={"status": "cancelled"})
            return httpx.Response(200, json={"status": "completed", "result": {"images": [{"b64_json": b64}]}})

        e._client = httpx.AsyncClient(base_url=e.base_url, transport=httpx.MockTransport(handler))
        card = self

        class Proc:
            def __init__(self, cmd):
                self.cmd, self.returncode = cmd, None

            async def communicate(self):
                t0 = time.monotonic()
                await asyncio.sleep(card.up_s)
                Image.new("RGB", (16, 16)).save(self.cmd[self.cmd.index("-o") + 1])
                card.spans.append(("upscale", t0, time.monotonic()))
                self.returncode = 0
                return b"", None

        async def fake_exec(*cmd, **kw):
            return Proc(list(cmd))

        monkeypatch.setattr(sdserver.asyncio, "create_subprocess_exec", fake_exec)

    def generate(self, label, dur, timeout=60.0, messages=None):
        async def on_progress(frac, msg):
            if messages is not None:
                messages.append(msg)
        body = {"prompt": label, "dur": dur, "width": 512, "height": 512, "sample_params": {"sample_steps": 40}}
        return asyncio.create_task(self.e.generate(body, timeout=timeout, on_progress=on_progress))

    def upscale(self, timeout=60.0, messages=None):
        async def on_wait(msg):
            if messages is not None:
                messages.append(msg)
        return asyncio.create_task(self.e.upscale(_png(), model=sdserver.Path("m.safetensors"), tile_size=128,
                                                  timeout=timeout, on_wait=on_wait))

    def order(self):
        return [label for label, _, _ in sorted(self.spans, key=lambda s: s[1])]

    def assert_never_shared(self):
        spans = sorted(self.spans, key=lambda s: s[1])
        for (a, _, a_end), (b, b_start, _) in zip(spans, spans[1:]):
            assert a_end <= b_start, f"{a} and {b} shared the card: {spans}"


async def test_generation_behind_a_waiting_upscale_does_not_deadlock(monkeypatch):
    """A generation runs, an upscale waits for it, then another generation arrives. The old scheme deadlocked here:
    the upscale waited for "no generations" while the new generation waited for the upscale."""
    card = Card(monkeypatch)
    g1 = card.generate("g1", 0.8)
    await asyncio.sleep(0.1)
    up = card.upscale()
    await asyncio.sleep(0.1)
    g2 = card.generate("g2", 0.3)
    await asyncio.wait_for(asyncio.gather(g1, up, g2), 10)
    assert card.order() == ["g1", "upscale", "g2"]
    card.assert_never_shared()
    assert not card.e.queue.turns and card.e._active == 0


async def test_upscale_behind_a_long_queue_runs_in_order_and_never_times_out_waiting(monkeypatch):
    """A long generation runs and two more are queued when an upscale arrives: everything runs in arrival
    order, and neither the upscale nor the last generation counts its wait toward its own timeout."""
    card = Card(monkeypatch, up_s=0.2)
    g3_msgs, up_msgs = [], []
    g1 = card.generate("g1", 1.2, timeout=1.5)
    await asyncio.sleep(0.1)
    g2 = card.generate("g2", 0.4, timeout=1.5)
    g3 = card.generate("g3", 0.4, timeout=1.5, messages=g3_msgs)
    await asyncio.sleep(0.1)
    snap = card.e.queue.snapshot()
    assert snap["running"] == "generate" and snap["waiting"] == 2 and snap["drain_eta_s"] > 0
    up = card.upscale(timeout=0.5, messages=up_msgs)  # it will wait far longer than 0.5 s
    await asyncio.wait_for(asyncio.gather(g1, g2, g3, up), 15)
    assert card.order() == ["g1", "g2", "g3", "upscale"]
    card.assert_never_shared()
    up_start = next(s for label, s, _ in card.spans if label == "upscale")
    g1_start = next(s for label, s, _ in card.spans if label == "g1")
    assert up_start - g1_start > 1.5  # it really waited longer than every timeout involved
    assert up_msgs[0].startswith("queued: position 3 in line, starts in about")
    assert any("position 2 in line" in m for m in g3_msgs) and any("position 1 in line" in m for m in g3_msgs)
    assert card.e.queue.snapshot() == {"running": None, "waiting": 0, "drain_eta_s": 0}


async def test_cancelled_waiting_job_leaves_the_line(monkeypatch):
    card = Card(monkeypatch)
    g1 = card.generate("g1", 0.6)
    await asyncio.sleep(0.1)
    g2 = card.generate("g2", 0.3)
    g3 = card.generate("g3", 0.3)
    await asyncio.sleep(0.1)
    g2.cancel()
    await asyncio.wait_for(asyncio.gather(g1, g3), 10)
    assert card.order() == ["g1", "g3"] and g2.cancelled()
    assert not card.e.queue.turns and card.e._active == 0


async def test_abandoned_generation_keeps_the_card_until_the_engine_stops(monkeypatch):
    """sd-server finishes a started job even when it is cancelled, so the next job waits for that."""
    card = Card(monkeypatch)
    g1 = card.generate("g1", 1.0)
    await asyncio.sleep(0.3)
    g1.cancel()
    await asyncio.sleep(0.1)
    up = card.upscale()
    await asyncio.wait_for(up, 10)
    assert card.order() == ["g1", "upscale"]
    card.assert_never_shared()
    assert not card.e.queue.turns


async def test_queue_wait_estimates():
    q = GpuQueue()
    async with q.turn("generate", 100.0):
        waiting = asyncio.create_task(_hold(q, "upscale", 30.0))
        await asyncio.sleep(0)
        last = asyncio.create_task(_hold(q, "generate", 50.0))
        await asyncio.sleep(0)
        place, secs = q.wait_for(q.turns[2])
        assert place == 2 and 125 <= secs <= 130  # ~100 s left of the running job, plus the 30 s upscale
        assert q.describe(q.turns[2]) == "queued: position 2 in line, starts in about 2 min"
        assert q.snapshot()["waiting"] == 2
    await asyncio.gather(waiting, last)
    assert not q.turns


async def _hold(q, kind, estimate):
    async with q.turn(kind, estimate):
        await asyncio.sleep(0)
