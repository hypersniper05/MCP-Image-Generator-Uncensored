"""Job flow: the server-chosen wait, reuse of identical requests, pending replies and get_job."""
import asyncio
import json

from PIL import Image
from starlette.testclient import TestClient

from imagegen_mcp import jobs
from imagegen_mcp.config import Config
from imagegen_mcp.sdserver import GpuQueue
from imagegen_mcp.server import build_app, create_server
from imagegen_mcp.service import OpResult
from test_server import StubService


def _svc(tmp_path, wait=0, delay=0.6):
    cfg = Config.model_validate({"outputs_dir": str(tmp_path), "models_dir": str(tmp_path / "m"),
                                 "generation": {"wait_seconds": wait}})
    svc = StubService(cfg, tmp_path)
    svc.delay = delay
    svc.calls = 0
    real = svc.generate

    async def counting(**kw):
        svc.calls += 1
        return await real(**kw)

    svc.generate = counting
    return svc


async def test_pending_reply_says_what_to_do_next(tmp_path):
    mcp = create_server(_svc(tmp_path))
    first = await mcp.call_tool("generate_image", {"prompt": "red", "wait_seconds": 999})  # old clients: ignored
    assert not first.is_error
    text, sc = first.content[0].text, first.structured_content
    assert text.startswith("Not finished yet. This is normal") and "Do NOT re-render the image again." in text
    assert sc["status"] == "running" and sc["reused"] is False and sc["wait_budget_seconds"] == 0
    assert sc["next"] == {"tool": "get_job", "arguments": {"job_id": sc["job_id"], "poll": 1}}
    assert json.dumps(sc["next"]["arguments"]) in text and "Do NOT re-render" in sc["next_step"]
    await asyncio.sleep(0.8)
    done = await mcp.call_tool("get_job", sc["next"]["arguments"])
    assert [c.type for c in done.content] == ["image", "text"] and done.structured_content["job_id"] == sc["job_id"]


async def _cut_off(mcp, args, tool="generate_image"):
    """A call whose client gave up (time limit, disconnect) before any reply: the server never sent the job id."""
    task = asyncio.ensure_future(mcp.call_tool(tool, args))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_a_retry_after_a_cut_off_call_attaches_to_the_running_job(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "REUSE_MIN_AGE", 0.05)
    svc = _svc(tmp_path, wait=30, delay=1.0)
    mcp = create_server(svc)
    await _cut_off(mcp, {"prompt": "red"})
    first = next(iter(svc.jobs.jobs.values()))
    svc.cfg.generation.wait_seconds = 0
    b = await mcp.call_tool("generate_image", {"prompt": "red"})  # the model retried after its client timed out
    assert b.structured_content["job_id"] == first.id and b.structured_content["reused"] is True
    assert b.content[0].text.startswith(f"This exact request is already running as job {first.id}")
    c = (await mcp.call_tool("generate_image", {"prompt": "red", "seed": 7})).structured_content
    assert c["job_id"] != first.id  # different arguments: a new job
    await asyncio.sleep(1.2)
    assert svc.calls == 2


async def test_another_version_after_a_reply_is_a_new_render(tmp_path, monkeypatch):
    """A client that runs calls one at a time asks for 3 versions: each call already got its job id back, so the next
    identical call is a new request, not a retry."""
    monkeypatch.setattr(jobs, "REUSE_MIN_AGE", 0.0)
    svc = _svc(tmp_path, delay=0.5)
    mcp = create_server(svc)
    ids = [(await mcp.call_tool("generate_image", {"prompt": "red"})).structured_content["job_id"] for _ in range(3)]
    assert len(set(ids)) == 3
    await asyncio.sleep(0.7)
    assert svc.calls == 3


async def test_cut_off_retries_spread_over_their_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "REUSE_MIN_AGE", 0.05)
    svc = _svc(tmp_path, wait=30, delay=1.0)
    mcp = create_server(svc)
    await asyncio.gather(*[_cut_off(mcp, {"prompt": "red"}) for _ in range(2)])  # 2 versions, both cut off
    originals = {j.id for j in svc.jobs.jobs.values()}
    svc.cfg.generation.wait_seconds = 0
    retried = {(await mcp.call_tool("generate_image", {"prompt": "red"})).structured_content["job_id"] for _ in range(2)}
    assert retried == originals and svc.calls == 2
    await asyncio.sleep(1.2)


async def test_an_unfetched_result_of_a_cut_off_call_is_returned_to_the_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "REUSE_MIN_AGE", 0.0)
    svc = _svc(tmp_path, wait=30, delay=0.2)
    mcp = create_server(svc)
    await _cut_off(mcp, {"prompt": "red"})
    first = next(iter(svc.jobs.jobs.values()))
    await asyncio.sleep(0.4)  # finished, but nobody got it
    again = await mcp.call_tool("generate_image", {"prompt": "red"})
    assert again.structured_content["job_id"] == first.id and again.content[0].type == "image"
    new = (await mcp.call_tool("generate_image", {"prompt": "red"})).structured_content  # delivered: a re-roll
    assert new["job_id"] != first.id
    await asyncio.sleep(0.3)
    assert svc.calls == 2


async def test_failed_and_cancelled_jobs_are_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "REUSE_MIN_AGE", 0.0)
    svc = _svc(tmp_path, wait=30, delay=5.0)
    mcp = create_server(svc)
    await _cut_off(mcp, {"prompt": "red"})
    job = next(iter(svc.jobs.jobs.values()))
    svc.jobs.cancel(job.id)
    await job.done.wait()
    svc.cfg.generation.wait_seconds = 0
    r = (await mcp.call_tool("generate_image", {"prompt": "red"})).structured_content
    assert r["job_id"] != job.id and r["reused"] is False
    svc.jobs.cancel(r["job_id"])


async def test_get_job_several_ids_and_no_id(tmp_path):
    svc = _svc(tmp_path, delay=0.2)
    mcp = create_server(svc)
    fast = (await mcp.call_tool("generate_image", {"prompt": "red"})).structured_content["job_id"]
    await asyncio.sleep(0.05)  # the fast job has started its 0.2 s render
    svc.delay = 3.0
    slow = (await mcp.call_tool("generate_image", {"prompt": "blue"})).structured_content["job_id"]
    await asyncio.sleep(0.4)
    r = await mcp.call_tool("get_job", {"job_ids": [fast, slow]})
    sc = r.structured_content
    assert [e["job_id"] for e in sc["jobs"]] == [fast, slow] and sc["jobs"][0]["images"]
    assert sc["next"]["arguments"] == {"job_id": slow, "poll": 2} and any(c.type == "image" for c in r.content)
    listed = await mcp.call_tool("get_job", {})
    assert not listed.is_error and {fast, slow} <= {j["job_id"] for j in listed.structured_content["jobs"]}
    unknown = await mcp.call_tool("get_job", {"job_id": "nope"})
    assert unknown.is_error and fast in unknown.content[0].text
    await asyncio.sleep(3.0)


async def test_a_failed_job_tells_the_model_not_to_retry(tmp_path):
    svc = _svc(tmp_path, wait=5)

    async def broken(**kw):
        raise ValueError("bad size")

    svc.generate = broken
    r = await create_server(svc).call_tool("generate_image", {"prompt": "red"})
    assert r.is_error and "Do not retry automatically" in r.content[0].text


def test_the_user_sets_the_wait_on_the_url_or_header(tmp_path):
    svc = _svc(tmp_path, wait=20, delay=0.5)
    _, app = build_app(svc)

    with TestClient(app) as client:
        def call(path, prompt, headers=None):
            r = client.post(path, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                        "params": {"name": "generate_image", "arguments": {"prompt": prompt}}},
                            headers={"Accept": "application/json, text/event-stream", **(headers or {})})
            line = [ln[5:] for ln in r.text.splitlines() if ln.startswith("data:")][-1]
            res = json.loads(line)["result"]
            return res.get("structuredContent") or json.loads(res["content"][-1]["text"])

        assert call("/mcp?max_wait=0", "a")["wait_budget_seconds"] == 0
        assert call("/mcp?max_wait=999", "b", {"X-Imagegen-Max-Wait": "0"})["wait_budget_seconds"] == 0  # header wins
        assert call("/mcp?max_wait=0", "c", {"X-Imagegen-Max-Wait": "999"}).get("images")  # 999 -> 280: it finishes


async def test_waiting_jobs_report_their_place_in_line():
    q = GpuQueue()
    job = jobs.Job(id="j1", kind="generate")

    async def holder():
        async with q.turn("generate", 60.0):
            await asyncio.sleep(0.3)

    async def waiter():
        jobs.CURRENT.set(job)
        async with q.turn("generate", 30.0):
            pass

    h = asyncio.create_task(holder())
    await asyncio.sleep(0.01)
    w = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    place, eta = job.place()
    assert place == 1 and 85 <= eta <= 90  # ~60 s left of the running job + its own 30 s
    assert job.summary()["queue_position"] == 1
    await asyncio.gather(h, w)
    assert job.place() == (None, None)


async def test_parallel_identical_calls_are_separate_jobs(tmp_path):
    svc = _svc(tmp_path, delay=0.3)
    mcp = create_server(svc)
    r = await asyncio.gather(*[mcp.call_tool("generate_image", {"prompt": "red"}) for _ in range(3)])
    assert len({x.structured_content["job_id"] for x in r}) == 3  # "make 3 versions" stays 3 renders
    await asyncio.sleep(0.5)


async def test_several_finished_jobs_share_one_inline_budget(tmp_path):
    svc = _svc(tmp_path, delay=0.0)
    mcp = create_server(svc)
    ids = [(await mcp.call_tool("generate_image", {"prompt": p})).structured_content["job_id"] for p in "abc"]
    await asyncio.sleep(0.1)
    for j in ids:
        svc.jobs.get(j).delivered = svc.jobs.get(j).announced = False
    total = svc.cfg.outputs.inline_max_bytes
    r = await mcp.call_tool("get_job", {"job_ids": ids})
    imgs = [c for c in r.content if c.type == "image"]
    assert len(imgs) == 3 and sum(len(c.data) * 3 // 4 for c in imgs) <= total
    assert all(svc.jobs.get(j).delivered for j in ids)


def test_junk_and_huge_max_wait_values(tmp_path, monkeypatch):
    from imagegen_mcp import server as server_mod
    svc = _svc(tmp_path, wait=20, delay=0.0)
    _, app = build_app(svc)
    with TestClient(app) as client:
        def budget(path, prompt, headers=None):
            r = client.post(path, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                        "params": {"name": "generate_image", "arguments": {"prompt": prompt}}},
                            headers={"Accept": "application/json, text/event-stream", **(headers or {})})
            line = [ln[5:] for ln in r.text.splitlines() if ln.startswith("data:")][-1]
            res = json.loads(line)["result"]
            assert not res.get("isError"), res
            return res

        budget("/mcp?max_wait=%C2%B2", "unicode digit")  # ignored, not an error
        budget("/mcp?max_wait=-5", "negative")
        monkeypatch.setattr(server_mod, "MAX_WAIT_LIMIT", 0)  # with limit 0 the clamp shows in the reply
        svc.delay = 0.5
        for path, hdr, p in (("/mcp?max_wait=999", None, "d"), ("/mcp", {"X-Imagegen-Max-Wait": "999"}, "e")):
            sc = budget(path, p, hdr)["structuredContent"]
            assert sc["wait_budget_seconds"] == 0


def test_old_configs_with_a_long_wait_still_load():
    assert Config.model_validate({"generation": {"wait_seconds": 3600}}).generation.wait_seconds == 3600


async def test_poll_counter_and_job_labels(tmp_path):
    svc = _svc(tmp_path, delay=1.0)
    mcp = create_server(svc)
    first = (await mcp.call_tool("generate_image", {"prompt": "a red fox in the snow"})).structured_content
    second = (await mcp.call_tool("get_job", first["next"]["arguments"])).structured_content
    third = (await mcp.call_tool("get_job", second["next"]["arguments"])).structured_content
    assert [x["next"]["arguments"]["poll"] for x in (first, second, third)] == [1, 2, 3]
    listing = await mcp.call_tool("get_job", {})
    assert '"a red fox in the snow"' in listing.content[0].text
    assert listing.structured_content["jobs"][0]["request"] == "a red fox in the snow"
    await asyncio.sleep(1.2)
