"""The in-chat image viewer (MCP Apps): metadata on the image tools, the ui:// page, the app-only job_status tool."""
import asyncio
import base64
import re
import shutil
import subprocess

import pytest

from imagegen_mcp import viewer
from imagegen_mcp.server import create_server
from test_jobflow import _svc


async def test_image_tools_declare_the_viewer_and_job_status_is_app_only(tmp_path):
    tools = {t.name: t for t in await create_server(_svc(tmp_path)).list_tools()}
    for name in ("generate_image", "edit_image", "generate_panorama", "remove_background", "remove_watermark",
                 "upscale_image"):
        assert tools[name].meta["ui"] == {"resourceUri": viewer.URI}, name
    assert tools["job_status"].meta["ui"] == {"resourceUri": viewer.URI, "visibility": ["app"]}
    for name in ("get_job", "cancel_job", "list_images", "view_image", "server_status"):
        assert not tools[name].meta, name  # no second viewer under get_job


async def test_the_viewer_page_is_served_as_an_mcp_app(tmp_path):
    mcp = create_server(_svc(tmp_path))
    res = {r.uri: r for r in await mcp.list_resources()}
    assert res[viewer.URI].mime_type == "text/html;profile=mcp-app"
    page = (await mcp.read_resource(viewer.URI))[0]
    html = page.content if isinstance(page.content, str) else page.content.decode()
    assert "ui/initialize" in html and '"job_status"' in html and "<script src" not in html  # no external scripts


async def test_job_status_reports_progress_then_small_previews_without_marking_delivered(tmp_path):
    svc = _svc(tmp_path, delay=0.3)
    mcp = create_server(svc)
    job_id = (await mcp.call_tool("generate_image", {"prompt": "red"})).structured_content["job_id"]
    running = await mcp.call_tool("job_status", {"job_id": job_id})
    assert not running.is_error and running.structured_content["status"] == "running"
    assert "call get_job with" in running.content[0].text  # useful to a model in a client without the viewer
    await asyncio.sleep(0.5)
    done = await mcp.call_tool("job_status", {"job_id": job_id})
    img = [c for c in done.content if c.type == "image"]
    assert img and len(base64.b64decode(img[0].data)) <= 100_000
    file = done.structured_content["images"][0]["file"]
    assert file.endswith(".png") and file in done.content[-1].text and "get_job" in done.content[-1].text
    assert done.structured_content["delivered"] is False
    assert not svc.jobs.get(job_id).delivered  # the model can still fetch it with get_job
    cached = svc.jobs.get(job_id).viewer_previews
    again = await mcp.call_tool("job_status", {"job_id": job_id})
    assert svc.jobs.get(job_id).viewer_previews is cached and again.content[0].data == img[0].data
    gone = await mcp.call_tool("job_status", {"job_id": "nope"})  # e.g. a reopened chat after a restart
    assert not gone.is_error and gone.structured_content["status"] == "expired"


async def test_finished_results_say_completed(tmp_path):
    r = await create_server(_svc(tmp_path, wait=5, delay=0.0)).call_tool("generate_image", {"prompt": "red"})
    assert r.structured_content["status"] == "completed" and r.structured_content["images"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_viewer_script_is_valid_javascript(tmp_path):
    script = re.search(r"<script>(.*?)</script>", viewer.HTML, re.S).group(1)
    js = tmp_path / "viewer.js"
    js.write_text(script, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
