"""Server wiring tests that run without a model (the service is stubbed)."""

import io
from types import SimpleNamespace

from PIL import Image
from starlette.testclient import TestClient

from imagegen_mcp.config import Config
from imagegen_mcp.sdserver import EngineProgress
from imagegen_mcp.server import build_app, create_server
from imagegen_mcp.service import ImageService, OpResult, SavedImage


class StubService(ImageService):
    def __init__(self, cfg, tmp_path):
        super().__init__(cfg)
        self.outputs = tmp_path
        self.state = "ready"
        self.engine = SimpleNamespace(state="ready", progress=EngineProgress())
        self.delay = 0.0

    async def generate(self, **kw):
        import asyncio
        await kw["progress"](0.5, "half way")
        await asyncio.sleep(self.delay)
        img = Image.new("RGB", (64, 32), (200, 10, 10))
        saved = self._save(img, "png", "image")
        return OpResult([saved], {"seed": 1, "width": 64, "height": 32}, ["a note"])


def _cfg(tmp_path) -> Config:
    return Config.model_validate({"outputs_dir": str(tmp_path), "models_dir": str(tmp_path / "m")})


async def test_tools_are_registered_with_schemas(tmp_path):
    mcp = create_server(StubService(_cfg(tmp_path), tmp_path))
    tools = {t.name: t for t in await mcp.list_tools()}
    assert set(tools) == {"generate_image", "edit_image", "generate_panorama", "remove_background", "server_status",
                          "get_job", "cancel_job", "list_images", "view_image", "remove_watermark", "upscale_image"}
    assert tools["remove_watermark"].input_schema["required"] == ["image"]
    gen = tools["generate_image"].input_schema
    assert "prompt" in gen["required"]
    assert "ctx" not in gen["properties"]
    assert {"aspect_ratio", "size", "width", "height", "transparent", "seed"} <= set(gen["properties"])
    edit = tools["edit_image"].input_schema
    assert edit["properties"]["images"]["maxItems"] == 10


async def test_generate_returns_image_and_link(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    mcp = create_server(svc)
    result = await mcp.call_tool("generate_image", {"prompt": "red"})
    kinds = [c.type for c in result.content]
    assert kinds == ["image", "text"]
    assert "http://localhost:5005/outputs/" in result.content[1].text
    assert result.structured_content["images"][0]["width"] == 64
    assert result.structured_content["job_id"]


async def test_slow_job_returns_job_id_then_get_job_returns_image(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    svc.delay = 0.6
    mcp = create_server(svc)
    first = await mcp.call_tool("generate_image", {"prompt": "red", "wait_seconds": 0})
    assert not first.is_error and first.content[0].type == "text"
    job_id = first.structured_content["job_id"]
    assert first.structured_content["status"] == "running"
    assert "get_job" in first.content[0].text
    done = await mcp.call_tool("get_job", {"job_id": job_id, "wait_seconds": 5})
    assert [c.type for c in done.content] == ["image", "text"]
    missing = await mcp.call_tool("get_job", {"job_id": "nope"})
    assert missing.is_error


async def test_model_not_ready_is_a_tool_error(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    svc.state = "downloading"
    mcp = create_server(svc)
    r = await mcp.call_tool("generate_image", {"prompt": "red"})
    assert r.is_error and "downloading" in r.content[0].text


def test_http_routes_and_host_header(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    (tmp_path / "2026-01-01").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "2026-01-01" / "a.png")
    _, app = build_app(svc)
    with TestClient(app, base_url="http://192.168.1.77:5005") as client:
        assert client.get("/health").json()["status"] == "ready"
        r = client.get("/outputs/2026-01-01/a.png")
        assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == (8, 8)
        assert client.get("/outputs/../config.yaml").status_code == 404
        assert "pannellum" in client.get("/view/2026-01-01/a.png").text
        # MCP endpoint must accept a LAN Host header (no 421 from DNS-rebinding protection)
        r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                        headers={"Accept": "application/json, text/event-stream",
                                 "MCP-Protocol-Version": "2025-06-18"})
        assert r.status_code != 421


def test_outputs_serve_images_only(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    (tmp_path / "2026-01-01").mkdir()
    (tmp_path / ".hidden").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "2026-01-01" / "a.png")
    Image.new("RGB", (8, 8)).save(tmp_path / ".hidden" / "b.png")
    (tmp_path / ".previews.json").write_text("{}")
    (tmp_path / "cleanup-manifest-20261006.txt").write_text("x")
    _, app = build_app(svc)
    with TestClient(app) as client:
        assert client.get("/outputs/2026-01-01/a.png").status_code == 200
        for bad in (".previews.json", "cleanup-manifest-20261006.txt", ".hidden/b.png", "2026-01-01",
                    "2026-01-01/../.previews.json", "2026-01-01/a.png%00", "2026-01-01/%2e%2e/.previews.json",
                    "a/" * 2100 + "x.png"):
            assert client.get("/outputs/" + bad).status_code == 404, bad
            assert client.get("/view/" + bad).status_code == 404, bad
        listed = [e["file"] for e in client.get("/api/images").json()["outputs_and_uploads"]]
        assert listed == ["2026-01-01/a.png"]  # nothing from hidden folders


async def test_view_image_links_only_served_files(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    mcp = create_server(svc)
    (tmp_path / ".hidden").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / ".hidden" / "b.png")
    (tmp_path / "2026-01-01").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "2026-01-01" / "a.png")
    shown = await mcp.call_tool("view_image", {"image": "2026-01-01/a.png"})
    assert shown.structured_content["url"].endswith("/outputs/2026-01-01/a.png")
    hidden = await mcp.call_tool("view_image", {"image": ".hidden/b.png"})
    assert hidden.is_error or "url" not in hidden.structured_content


def test_list_images_links_follow_the_client_host(tmp_path):
    import json
    svc = StubService(_cfg(tmp_path), tmp_path)
    (tmp_path / "2026-01-01").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "2026-01-01" / "a.png")
    _, app = build_app(svc)
    with TestClient(app) as client:
        r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                      "params": {"name": "list_images", "arguments": {}}},
                        headers={"Accept": "application/json, text/event-stream", "Host": "198.51.100.7:5005",
                                 "MCP-Protocol-Version": "2025-06-18"})
    lines = [ln[5:] for ln in r.text.splitlines() if ln.startswith("data:")] or [r.text]
    result = json.loads(lines[-1])["result"]
    assert not result.get("isError"), result
    listing = result.get("structuredContent") or json.loads(result["content"][0]["text"])
    urls = [e["url"] for e in listing.get("result", listing)["outputs_and_uploads"]]
    assert urls == ["http://198.51.100.7:5005/outputs/2026-01-01/a.png"]


def test_upload_and_list(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    _, app = build_app(svc)
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), (0, 128, 255)).save(buf, "JPEG")
    with TestClient(app) as client:
        assert "Upload images" in client.get("/upload").text
        assert "Image Gen MCP" in client.get("/", headers={"Accept": "text/html"}).text
        r = client.post("/upload?name=my photo.jpg", content=buf.getvalue(), headers={"Content-Type": "image/jpeg"})
        res = r.json()
        assert r.status_code == 200 and res["file"].startswith("uploads/") and res["file"].endswith(".jpg")
        r2 = client.post("/upload", files={"file": ("x.png", buf.getvalue(), "image/jpeg")})
        assert r2.status_code == 200 and r2.json()["width"] == 40
        bad = client.post("/upload", content=b"not an image")
        assert bad.status_code == 400
        listing = client.get("/api/images").json()
        files = [e["file"] for e in listing["outputs_and_uploads"]]
        assert res["file"] in files
        assert client.get("/outputs/" + res["file"]).status_code == 200


async def test_large_images_are_inlined_as_small_previews(tmp_path):
    import numpy as np

    class BigStub(StubService):
        async def generate(self, **kw):
            noise = (np.random.default_rng(0).random((1536, 1536, 3)) * 255).astype("uint8")
            saved = self._save(Image.fromarray(noise), "png", "image")  # ~7 MB, incompressible
            return OpResult([saved], {"seed": 1})

    mcp = create_server(BigStub(_cfg(tmp_path), tmp_path))
    r = await mcp.call_tool("generate_image", {"prompt": "noise"})
    img = r.content[0]
    assert img.type == "image" and len(img.data) < 1_000_000
    assert r.structured_content["images"][0]["inline"] == "preview"
    assert "reduced preview" in r.content[1].text


async def test_long_invalid_input_error_is_short(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    mcp = create_server(svc)
    r = await mcp.call_tool("remove_background", {"image": "Z" * 300_000})
    assert r.is_error is True or r.structured_content is not None  # matter may be missing in the stub
    text = r.content[0].text
    assert len(text) < 1000


def test_cors_preflight_and_links_follow_host(tmp_path):
    from imagegen_mcp.service import base_url_from_headers
    svc = StubService(_cfg(tmp_path), tmp_path)
    _, app = build_app(svc)
    with TestClient(app) as client:
        r = client.options("/mcp", headers={"Origin": "http://mac.local:5000", "Access-Control-Request-Method": "POST",
                                            "Access-Control-Request-Headers": "content-type,mcp-protocol-version"})
        assert r.status_code == 200 and r.headers.get("access-control-allow-origin")
        buf = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buf, "PNG")
        up = client.post("/upload", content=buf.getvalue(), headers={"Host": "198.51.100.7:5005"}).json()
        assert up["url"].startswith("http://198.51.100.7:5005/outputs/uploads/")
    assert base_url_from_headers({"host": "10.0.0.2:5005"}) == "http://10.0.0.2:5005"
    assert base_url_from_headers({"host": "a", "x-forwarded-host": "img.example.com",
                                  "x-forwarded-proto": "https"}) == "https://img.example.com"
    assert base_url_from_headers({"host": "evil/../x"}) is None


# Python ports of the llama.cpp web UI checks (tools/server/webui, b11090): a tool result is drawn with
# its data-URI images only if it is classified as plain text (no markdown), and an image line must match
# DATA_URI_BASE64_REGEX exactly.
_DATA_URI = __import__("re").compile(r"^data:([^;]+);base64,([A-Za-z0-9+/]+=*)$")
_MD_LINE = __import__("re").compile(r"^(#{1,6} |> |[-*+] |\d+[.)] )")


def _looks_like_markdown(text: str) -> bool:
    import re
    if any(_MD_LINE.match(line) for line in text.splitlines()):
        return True
    if "```" in text or re.search(r"\[[^\]]*\]\([^)]*\)", text) or re.search(r"(\*\*|__)[^*_]+(\*\*|__)", text):
        return True
    return "Title:" in text and "URL:" in text


async def test_data_uri_text_mode_for_llamacpp(tmp_path):
    import numpy as np
    svc = StubService(_cfg(tmp_path), tmp_path)
    mcp = create_server(svc)
    noise = (np.random.default_rng(0).random((1024, 1024, 3)) * 255).astype("uint8")
    saved = svc._save(Image.fromarray(noise), "png", "image")
    r = mcp.build_result(OpResult([saved], {"seed": 1, "steps": 40}), data_uri_text=True)
    assert [c.type for c in r.content] == ["text", "text"]
    head, uri = r.content[0].text, r.content[1].text
    assert not _looks_like_markdown(head)
    assert saved.filename in head and "markdown image" in head
    m = _DATA_URI.match(uri)
    assert m and m.group(1) in ("image/jpeg", "image/png") and len(uri) < 1_000_000
    # the preview maps back to the full-resolution original when it is sent as an input
    img = await svc.loader.load(uri)
    assert img.size == (1024, 1024) and "full-resolution original" in img.info.get("imagegen_note", "")


def test_normal_mode_keeps_image_content(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    mcp = create_server(svc)
    saved = svc._save(Image.new("RGB", (64, 64), (1, 2, 3)), "png", "image")
    r = mcp.build_result(OpResult([saved], {}))
    assert [c.type for c in r.content] == ["image", "text"]


def test_hashed_upload_is_idempotent_and_outputs_have_corp(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    _, app = build_app(svc)
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (9, 9, 9)).save(buf, "PNG")
    with TestClient(app) as client:
        a = client.post("/upload?hash=1&name=chat", content=buf.getvalue()).json()
        b = client.post("/upload?hash=1&name=chat", content=buf.getvalue()).json()
        assert a["file"] == b["file"] and a["file"].startswith("uploads/chat/")
        r = client.get("/outputs/" + a["file"])
        assert r.headers.get("cross-origin-resource-policy") == "cross-origin"


async def test_tool_descriptions_carry_the_configured_guidance(tmp_path):
    cfg = Config.model_validate({"outputs_dir": str(tmp_path), "models_dir": str(tmp_path / "m"),
                                 "generation": {"negative_prompt": "blurry, low quality", "cfg_scale": 3.5}})
    mcp = create_server(StubService(cfg, tmp_path))
    tools = {t.name: t for t in await mcp.list_tools()}
    gen = tools["generate_image"].description
    assert '"blurry, low quality"' in gen and "leave unset (3.5" in gen and 'pass size="xl"' in gen
    assert "omit seed (never pass the earlier seed)" in gen and '"Still working"' in gen
    edit = tools["edit_image"].description
    assert "keep everything else unchanged" in edit and "Never reuse the seed" in edit
    assert "every direction" in tools["generate_panorama"].description
    assert "__" not in mcp.instructions and "3.5 for new images" in mcp.instructions
    assert "added to" in tools["generate_image"].input_schema["properties"]["negative_prompt"]["description"]
    cpu = create_server(StubService(Config.model_validate({"device": "cpu", "outputs_dir": str(tmp_path),
                                                            "models_dir": str(tmp_path / "m")}), tmp_path))
    cpu_gen = {t.name: t for t in await cpu.list_tools()}["generate_image"].description
    assert "the default here is 1 because this server runs on a CPU" in cpu_gen
    assert "lowered for CPU speed; the official setting is 40" in cpu_gen and 'pass size="xl"' not in cpu_gen
    assert "not a hard pixel mask" in tools["edit_image"].description


def test_negative_prompt_merging():
    from imagegen_mcp.service import merge_negative

    notes: list[str] = []
    assert merge_negative("blurry, low quality", "text, Blurry , watermark", 4.0, notes) == \
        "blurry, low quality, text, watermark"
    assert merge_negative("blurry", "", 4.0, notes) == "blurry" and not notes
    assert merge_negative("blurry", "text", 1.0, notes) == "" and "no effect" in notes[0]


async def test_view_image_returns_the_original_file(tmp_path):
    import base64

    svc = StubService(_cfg(tmp_path), tmp_path)
    mcp = create_server(svc)
    day = tmp_path / "2026-09-23"
    day.mkdir()
    Image.new("RGB", (300, 200), (10, 200, 30)).save(day / "image-1.png")
    original = (day / "image-1.png").read_bytes()
    for ref in ("2026-09-23/image-1.png", "image-1.png", "http://192.168.1.5:5005/outputs/2026-09-23/image-1.png"):
        result = await mcp.call_tool("view_image", {"image": ref})
        assert not result.is_error
        img = next(c for c in result.content if c.type == "image")
        assert base64.b64decode(img.data) == original and img.mime_type == "image/png"
        assert result.structured_content["width"] == 300 and result.structured_content["file"] == "2026-09-23/image-1.png"
    Image.new("RGBA", (64, 48), (1, 2, 3, 128)).save(day / "cut.webp", lossless=True)
    result = await mcp.call_tool("view_image", {"image": "2026-09-23/cut.webp"})
    img = next(c for c in result.content if c.type == "image")
    assert img.mime_type == "image/png" and Image.open(io.BytesIO(base64.b64decode(img.data))).size == (64, 48)
    missing = await mcp.call_tool("view_image", {"image": "260921-some-news-photo.jpg"})
    assert missing.is_error and "img_1" in missing.content[0].text


async def test_unknown_attachment_name_error_suggests_the_chat_handle(tmp_path):
    from imagegen_mcp import imaging

    loader = imaging.ImageLoader(tmp_path)
    try:
        await loader.load("260921-zohran-photo.jpg")
    except imaging.ImageInputError as exc:
        assert "img_1" in str(exc) and "not the file name" in str(exc)
    else:
        raise AssertionError("expected an error")


async def test_recommended_size_changes_the_size_guidance(tmp_path):
    def gen_desc(rec):
        cfg = Config.model_validate({"outputs_dir": str(tmp_path), "models_dir": str(tmp_path / "m"),
                                     "generation": {"recommended_size": rec}})
        return create_server(StubService(cfg, tmp_path))

    for rec, expect, absent in (("xl", 'pass size="xl"', 'pass size="large"'),
                                ("large", 'pass size="large"', 'pass size="xl"'),
                                ("medium", 'leave size unset ("medium"', 'pass size="xl"')):
        mcp = gen_desc(rec)
        desc = {t.name: t for t in await mcp.list_tools()}["generate_image"].description
        assert expect in desc and absent not in desc, rec
        assert "__" not in mcp.instructions


async def test_remove_background_tool_runs(tmp_path):
    svc = StubService(_cfg(tmp_path), tmp_path)
    svc.matter = object()

    async def fake_remove_background(**kw):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        return OpResult([svc._save(img, "png", "cutout")], {"width": 32, "height": 32}, [])

    svc.remove_background = fake_remove_background
    mcp = create_server(svc)
    res = await mcp.call_tool("remove_background", {"image": "x.png"})
    assert not res.is_error and res.structured_content["images"][0]["width"] == 32
