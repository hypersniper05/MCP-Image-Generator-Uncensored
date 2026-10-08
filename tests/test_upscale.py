import base64
import io
import json
import re
from types import SimpleNamespace
from xml.etree import ElementTree

import httpx
import numpy as np
import pytest
from PIL import Image

from imagegen_mcp import imaging, models
from imagegen_mcp.config import Config
from imagegen_mcp.devices import build_device_plan
from imagegen_mcp.sdserver import EngineError, EngineProgress, SdServer
from imagegen_mcp.server import create_server
from imagegen_mcp.service import ImageService, ServiceUnavailable


def test_upscaler_model_registry():
    f = models.upscaler_file(Config())
    assert f.dest == "upscalers/4xNomos2_otf_esrgan.safetensors" and f.optional and len(f.sha256) == 64
    alt = models.upscaler_file(Config.model_validate({"upscale": {"model": "RealESRGAN_x4plus"}}))
    assert alt.dest == "upscalers/RealESRGAN_x4plus.pth" and "BSD-3-Clause" in alt.license


class FakeEngine:
    """Stands in for SdServer.upscale (sd-cli -M upscale): a 4x nearest-neighbour enlargement."""

    def __init__(self):
        self.state = "ready"
        self.progress = EngineProgress()
        self.calls = []

    def vae_tiling_for(self, w, h):
        return None

    async def upscale(self, image_png, *, model, tile_size, timeout, **_):
        im = Image.open(io.BytesIO(image_png))
        self.calls.append({"size": im.size, "mode": im.mode, "model": model.name, "tile_size": tile_size})
        out = im.convert("RGB").resize((im.width * 4, im.height * 4), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return buf.getvalue()


def _service(tmp_path, **up):
    cfg = Config.model_validate({"outputs_dir": str(tmp_path / "out"), "models_dir": str(tmp_path / "m"),
                                 "upscale": up})
    svc = ImageService(cfg)
    svc.state = "ready"
    svc.engine = FakeEngine()
    svc.outputs.mkdir(parents=True)
    svc.upscaler_name = "4xNomos2_otf_esrgan"
    svc.upscaler_path = tmp_path / "m" / "upscalers" / "4xNomos2_otf_esrgan.safetensors"
    return svc


def _data_url(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def test_upscale_4x_and_2x_keep_alpha(tmp_path):
    svc = _service(tmp_path)
    src = Image.new("RGBA", (300, 200), (200, 30, 30, 255))
    alpha = np.zeros((200, 300), np.uint8)
    alpha[:, 150:] = 255
    src.putalpha(Image.fromarray(alpha))
    res = await svc.upscale(image=_data_url(src), scale=4)
    out = res.images[0].image
    assert out.size == (1200, 800) and out.mode == "RGBA"
    a = np.asarray(out.getchannel("A"))
    assert a[:, :500].max() < 30 and a[:, 700:].min() > 225  # transparency follows the input
    call = svc.engine.calls[-1]
    assert call["size"] == (300, 200) and call["mode"] == "RGB" and call["model"] == "4xNomos2_otf_esrgan.safetensors"
    assert res.info["scale"] == 4 and res.info["input"] == "300x200"
    res2 = await svc.upscale(image=_data_url(src.convert("RGB")), scale=2, output_format="jpeg")
    assert res2.images[0].image.size == (600, 400) and res2.images[0].mime == "image/jpeg"


async def test_large_inputs_are_reduced_to_fit_the_engine_limit(tmp_path):
    svc = _service(tmp_path)
    src = Image.new("RGB", (3000, 1000), (10, 120, 200))
    res = await svc.upscale(image=_data_url(src), scale=2)
    assert svc.engine.calls[-1]["size"] == (2048, 683)  # the 4x pass may not exceed 8192 px per side
    assert res.images[0].image.size == (6000, 2000)
    assert any("reduced" in n for n in res.notes)
    res4 = await svc.upscale(image=_data_url(src), scale=4)
    assert res4.images[0].image.size == (8192, 2730)
    assert any("8192" in n for n in res4.notes)
    with pytest.raises(ValueError):
        await svc.upscale(image=_data_url(src), scale=3)


async def test_upscale_unavailable_without_the_model(tmp_path):
    svc = _service(tmp_path)
    svc.upscaler_name = None
    svc.upscaler_error = "the upscaler model is still downloading; try again in a minute"
    with pytest.raises(ServiceUnavailable, match="still downloading"):
        await svc.upscale(image="x.png")


class FakeProc:
    def __init__(self, cmd, delay=0.0, rc=0, write=True):
        self.cmd, self.delay, self.returncode, self.write, self.killed = cmd, delay, None, write, False
        self._rc = rc

    async def communicate(self):
        import asyncio
        await asyncio.sleep(self.delay)
        if self.write:
            out = self.cmd[self.cmd.index("-o") + 1]
            Image.new("RGB", (8, 8)).save(out)
        self.returncode = self._rc
        return b"[INFO] upscaled\n[ERROR] boom" if self._rc else b"[INFO] upscaled", None

    def kill(self):
        self.killed = True

    async def wait(self):
        return -9


def _engine_with_cli(monkeypatch, **proc_kw):
    from imagegen_mcp import sdserver
    cfg = Config.model_validate({"cpu": {"threads": 4}})
    e = SdServer(cfg, build_device_plan(cfg), {})
    procs = []

    async def fake_exec(*cmd, **kw):
        procs.append(FakeProc(list(cmd), **proc_kw))
        return procs[-1]

    monkeypatch.setattr(sdserver.asyncio, "create_subprocess_exec", fake_exec)
    return e, procs


async def test_engine_upscale_runs_sd_cli(monkeypatch, tmp_path):
    e, procs = _engine_with_cli(monkeypatch)
    png = io.BytesIO()
    Image.new("RGB", (2, 2)).save(png, "PNG")
    out = await e.upscale(png.getvalue(), model=tmp_path / "4xNomos2_otf_esrgan.safetensors", tile_size=128, timeout=60)
    assert Image.open(io.BytesIO(out)).size == (8, 8)
    cmd = procs[0].cmd
    assert cmd[0].endswith("sd-cli") and cmd[cmd.index("-M") + 1] == "upscale"
    assert cmd[cmd.index("--upscale-model") + 1].endswith("4xNomos2_otf_esrgan.safetensors")
    assert cmd[cmd.index("--upscale-tile-size") + 1] == "128" and cmd[cmd.index("--backend") + 1] == "cuda0"
    assert cmd[cmd.index("-t") + 1] == "4"


async def test_engine_upscale_failure_and_timeout(monkeypatch, tmp_path):
    e, procs = _engine_with_cli(monkeypatch, rc=1, write=False)
    with pytest.raises(EngineError, match="exit code 1.*boom"):
        await e.upscale(b"png", model=tmp_path / "m.safetensors", tile_size=128, timeout=60)
    e, procs = _engine_with_cli(monkeypatch, delay=5)
    with pytest.raises(EngineError, match="did not finish in 1 s"):
        await e.upscale(b"png", model=tmp_path / "m.safetensors", tile_size=128, timeout=1)
    assert procs[0].killed and not e.queue.turns


async def test_upscale_waits_for_its_turn(monkeypatch, tmp_path):
    import asyncio
    e, procs = _engine_with_cli(monkeypatch)
    waits = []

    async def on_wait(msg):
        waits.append(msg)

    async def generation_holds_the_card():
        async with e.queue.turn("generate", 60.0):
            await asyncio.sleep(2.5)
            assert not procs  # nothing ran while the generation had the card

    task = asyncio.create_task(generation_holds_the_card())
    await asyncio.sleep(0)
    await e.upscale(b"png", model=tmp_path / "m.safetensors", tile_size=128, timeout=1, on_wait=on_wait)
    await task
    assert procs and waits[0].startswith("queued: position 1 in line, starts in about 60 s")


def test_sd_server_never_gets_an_upscaler(tmp_path):
    # An upscaler model in --hires-upscalers-dir slowed every denoising step by ~45% (measured).
    cfg = Config.model_validate({"models_dir": str(tmp_path)})
    (tmp_path / "upscalers").mkdir()
    e = SdServer(cfg, build_device_plan(cfg), {"diffusion": "d", "vae": "v", "text_encoder": "t", "vision": "x"})
    cmd = e.command()
    assert cmd[cmd.index("--hires-upscalers-dir") + 1] != str(tmp_path / "upscalers")
    assert "upscaler=" not in " ".join(cmd)


async def test_upscale_tool_follows_the_config(tmp_path):
    on = {t.name: t for t in await create_server(_service(tmp_path / "a")).list_tools()}
    off = {t.name for t in await create_server(_service(tmp_path / "b", enabled=False)).list_tools()}
    assert "upscale_image" in on and "upscale_image" not in off
    assert on["upscale_image"].input_schema["required"] == ["image"]
    assert on["upscale_image"].input_schema["properties"]["scale"]["enum"] == [2, 4]


async def test_generation_and_upscale_never_overlap(monkeypatch, tmp_path):
    """A generate and an upscale arriving at the same moment (two connections) never share the card."""
    import asyncio
    import time
    from imagegen_mcp import sdserver
    cfg = Config.model_validate({})
    e = SdServer(cfg, build_device_plan(cfg), {})
    e.proc = SimpleNamespace(returncode=None)

    async def running():
        return None

    e.ensure_running = running
    spans = {"gen": [], "up": []}
    state = {"polls": 0, "start": 0.0}
    png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(png, "PNG")
    b64 = base64.b64encode(png.getvalue()).decode()

    def handler(request):
        if request.url.path == "/sdcpp/v1/img_gen":
            state.update(polls=0, start=time.monotonic())
            return httpx.Response(200, json={"id": "g"})
        state["polls"] += 1
        if state["polls"] < 3:
            return httpx.Response(200, json={"status": "generating"})
        spans["gen"].append((state["start"], time.monotonic()))
        return httpx.Response(200, json={"status": "completed", "result": {"images": [{"b64_json": b64}]}})

    e._client = httpx.AsyncClient(base_url=e.base_url, transport=httpx.MockTransport(handler))

    class Proc:
        def __init__(self, cmd):
            self.cmd, self.returncode = cmd, None

        async def communicate(self):
            t0 = time.monotonic()
            await asyncio.sleep(0.6)
            Image.new("RGB", (8, 8)).save(self.cmd[self.cmd.index("-o") + 1])
            spans["up"].append((t0, time.monotonic()))
            self.returncode = 0
            return b"", None

    async def fake_exec(*cmd, **kw):
        return Proc(list(cmd))

    monkeypatch.setattr(sdserver.asyncio, "create_subprocess_exec", fake_exec)

    async def gen():
        return await e.generate({}, timeout=60)

    async def up():
        return await e.upscale(png.getvalue(), model=tmp_path / "m.safetensors", tile_size=128, timeout=60)

    for first, second in ((gen, up), (up, gen)):
        spans["gen"].clear()
        spans["up"].clear()
        await asyncio.gather(first(), second())
        (g0, g1), (u0, u1) = spans["gen"][0], spans["up"][0]
        assert g1 <= u0 or u1 <= g0, (first.__name__, spans)
    assert e._active == 0 and not e.queue.turns



def _jpeg_url(im, xmp=None):
    return "data:image/jpeg;base64," + base64.b64encode(imaging.encode(im, "jpeg", xmp=xmp)).decode()


def _png_url(im, xmp=None):
    return "data:image/png;base64," + base64.b64encode(imaging.encode(im, "png", xmp=xmp)).decode()


def _noise(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def _xmp_of(path):
    info = Image.open(path).info
    x = info.get("xmp") or info.get("XML:com.adobe.xmp") or b""
    return x if isinstance(x, str) else x.decode()


async def test_panorama_upscale_stays_a_360_panorama(tmp_path):
    from imagegen_mcp import panorama
    svc = _service(tmp_path)
    src = _noise(400, 200)
    res = await svc.upscale(image=_png_url(src, panorama.gpano_xmp(400, 200, heading=90, hfov=75)),
                            output_format="png")
    # The model saw 32 wrapped columns on each side (left pad = the right edge, right pad = the left edge).
    assert svc.engine.calls[-1]["size"] == (464, 200)
    saved = res.images[0]
    out = saved.image
    assert out.size == (1600, 800)
    # The padding is cropped off exactly: the fake model is a 4x nearest-neighbour enlargement.
    expected = np.asarray(src.resize((1600, 800), Image.Resampling.NEAREST))
    assert np.array_equal(np.asarray(out.convert("RGB")), expected)
    xmp = _xmp_of(saved.path)
    assert panorama.sphere_xmp(Image.open(saved.path)) is not None
    assert "<GPano:FullPanoWidthPixels>1600<" in xmp and "<GPano:CroppedAreaImageHeightPixels>800<" in xmp
    assert "<GPano:InitialViewHeadingDegrees>90<" in xmp and ">75.0<" in xmp  # the input's view is kept
    assert saved.view_url and saved.view_url.endswith("/view/" + saved.filename)
    assert res.info["panorama"].startswith("360")


async def test_panorama_upscale_defaults_to_jpeg_and_detects_jpeg_xmp(tmp_path):
    from imagegen_mcp import panorama
    svc = _service(tmp_path)
    res = await svc.upscale(image=_jpeg_url(_noise(256, 128), panorama.gpano_xmp(256, 128)), scale=2)
    saved = res.images[0]
    assert saved.mime == "image/jpeg" and saved.image.size == (512, 256) and saved.view_url
    assert svc.engine.calls[-1]["size"] == (320, 128)
    assert panorama.sphere_xmp(Image.open(saved.path)) is not None


async def test_untagged_2to1_image_is_plain_unless_forced(tmp_path):
    svc = _service(tmp_path)
    src = _noise(300, 150)
    plain = await svc.upscale(image=_png_url(src))
    assert svc.engine.calls[-1]["size"] == (300, 150)
    assert plain.images[0].view_url is None and plain.images[0].mime == "image/png"
    assert "panorama" not in plain.info
    forced = await svc.upscale(image=_png_url(src), as_panorama=True)
    assert svc.engine.calls[-1]["size"] == (364, 150)
    assert forced.images[0].view_url and forced.images[0].mime == "image/jpeg"
    assert "<GPano:FullPanoWidthPixels>1200<" in _xmp_of(forced.images[0].path)  # a fresh Photo Sphere tag
    with pytest.raises(ValueError, match="2:1"):
        await svc.upscale(image=_png_url(_noise(300, 200)), as_panorama=True)


async def test_panorama_detection_can_be_switched_off_and_ignores_partial_or_wrong_shape(tmp_path):
    from imagegen_mcp import panorama
    svc = _service(tmp_path)
    tagged = _png_url(_noise(256, 128), panorama.gpano_xmp(256, 128))
    off = await svc.upscale(image=tagged, as_panorama=False)
    assert svc.engine.calls[-1]["size"] == (256, 128) and off.images[0].view_url is None
    # A partial panorama (cropped area narrower than the full sphere) does not wrap around.
    partial = panorama.gpano_xmp(256, 128).replace(b"<GPano:FullPanoWidthPixels>256<",
                                                   b"<GPano:FullPanoWidthPixels>1024<")
    res = await svc.upscale(image=_png_url(_noise(256, 128), partial))
    assert svc.engine.calls[-1]["size"] == (256, 128) and res.images[0].view_url is None
    # Tagged as 360 but not 2:1: upscaled as a plain image, with a note.
    odd = await svc.upscale(image=_png_url(_noise(300, 200), panorama.gpano_xmp(300, 200)))
    assert odd.images[0].view_url is None and any("not 2:1" in n for n in odd.notes)


async def test_large_panorama_is_reduced_then_padded_to_a_full_8192_result(tmp_path):
    from imagegen_mcp import panorama
    svc = _service(tmp_path)
    res = await svc.upscale(image=_jpeg_url(_noise(2880, 1440), panorama.gpano_xmp(2880, 1440)))
    assert svc.engine.calls[-1]["size"] == (2112, 1024)  # 2048 wide + 32 wrapped columns per side
    assert res.images[0].image.size == (8192, 4096)
    assert any("reduced" in n for n in res.notes) and res.images[0].view_url
    # Sizes whose float scale factor rounds down still give exactly 8192 px on the long side.
    odd = await svc.upscale(image=_jpeg_url(_noise(2366, 1183), panorama.gpano_xmp(2366, 1183)))
    assert odd.images[0].image.size == (8192, 4096)


def test_other_gpano_spellings_are_detected_and_resized_in_place():
    from imagegen_mcp import panorama
    ns = panorama.GPANO_NS
    im = Image.new("RGB", (8, 4))
    # ExifTool shorthand: single quotes, another prefix, pose fields that must survive.
    im.info["xmp"] = (f"<x:xmpmeta><rdf:Description xmlns:gp='{ns}' gp:ProjectionType='equirectangular' "
                      f"gp:FullPanoWidthPixels='8' gp:CroppedAreaImageWidthPixels='8' gp:PosePitchDegrees='3.5' "
                      f"gp:InitialViewHeadingDegrees='nan'/></x:xmpmeta>").encode()
    xmp = panorama.sphere_xmp(im)
    assert xmp is not None
    out = panorama.resize_sphere_xmp(xmp, 32, 16).decode()
    assert "gp:FullPanoWidthPixels='32'" in out and "gp:CroppedAreaImageWidthPixels='32'" in out
    assert "gp:PosePitchDegrees='3.5'" in out and "gp:InitialViewHeadingDegrees='nan'" in out
    im.info["xmp"] = f'<rdf:Description xmlns:GPano="{ns}" GPano:ProjectionType="cylindrical"/>'.encode()
    assert panorama.sphere_xmp(im) is None
    assert panorama.sphere_xmp(Image.new("RGB", (8, 4))) is None
    # Element form: closing tags are left alone.
    el = panorama.resize_sphere_xmp(panorama.gpano_xmp(8, 4).decode(), 64, 32).decode()
    assert "<GPano:FullPanoWidthPixels>64</GPano:FullPanoWidthPixels>" in el
    assert el.count(">64<") == 2 and re.search(r"</GPano:\w+>\s*\d", el) is None
    ElementTree.fromstring(el.split("?>", 1)[1].rsplit("<?xpacket", 1)[0])  # still well-formed XML
    row = Image.fromarray(np.arange(6, dtype=np.uint8).reshape(1, 6), "L")
    assert np.asarray(panorama.wrap_pad(row, 2)).tolist() == [[4, 5, 0, 1, 2, 3, 4, 5, 0, 1]]


async def test_viewer_links_work_as_inputs_and_listing_flags_panoramas(tmp_path):
    from imagegen_mcp import panorama
    svc = _service(tmp_path)
    res = await svc.upscale(image=_jpeg_url(_noise(256, 128), panorama.gpano_xmp(256, 128)))
    rel = res.images[0].filename
    for ref in (f"http://localhost:5005/view/{rel}", f"/view/{rel}"):
        assert (await svc.loader.load(ref)).size == (1024, 512)
    plain = await svc.upscale(image=_png_url(_noise(64, 64)))
    listed = {e["file"]: e for e in svc.list_images()["outputs_and_uploads"]}
    assert listed[rel]["viewer_url"].endswith("/view/" + rel)
    assert "viewer_url" not in listed[plain.images[0].filename]


async def test_upscale_tool_has_a_panorama_switch(tmp_path):
    tools = {t.name: t for t in await create_server(_service(tmp_path)).list_tools()}
    props = tools["upscale_image"].input_schema["properties"]
    assert "panorama" in props and "360" in tools["upscale_image"].description
