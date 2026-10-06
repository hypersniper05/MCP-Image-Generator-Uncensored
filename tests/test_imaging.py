import base64
import io

import numpy as np
import pytest
from PIL import Image

from imagegen_mcp import imaging, panorama, transparency


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("kw,expected", [
    ({}, (1024, 1024)),
    ({"aspect_ratio": "16:9"}, (1376, 768)),
    ({"aspect_ratio": "9:16"}, (768, 1376)),
    ({"width": 1000, "height": 700}, (992, 704)),
    ({"width": 1920, "aspect_ratio": "16:9"}, (1920, 1088)),
    ({"size": "xl"}, (2048, 2048)),
    ({"size": "small", "aspect_ratio": "1:1"}, (512, 512)),
])
def test_resolve_size(kw, expected):
    w, h, _ = imaging.resolve_size(width=kw.get("width"), height=kw.get("height"),
                                   aspect_ratio=kw.get("aspect_ratio"), size=kw.get("size"),
                                   default=(1024, 1024), max_pixels=4_200_000)
    assert (w, h) == expected
    assert w % 32 == 0 and h % 32 == 0


def test_resolve_size_caps_pixels():
    w, h, notes = imaging.resolve_size(width=4096, height=4096, aspect_ratio=None, size=None,
                                       default=(1024, 1024), max_pixels=1_100_000)
    assert w * h <= 1_100_000 and notes


def test_reference_aspect_is_used():
    w, h, _ = imaging.resolve_size(width=None, height=None, aspect_ratio=None, size=None, default=(1024, 1024),
                                   max_pixels=4_200_000, reference=(1600, 900))
    assert abs(w / h - 16 / 9) < 0.05


def test_invalid_aspect():
    with pytest.raises(ValueError):
        imaging.parse_aspect("banana")


async def test_loader_accepts_data_url_and_base64(tmp_path):
    loader = imaging.ImageLoader(tmp_path)
    img = Image.new("RGBA", (40, 30), (255, 0, 0, 128))
    b64 = _png_b64(img)
    a = await loader.load("data:image/png;base64," + b64)
    b = await loader.load(b64)
    assert a.size == b.size == (40, 30) and a.mode == "RGBA"


async def test_loader_reads_own_outputs_but_not_other_files(tmp_path):
    out = tmp_path / "outputs"
    (out / "2026-01-01").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(out / "2026-01-01" / "x.png")
    loader = imaging.ImageLoader(out)
    assert (await loader.load("2026-01-01/x.png")).size == (8, 8)
    assert (await loader.load("/outputs/2026-01-01/x.png")).size == (8, 8)
    assert (await loader.load("http://localhost:5005/outputs/2026-01-01/x.png")).size == (8, 8)
    assert (await loader.load("http://10.0.0.5:5005/outputs/2026-01-01/x.png")).size == (8, 8)
    with pytest.raises(imaging.ImageInputError):
        await loader.load("../../etc/passwd")
    with pytest.raises(imaging.ImageInputError):
        await loader.load("/outputs/../secret.png")


async def test_loader_rejects_loopback_urls(tmp_path):
    loader = imaging.ImageLoader(tmp_path)
    with pytest.raises(imaging.ImageInputError):
        await loader.load("http://127.0.0.1:7861/sdcpp/v1/capabilities")


def test_encode_formats_and_xmp():
    img = Image.new("RGB", (512, 256), (10, 20, 30))
    xmp = panorama.gpano_xmp(512, 256)
    jpg = imaging.encode(img, "jpeg", xmp=xmp)
    png = imaging.encode(img, "png", xmp=xmp)
    for blob, fmt in ((jpg, "JPEG"), (png, "PNG")):
        assert b"<GPano:ProjectionType>equirectangular</GPano:ProjectionType>" in blob
        im = Image.open(io.BytesIO(blob))
        assert im.format == fmt and im.size == (512, 256)


def test_has_transparency_and_flatten():
    opaque = Image.new("RGBA", (4, 4), (1, 2, 3, 255))
    clear = Image.new("RGBA", (4, 4), (1, 2, 3, 0))
    assert not imaging.has_transparency(opaque)
    assert imaging.has_transparency(clear)
    assert imaging.flatten(clear).getpixel((0, 0)) == (255, 255, 255)


def test_panorama_roll_and_mask():
    a = np.zeros((64, 256, 3), dtype=np.uint8)
    a[:, :128] = 255
    img = Image.fromarray(a)
    rolled = panorama.roll_half(img)
    back = panorama.roll_half(rolled, back=True)
    assert np.array_equal(np.asarray(back), a)
    hard, soft = panorama.seam_mask((1024, 512), 0.125)
    h = np.asarray(hard)
    assert h[:, 512].min() == 255 and h[:, 0].max() == 0
    assert np.asarray(soft)[:, 512].min() == 255


def test_seam_score_detects_seam():
    x = np.linspace(0, 255, 512, dtype=np.float32)
    grad = Image.fromarray(np.tile(x, (64, 1)).astype(np.uint8)).convert("RGB")
    assert panorama.seam_score(grad) > 50
    wave = np.round(127 + 100 * np.sin(np.linspace(0, 2 * np.pi, 512, endpoint=False)))
    smooth = Image.fromarray(np.tile(wave, (64, 1)).astype(np.uint8)).convert("RGB")
    assert panorama.seam_score(smooth) < 2.0
    assert panorama.seam_score(panorama.crossfade_wrap(grad)) < panorama.seam_score(grad)


def test_prompts():
    assert transparency.rgba_prompt("a red apple").startswith("This is an RGBA image with transparency. a red apple.")
    assert transparency.rgba_prompt("a red apple").endswith("the background is transparent.")
    p = panorama.panorama_prompt("a beach", 2048, 1024)
    assert p.startswith("Generate a complete 360-degree equirectangular panorama. a beach")
    assert "2048 x 1024" in p and "left and right edges seamless" in p
    assert "<image1>" in panorama.panorama_from_image_prompt("a beach", 2048, 1024)


def test_normalize_mask():
    m = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    m.paste((255, 255, 255, 255), (0, 0, 5, 10))
    out = imaging.normalize_mask(m, (20, 20))
    a = np.asarray(out)
    assert out.mode == "RGB" and out.size == (20, 20)
    assert a[:, :8].min() == 255 and a[:, 12:].max() == 0


def test_apply_transparency_hybrid_uses_min_of_alphas():
    class FakeMatter:
        def matte(self, img):
            m = np.zeros((img.height, img.width), dtype=np.uint8)
            m[:, : img.width // 2] = 255
            return Image.fromarray(m, "L")

    a = np.full((10, 20, 4), 255, dtype=np.uint8)
    a[:, 15:, 3] = 0  # native alpha: right quarter transparent
    img = Image.fromarray(a, "RGBA")
    out, info = transparency.apply_transparency(img, "hybrid", FakeMatter())
    alpha = np.asarray(out.getchannel("A"))
    assert alpha[:, :10].min() == 255 and alpha[:, 10:].max() == 0
    assert info["final"]["transparent_pct"] == 50.0


def _big_png_b64() -> str:
    noise = (np.random.default_rng(1).random((300, 300, 3)) * 255).astype("uint8")
    return _png_b64(Image.fromarray(noise))  # ~360 KB of base64: longer than any file name limit


async def test_loader_large_data_url_and_raw_base64(tmp_path):
    loader = imaging.ImageLoader(tmp_path)
    b64 = _big_png_b64()
    assert len(b64) > 100_000
    assert (await loader.load("data:image/png;base64," + b64)).size == (300, 300)
    assert (await loader.load(b64)).size == (300, 300)
    assert (await loader.load("data:image/png;base64," + b64.replace("+", "-").replace("/", "_"))).size == (300, 300)


async def test_loader_errors_do_not_echo_payload(tmp_path):
    loader = imaging.ImageLoader(tmp_path)
    junk = "data:image/png;base64," + "A" * 200_000
    with pytest.raises(imaging.ImageInputError) as e:
        await loader.load(junk)
    assert len(str(e.value)) < 300
    with pytest.raises(imaging.ImageInputError) as e:
        await loader.load("x" * 5000)
    assert len(str(e.value)) < 300


async def test_loader_own_url_on_any_host(tmp_path):
    (tmp_path / "2026-01-01").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "2026-01-01" / "x.png")
    loader = imaging.ImageLoader(tmp_path)
    for host in ("192.0.2.10:5005", "198.51.100.7:5005", "localhost:5005"):
        assert (await loader.load(f"http://{host}/outputs/2026-01-01/x.png")).size == (8, 8)


def test_degrid_removes_nyquist_grid_and_keeps_alpha():
    import numpy as np

    h, w = 128, 160
    y, x = np.arange(h)[:, None], np.arange(w)[None, :]
    base = 100 + 50 * np.sin(x / 20.0) * np.cos(y / 15.0)
    grid = 6 * (-1.0) ** x + 6 * (-1.0) ** y + 6 * (-1.0) ** (x + y)
    rgb = np.repeat(np.clip(base + grid, 0, 255)[:, :, None], 3, 2).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    img.putalpha(Image.new("L", (w, h), 200))
    out = imaging.degrid(img)
    assert out.mode == "RGBA" and out.getchannel("A").getextrema() == (200, 200)
    err = np.asarray(out.convert("L"), dtype=np.float64) - np.rint(base)
    assert err.std() < 1.0
    tiny = Image.new("RGB", (40, 40))
    assert imaging.degrid(tiny) is tiny


async def test_fetch_checks_every_redirect_hop(monkeypatch, tmp_path):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.org":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:5005/secret.png"})
        return httpx.Response(200, content=b"never")

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(imaging, "_is_private_host", lambda host: host == "127.0.0.1")
    loader = imaging.ImageLoader(tmp_path)
    with pytest.raises(imaging.ImageInputError, match="loopback"):
        await loader._fetch("http://example.org/cat.png")


async def test_bare_result_name_resolves_to_its_date_folder(tmp_path):
    (tmp_path / "2026-09-23").mkdir()
    Image.new("RGB", (40, 30), (1, 2, 3)).save(tmp_path / "2026-09-23" / "image-021530-ab12cd34.png")
    loader = imaging.ImageLoader(tmp_path)
    img = await loader.load("image-021530-ab12cd34.png")
    assert img.size == (40, 30)
    with pytest.raises(imaging.ImageInputError):
        await loader.load("image-missing-00000000.png")


def test_tile_helpers():
    rng = np.random.default_rng(0)
    a = (rng.random((64, 96, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(a)
    back = panorama.roll_both(panorama.roll_both(img), back=True)
    assert np.array_equal(np.asarray(back), a)
    hard, soft = panorama.cross_mask((768, 512), 0.125)
    h = np.asarray(hard)
    # a horizontal band through the middle row, a vertical band through the middle column, corners kept
    assert h[256, 0] == 255 and h[0, 384] == 255 and h[0, 0] == 0 and h[511, 767] == 0
    smooth = Image.fromarray(np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (64, 1)))
    assert panorama.tile_seam_score(smooth) > 10  # a ramp does not wrap
    assert panorama.tile_seam_score(Image.new("L", (32, 32), 128)) < 1


async def test_tileable_and_transparent_are_exclusive(tmp_path):
    from imagegen_mcp.config import Config
    from imagegen_mcp.service import ImageService

    svc = ImageService(Config.model_validate({"outputs_dir": str(tmp_path), "models_dir": str(tmp_path / "m")}))
    with pytest.raises(ValueError, match="cannot be combined"):
        await svc.generate(prompt="x", tileable=True, transparent=True)


def test_degrid_wrap_keeps_a_tile_periodic():
    rng = np.random.default_rng(3)
    h, w = 96, 128
    y, x = np.arange(h)[:, None], np.arange(w)[None, :]
    base = 120 + 40 * np.sin(2 * np.pi * 3 * x / w) * np.cos(2 * np.pi * 2 * y / h) + rng.normal(0, 8, (h, w))
    a = np.clip(base + 6 * (-1.0) ** x + 6 * (-1.0) ** y, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.repeat(a[:, :, None], 3, 2), "RGB")
    shift = (37, 53)
    rolled = Image.fromarray(np.roll(np.asarray(img), shift, axis=(0, 1)))
    # with wrap, the filter is circular: shifting the tile first gives the same pixels, so the wrap edges are
    # filtered exactly like the inside of the image
    out = np.asarray(imaging.degrid(img, wrap=(True, True)), dtype=np.int16)
    out_r = np.asarray(imaging.degrid(rolled, wrap=(True, True)), dtype=np.int16)
    assert np.abs(np.roll(out, shift, axis=(0, 1)) - out_r).max() <= 1
    ref = np.asarray(imaging.degrid(img), dtype=np.int16)  # padded: the edges are filtered differently
    ref_r = np.asarray(imaging.degrid(rolled), dtype=np.int16)
    assert np.abs(np.roll(ref, shift, axis=(0, 1)) - ref_r).max() > 1


def test_wrap_report_finds_a_seam_on_one_axis():
    rng = np.random.default_rng(4)
    ramp = np.tile(np.linspace(40, 215, 128), (128, 1))  # left and right edges do not meet
    a = (ramp + 4 * rng.standard_normal((128, 128))).clip(0, 255).astype(np.uint8)
    r = panorama.wrap_report(Image.fromarray(a))
    assert r["x"]["pct"] == 100 and r["x"]["ratio"] > 10
    assert r["y"]["ratio"] < 2


async def test_tileable_uses_circular_and_falls_back_when_the_engine_ignores_it(tmp_path):
    from types import SimpleNamespace

    from imagegen_mcp.config import Config
    from imagegen_mcp.sdserver import EngineProgress
    from imagegen_mcp.service import ImageService

    svc = ImageService(Config.model_validate({"outputs_dir": str(tmp_path / "out"), "models_dir": str(tmp_path / "m")}))
    svc.state = "ready"
    svc.outputs.mkdir(parents=True)
    png = io.BytesIO()
    Image.new("RGB", (64, 64), (90, 120, 150)).save(png, "PNG")
    bodies = []

    async def generate(body, timeout, on_progress=None):
        bodies.append(body)
        if body.get("circular_x") and engine.patched:
            engine.circular_acks += 1  # what SdServer counts from the "Using circular padding" log line
        return [png.getvalue()]

    engine = SimpleNamespace(state="ready", progress=EngineProgress(), vae_tiling_for=lambda w, h: None,
                             supports_circular=True, circular_acks=0, patched=True, generate=generate)
    svc.engine = engine
    res = await svc.generate(prompt="bricks", width=512, height=512, tileable=True)
    assert len(bodies) == 1 and bodies[0]["circular_x"] is True and bodies[0]["circular_y"] is True
    assert res.info["tile_method"] == "circular" and "tile_wrap" in res.info
    bodies.clear()
    await svc.generate(prompt="a cat", width=512, height=512)
    assert len(bodies) == 1 and "circular_x" not in bodies[0]
    # a build that ignores the keys: no log line, so the same image goes through the repair pass
    engine.patched = False
    bodies.clear()
    res = await svc.generate(prompt="bricks", width=512, height=512, tileable=True)
    assert len(bodies) == 2 and "mask_image" in bodies[1] and res.info["tile_method"] == "repair"
    bodies.clear()
    await svc.generate(prompt="bricks", width=512, height=512, tileable=True)
    assert "circular_x" not in bodies[0]  # not tried again after that
