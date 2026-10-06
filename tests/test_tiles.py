"""Seamless tiles stay seamless through edit_image and upscale_image (tagged with a small XMP packet)."""
import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from imagegen_mcp import imaging, panorama
from imagegen_mcp.config import Config
from imagegen_mcp.sdserver import EngineProgress
from imagegen_mcp.server import create_server
from imagegen_mcp.service import ImageService


def _noise(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def _url(im, fmt="png", xmp=None):
    mime = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}[fmt]
    return f"data:{mime};base64," + base64.b64encode(imaging.encode(im, fmt, xmp=xmp)).decode()


class Engine(SimpleNamespace):
    """sd-server stand-in: returns a plain image of the requested size and counts circular acknowledgements
    the way SdServer does from its "Using circular padding" log line (only when `patched`)."""

    def __init__(self, patched=True):
        super().__init__(state="ready", progress=EngineProgress(), supports_circular=True, circular_acks=0,
                         patched=patched, bodies=[], upscale_inputs=[])

    def vae_tiling_for(self, w, h):
        return None

    async def generate(self, body, timeout, on_progress=None):
        self.bodies.append(body)
        if (body.get("circular_x") or body.get("circular_y")) and self.patched:
            self.circular_acks += 1
        buf = io.BytesIO()
        _noise(body["width"], body["height"], 1).save(buf, "PNG")
        return [buf.getvalue()]

    async def upscale(self, image_png, *, model, tile_size, timeout, **_):
        im = Image.open(io.BytesIO(image_png))
        self.upscale_inputs.append(im.size)
        buf = io.BytesIO()
        im.convert("RGB").resize((im.width * 4, im.height * 4), Image.Resampling.NEAREST).save(buf, "PNG")
        return buf.getvalue()


def _service(tmp_path, patched=True):
    svc = ImageService(Config.model_validate({"outputs_dir": str(tmp_path / "out"), "models_dir": str(tmp_path / "m")}))
    svc.state = "ready"
    svc.outputs.mkdir(parents=True)
    svc.engine = Engine(patched)
    svc.upscaler_name = "4xNomos2_otf_esrgan"
    svc.upscaler_path = tmp_path / "m" / "upscalers" / "4xNomos2_otf_esrgan.safetensors"
    return svc


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp"])
def test_tile_tag_round_trips(fmt):
    tagged = Image.open(io.BytesIO(imaging.encode(_noise(32, 32), fmt, xmp=panorama.tile_xmp())))
    assert panorama.is_tile(tagged)
    assert not panorama.is_tile(Image.open(io.BytesIO(imaging.encode(_noise(32, 32), fmt))))
    pano = Image.open(io.BytesIO(imaging.encode(_noise(64, 32), fmt, xmp=panorama.gpano_xmp(64, 32))))
    assert not panorama.is_tile(pano) and panorama.sphere_xmp(pano) is not None


async def test_generated_tiles_are_tagged_and_their_edits_stay_seamless(tmp_path):
    svc = _service(tmp_path)
    tile = await svc.generate(prompt="mossy cobblestones", width=512, height=512, tileable=True)
    path = tile.images[0].path
    assert panorama.is_tile(Image.open(path))
    plain = await svc.generate(prompt="a cat", width=512, height=512)
    assert not panorama.is_tile(Image.open(plain.images[0].path))

    # A height map of the tile: wrap-around edit at the tile's own size (the edit default would be 1024x1024).
    svc.engine.bodies.clear()
    res = await svc.edit(prompt="Turn <image1> into a grayscale height map", images=[tile.images[0].filename])
    body = svc.engine.bodies[-1]
    assert body["circular_x"] is True and body["circular_y"] is True
    assert (body["width"], body["height"]) == (512, 512)
    assert res.info["tileable"].startswith("seamless") and "tile_wrap" in res.info
    assert panorama.is_tile(Image.open(res.images[0].path))  # so the next map in the chain stays seamless too

    # A normal image is edited normally, and tileable=false turns it off for a tile.
    svc.engine.bodies.clear()
    res = await svc.edit(prompt="make it blue", images=[plain.images[0].filename])
    assert "circular_x" not in svc.engine.bodies[-1] and (svc.engine.bodies[-1]["width"], svc.engine.bodies[-1]["height"]) == (1024, 1024)
    assert not panorama.is_tile(Image.open(res.images[0].path)) and "tileable" not in res.info
    await svc.edit(prompt="x", images=[tile.images[0].filename], tileable=False)
    assert "circular_x" not in svc.engine.bodies[-1]
    # tileable=true for an untagged texture from elsewhere; an explicit size wins over the input size.
    await svc.edit(prompt="x", images=[_url(_noise(256, 256))], tileable=True, width=768, height=768)
    body = svc.engine.bodies[-1]
    assert body["circular_x"] is True and (body["width"], body["height"]) == (768, 768)
    with pytest.raises(ValueError, match="cannot be combined"):
        await svc.edit(prompt="x", images=[tile.images[0].filename], transparent=True)


async def test_tile_edit_on_an_engine_without_circular_says_so(tmp_path):
    svc = _service(tmp_path, patched=False)
    res = await svc.edit(prompt="height map", images=[_url(_noise(256, 256), xmp=panorama.tile_xmp())])
    assert any("may not tile" in n for n in res.notes)
    assert not panorama.is_tile(Image.open(res.images[0].path)) and "tileable" not in res.info


async def test_tile_upscale_wraps_both_axes_and_stays_tagged(tmp_path):
    svc = _service(tmp_path)
    src = _noise(200, 120)
    res = await svc.upscale(image=_url(src, xmp=panorama.tile_xmp()))
    assert svc.engine.upscale_inputs[-1] == (264, 184)  # 32 wrapped columns and rows on each side
    out = res.images[0]
    assert out.image.size == (800, 480) and out.mime == "image/png" and out.view_url is None
    # The padding is cropped off exactly (the fake model is a 4x nearest-neighbour enlargement).
    assert np.array_equal(np.asarray(out.image.convert("RGB")),
                          np.asarray(src.resize((800, 480), Image.Resampling.NEAREST)))
    assert panorama.is_tile(Image.open(out.path)) and res.info["tileable"].startswith("seamless")
    # Untagged: plain upscale unless forced.
    await svc.upscale(image=_url(src))
    assert svc.engine.upscale_inputs[-1] == (200, 120)
    forced = await svc.upscale(image=_url(src), as_tileable=True)
    assert svc.engine.upscale_inputs[-1] == (264, 184) and panorama.is_tile(Image.open(forced.images[0].path))


def test_wrap_pad_both_axes():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = np.asarray(panorama.wrap_pad(Image.fromarray(a, "L"), 1, 1))
    assert out.tolist() == np.pad(a, 1, mode="wrap").tolist()
    rgb = panorama.wrap_pad(_noise(5, 4), 2)  # columns only (360 panorama)
    assert rgb.size == (9, 4)


async def test_edit_and_upscale_tools_have_a_tileable_switch(tmp_path):
    tools = {t.name: t for t in await create_server(_service(tmp_path)).list_tools()}
    assert "tileable" in tools["edit_image"].input_schema["properties"]
    assert "tileable" in tools["upscale_image"].input_schema["properties"]


async def test_older_untagged_tile_files_count_as_tiles(tmp_path):
    svc = _service(tmp_path)
    day = svc.outputs / "2026-10-05"
    day.mkdir()
    _noise(256, 256).save(day / "tile-125526-e926f9a0.png")  # made before tiles were tagged
    _noise(256, 256).save(day / "image-125526-aaaaaaaa.png")
    await svc.edit(prompt="height map", images=["2026-10-05/tile-125526-e926f9a0.png"])
    assert svc.engine.bodies[-1].get("circular_x") is True
    await svc.edit(prompt="height map", images=["2026-10-05/image-125526-aaaaaaaa.png"])
    assert "circular_x" not in svc.engine.bodies[-1]
    await svc.upscale(image="2026-10-05/tile-125526-e926f9a0.png", scale=2)
    assert svc.engine.upscale_inputs[-1] == (320, 320)
