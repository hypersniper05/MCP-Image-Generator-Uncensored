import json
import os
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from imagegen_mcp import imaging, loras
from imagegen_mcp.config import Config
from imagegen_mcp.sdserver import EngineProgress
from imagegen_mcp.server import create_server
from imagegen_mcp.service import ImageService


def _gguf(path, names):
    """A minimal GGUF v3 file: one string KV, one array KV and ``names`` as 2-D F16 tensors."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(names), 2)
    out += s("general.architecture") + struct.pack("<I", 8) + s("qwen_image")
    out += s("some.list") + struct.pack("<IIQ", 9, 5, 3) + struct.pack("<3i", 1, 2, 3)
    for i, n in enumerate(names):
        out += s(n) + struct.pack("<I", 2) + struct.pack("<QQ", 4, 8) + struct.pack("<IQ", 1, i * 64)
    path.write_bytes(out)


def _safetensors(path, tensors, meta=None):
    header, blobs, off = ({"__metadata__": meta} if meta else {}), [], 0
    for k, arr in tensors.items():
        b = arr.astype(np.float16).tobytes()
        header[k] = {"dtype": "F16", "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    hj = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))


def _read_safetensors(path):
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    header.pop("__metadata__", None)
    data = raw[8 + n:]
    return {k: np.frombuffer(data[v["data_offsets"][0]:v["data_offsets"][1]], np.float16).reshape(v["shape"])
            for k, v in header.items()}


def test_gguf_names_and_mlp_layout(tmp_path):
    split, fused, other = tmp_path / "s.gguf", tmp_path / "f.gguf", tmp_path / "o.gguf"
    _gguf(split, ["img_in.weight", "transformer_blocks.0.img_mlp.gate_layer.weight",
                  "transformer_blocks.0.img_mlp.proj.weight"])
    _gguf(fused, ["model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight"])
    _gguf(other, ["x.weight"])
    assert loras.gguf_tensor_names(split)[0] == "img_in.weight"
    assert loras.mlp_layout(split) == loras.SPLIT
    assert loras.mlp_layout(fused) == loras.FUSED
    assert loras.mlp_layout(other) is None
    (tmp_path / "bad.gguf").write_bytes(b"nope")
    assert loras.mlp_layout(tmp_path / "bad.gguf") is None
    assert loras.mlp_layout(tmp_path / "model.safetensors") is None


def test_fused_mlp_lora_is_split_for_a_split_model(tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((4, 8)), rng.standard_normal((12, 4))  # r=4, hidden=8, 2 x intermediate=12
    q = rng.standard_normal((4, 8))
    base = "diffusion_model.transformer_blocks.0."
    src = root / "wm.safetensors"
    _safetensors(src, {base + "img_mlp.gate_up.lora_A.weight": a, base + "img_mlp.gate_up.lora_B.weight": b,
                       base + "attn.to_q.lora_A.weight": q}, meta={"name": "wm"})
    assert loras.prepare_lora(src, root, loras.FUSED) == "wm.safetensors"
    assert loras.prepare_lora(src, root, None) == "wm.safetensors"
    rel = loras.prepare_lora(src, root, loras.SPLIT)
    assert rel == "converted/wm.safetensors"
    t = _read_safetensors(root / rel)
    assert not any("gate_up" in k for k in t)
    half = b.astype(np.float16)
    np.testing.assert_array_equal(t[base + "img_mlp.gate_layer.lora_B.weight"], half[:6])
    np.testing.assert_array_equal(t[base + "img_mlp.proj.lora_B.weight"], half[6:])
    for part in ("gate_layer", "proj"):
        np.testing.assert_array_equal(t[base + f"img_mlp.{part}.lora_A.weight"], a.astype(np.float16))
    np.testing.assert_array_equal(t[base + "attn.to_q.lora_A.weight"], q.astype(np.float16))
    # reused while the original is unchanged, rebuilt when it is newer
    out = root / rel
    os.utime(src, (1, 1))
    os.utime(out, (5, 5))
    loras.prepare_lora(src, root, loras.SPLIT)
    assert out.stat().st_mtime == 5
    os.utime(out, (1, 1))
    os.utime(src, (2, 2))
    loras.prepare_lora(src, root, loras.SPLIT)
    assert out.stat().st_mtime > 2
    # nothing to convert: the original is used
    plain = root / "plain.safetensors"
    _safetensors(plain, {base + "attn.to_q.lora_A.weight": q})
    assert loras.prepare_lora(plain, root, loras.SPLIT) == "plain.safetensors"


def test_restore_unchanged_keeps_original_pixels_outside_the_edit():
    rng = np.random.default_rng(1)
    original = Image.fromarray(rng.integers(0, 255, (400, 600, 3), dtype=np.uint8))
    sent = original.resize((300, 200), Image.Resampling.LANCZOS)
    result = np.asarray(sent).copy()
    result[80:120, 130:170] = 255  # the "watermark" area the model repainted
    out, changed = imaging.restore_unchanged(original, sent, Image.fromarray(result))
    assert out.size == original.size
    assert 0.01 < changed < 0.2
    o, n = np.asarray(original), np.asarray(out)
    assert (o[:100, :150] == n[:100, :150]).all()  # far from the edit: the original, untouched
    assert (n[200:230, 300:330] > 240).all()  # inside the edit: the model's pixels
    everything = np.full_like(np.asarray(sent), 7)
    assert imaging.restore_unchanged(original, sent, Image.fromarray(everything))[0] is None


def _service(tmp_path, **wm):
    cfg = Config.model_validate({"outputs_dir": str(tmp_path / "out"), "models_dir": str(tmp_path / "m"),
                                 "watermark": wm})
    svc = ImageService(cfg)
    svc.state = "ready"
    svc.engine = SimpleNamespace(state="ready", progress=EngineProgress(), vae_tiling_for=lambda w, h: None)
    svc.outputs.mkdir(parents=True)
    return svc


async def test_remove_watermark_sends_the_lora_and_keeps_the_input_size(tmp_path):
    svc = _service(tmp_path)
    svc.watermark_lora = "converted/watermark_remover_v2_qwen.safetensors"
    src = Image.new("RGB", (1500, 1000), (40, 90, 160))
    sent = {}

    async def fake_run(body, progress):
        sent.update(body)
        import base64
        import io
        ref = Image.open(io.BytesIO(base64.b64decode(body["ref_images"][0]))).convert("RGB")
        arr = np.asarray(ref).copy()
        arr[:40, :40] = 250
        return Image.fromarray(arr)

    svc._run = fake_run
    res = await svc.remove_watermark(image="data:image/png;base64," + imaging.to_png_b64(src))
    assert sent["prompt"] == "Remove the watermarks"
    assert sent["lora"] == [{"path": "converted/watermark_remover_v2_qwen.safetensors", "multiplier": 1.0}]
    assert sent["width"] * sent["height"] <= 1.1e6 and sent["width"] % 32 == 0
    img = res.images[0].image
    assert img.size == (1500, 1000)
    px = np.asarray(img.convert("RGB"))
    assert (px[500:, 500:] == (40, 90, 160)).all()  # untouched area: exactly the input
    assert px[5, 5].min() > 200  # the repainted corner comes from the model
    assert res.info["processed_at"] == f"{sent['width']}x{sent['height']}" and "changed_percent" in res.info


async def test_remove_watermark_unavailable_without_the_lora(tmp_path):
    from imagegen_mcp.service import ServiceUnavailable
    svc = _service(tmp_path)
    svc.watermark_error = "the watermark-removal LoRA could not be downloaded; see the server log"
    with pytest.raises(ServiceUnavailable, match="could not be downloaded"):
        await svc.remove_watermark(image="x.png")


async def test_remove_watermark_tool_follows_the_config(tmp_path):
    on = create_server(_service(tmp_path / "a"))
    off = create_server(_service(tmp_path / "b", enabled=False))
    on_tools = {t.name: t for t in await on.list_tools()}
    assert "remove_watermark" in on_tools
    assert "remove_watermark" not in {t.name for t in await off.list_tools()}
    assert "remove_watermark" in on.instructions and "remove_watermark" not in off.instructions
    assert "watermarks, use remove_watermark" in on_tools["edit_image"].description


async def test_remove_watermark_keeps_the_color_profile(tmp_path):
    import io
    from PIL import ImageCms
    svc = _service(tmp_path)
    svc.watermark_lora = "wm.safetensors"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (10, 20, 30)).save(buf, "PNG", icc_profile=icc)

    async def fake_run(body, progress):
        return Image.new("RGB", (body["width"], body["height"]), (10, 20, 30))

    svc._run = fake_run
    import base64
    for fmt in ("png", "jpeg", "webp"):
        res = await svc.remove_watermark(image="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                                         output_format=fmt)
        assert Image.open(io.BytesIO(res.images[0].data)).info.get("icc_profile") == icc, fmt


async def test_description_follows_restore_unchanged(tmp_path):
    keep = {t.name: t for t in await create_server(_service(tmp_path / "a")).list_tools()}
    redraw = {t.name: t for t in await create_server(_service(tmp_path / "b", restore_unchanged=False)).list_tools()}
    assert "only the areas the model changed" in keep["remove_watermark"].description
    assert "whole image is redrawn" in redraw["remove_watermark"].description


def _store_files(tmp_path):
    from imagegen_mcp import models
    req = models.ModelFile("diffusion", "http://files.test/req", "d/req.bin", 4, None, "required", "MIT")
    opt = models.ModelFile("watermark_lora", "http://files.test/opt", "loras/opt.bin", 4, None, "optional", "MIT",
                           optional=True)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "req.bin").write_bytes(b"1234")
    return req, opt


async def test_missing_optional_file_never_blocks_startup(tmp_path, monkeypatch):
    from imagegen_mcp import models
    req, opt = _store_files(tmp_path)
    monkeypatch.setattr(models, "required_files", lambda cfg: [req, opt])
    monkeypatch.setattr(models.shutil, "disk_usage", lambda p: SimpleNamespace(free=1000))  # nearly full disk
    store = models.ModelStore(Config.model_validate({"models_dir": str(tmp_path)}))
    paths = await store.ensure()
    assert "diffusion" in paths and "watermark_lora" not in paths
    assert store.pending_optional == [opt] and store.status.state == "ready"
    assert await store.fetch_optional() == {}  # no room for it: skipped, not raised
    assert store.status.files["loras/opt.bin"] == "error" and store.pending_optional == []


@pytest.mark.parametrize("status,attempts", [(404, 1), (503, 3)])
async def test_optional_download_gives_up_quickly(tmp_path, monkeypatch, status, attempts):
    import httpx
    from imagegen_mcp import models
    req, opt = _store_files(tmp_path)
    monkeypatch.setattr(models, "required_files", lambda cfg: [req, opt])
    calls, sleeps = [], []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(status)

    real = httpx.AsyncClient
    monkeypatch.setattr(models.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))

    async def no_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(models.asyncio, "sleep", no_sleep)
    store = models.ModelStore(Config.model_validate({"models_dir": str(tmp_path)}))
    await store.ensure()
    assert await store.fetch_optional() == {}
    assert len(calls) == attempts and len(sleeps) == attempts - 1  # 404 is not retried; 503 only a few times
    assert store.status.files["loras/opt.bin"] == "error"


async def test_background_download_enables_the_tool(tmp_path):
    from imagegen_mcp import models
    svc = _service(tmp_path)
    root = tmp_path / "m" / "loras"
    root.mkdir(parents=True)
    lora = root / "watermark_remover_v2_qwen.safetensors"
    base = "diffusion_model.transformer_blocks.0.img_mlp.gate_up."
    _safetensors(lora, {base + "lora_A.weight": np.ones((2, 4)), base + "lora_B.weight": np.ones((6, 2))})
    gguf = tmp_path / "m" / "model.gguf"
    _gguf(gguf, ["transformer_blocks.0.img_mlp.proj.weight"])
    svc._paths = {"diffusion": gguf}
    svc.store.pending_optional = [models.watermark_lora_file()]
    svc._prepare_watermark_lora({})
    assert svc.watermark_lora is None and "still downloading" in svc.watermark_error

    async def fetched():
        svc.store.pending_optional = []
        return {"watermark_lora": lora}

    svc.store.fetch_optional = fetched
    await svc._fetch_optional()
    assert svc.watermark_lora == "converted/watermark_remover_v2_qwen.safetensors" and svc.watermark_error is None
    assert svc.status()["watermark_removal"] == "ready"


def test_token_budget_scales_references_evenly(tmp_path):
    svc = _service(tmp_path)
    refs = [Image.new("RGB", (1184, 896)) for _ in range(10)]  # 10 x 1.06 MP
    notes = []
    out, per_ref = svc._fit_token_budget(2048, 2048, refs, notes)
    total = 2048 * 2048 + sum(im.width * im.height for im in out)
    assert total / 256 <= 24576 and len({im.size for im in out}) == 1 and notes
    assert 10 * per_ref + 2048 * 2048 <= 24576 * 256  # the engine resizes references to per_ref
    small = [Image.new("RGB", (512, 512))]
    assert svc._fit_token_budget(1024, 1024, small, [])[0] is small  # within budget: untouched
    off = _service(tmp_path / "off")
    off.cfg.generation.token_budget = 0
    assert off._fit_token_budget(2048, 2048, refs, [])[0] is refs


async def test_edit_sends_the_budgeted_reference_size(tmp_path):
    svc = _service(tmp_path)
    sent = {}

    async def fake_run(body, progress):
        sent.update(body)
        return Image.new("RGB", (body["width"], body["height"]))

    async def fake_run_checked(body, progress):
        return await fake_run(body, progress), False

    svc._run = fake_run
    svc._run_checked = fake_run_checked
    imgs = ["data:image/png;base64," + imaging.to_png_b64(Image.new("RGB", (1184, 896))) for _ in range(10)]
    res = await svc.edit(prompt="collage", images=imgs, size="xl")
    per_ref = int(sent["ref_image_args"].split("=")[1])
    assert 10 * per_ref + sent["width"] * sent["height"] <= 24576 * 256
    assert any("memory budget" in n for n in res.notes)


def test_rgb_icc_only_for_rgb_profiles():
    from PIL import ImageCms
    rgb = Image.new("RGB", (4, 4))
    rgb.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    lab = Image.new("RGB", (4, 4))
    lab.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    assert imaging.rgb_icc(rgb) == rgb.info["icc_profile"]
    assert imaging.rgb_icc(lab) is None and imaging.rgb_icc(Image.new("RGB", (4, 4))) is None


async def test_large_results_are_read_back_from_disk(tmp_path):
    from imagegen_mcp import service as service_mod
    svc = _service(tmp_path)
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (1200, 1200, 3), dtype=np.uint8))
    saved = svc._save(noise, "png", "image")
    assert saved.size_bytes > service_mod.SAVED_CACHE_BYTES and saved.cached is None
    assert saved.data == saved.path.read_bytes() and saved.image.size == (1200, 1200)
    small = svc._save(Image.new("RGB", (64, 64)), "png", "image")
    assert small.cached is not None


async def test_own_files_are_not_limited_by_the_request_size(tmp_path):
    svc = _service(tmp_path)
    svc.loader.max_bytes = 100
    saved = svc._save(Image.new("RGB", (256, 256), (9, 9, 9)), "png", "image")
    assert saved.size_bytes > 100
    assert (await svc.loader.load(saved.filename)).size == (256, 256)
    with pytest.raises(imaging.ImageInputError, match="larger than"):
        await svc.loader.load("data:image/png;base64," + imaging.to_png_b64(Image.new("RGB", (256, 256))))
