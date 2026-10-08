"""Transparent output clean-up: exact alpha, clean edge colours, nothing left in fully transparent pixels."""
import io

import numpy as np
from PIL import Image

from imagegen_mcp import imaging, transparency

RED, PURPLE = np.array([220, 30, 30], np.float32), np.array([170, 60, 200], np.float32)


def _disc(size=160, r=50, edge=6, inner_alpha=254, bg=PURPLE):
    """A red disc on a purple fill with a soft edge whose colour is the red/purple mix, like a decoded RGBA image."""
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(yy - size / 2, xx - size / 2)
    a = np.clip((r + edge / 2 - d) / edge, 0, 1)
    rgb = a[..., None] * RED + (1 - a[..., None]) * bg
    alpha = np.where(a >= 1, inner_alpha, np.rint(a * 255)).astype(np.uint8)
    return np.dstack([np.clip(np.rint(rgb), 0, 255).astype(np.uint8), alpha]), a


def _toward_bg(rgb, a, bg=PURPLE):
    """Mean share of the background colour left in the soft edge pixels (0 = clean, 1 - alpha = untouched)."""
    band = (a > 0.05) & (a < 0.95)
    d = bg - RED
    return float((((rgb[band].astype(np.float32) - RED) @ d) / (d @ d)).mean())


def test_finish_cutout_snaps_alpha_cleans_edges_and_clears_hidden_pixels():
    src, a = _disc()
    before = _toward_bg(src[..., :3], a)
    out = np.asarray(transparency.finish_cutout(Image.fromarray(src, "RGBA")))
    alpha = out[..., 3]
    assert (alpha[a >= 1] == 255).all()  # the subject is fully opaque, not 254
    assert not ((alpha > 0) & (alpha <= 2)).any() and not ((alpha >= 250) & (alpha < 255)).any()
    assert out[alpha == 0, :3].max() == 0  # nothing of the old background is left in invisible pixels
    after = _toward_bg(out[..., :3], a)
    assert before > 0.3 and after < 0.15, (before, after)


def test_finish_cutout_without_decontamination_only_snaps_and_clears():
    src, a = _disc()
    out = np.asarray(transparency.finish_cutout(Image.fromarray(src, "RGBA"), decontaminate=False))
    soft = (out[..., 3] > 0) & (out[..., 3] < 255)
    assert (out[soft, :3] == src[soft, :3]).all() and out[out[..., 3] == 0, :3].max() == 0


def test_hidden_pixels_edge_mode_fills_with_the_nearest_visible_colour():
    src, a = _disc()
    out = np.asarray(transparency.finish_cutout(Image.fromarray(src, "RGBA"), hidden="edge"))
    hidden = out[..., 3] == 0
    near = hidden & (np.hypot(*np.mgrid[0:160, 0:160] - 80) < 60)
    assert np.abs(out[near, :3].astype(int) - RED.astype(int)).max() < 60  # reddish, not purple or black
    vis = out[..., 3] > 0
    ref = np.asarray(transparency.finish_cutout(Image.fromarray(src, "RGBA")))
    assert (out[vis] == ref[vis]).all()  # visible pixels are the same in both modes


def test_thin_lines_keep_their_colour():
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[..., :3] = PURPLE.astype(np.uint8)
    for x, w in ((10, 1), (20, 2), (30, 3), (40, 4)):  # opaque green lines, 1-4 px wide
        rgba[8:56, x:x + w] = (30, 200, 60, 255)
    out = np.asarray(transparency.finish_cutout(Image.fromarray(rgba, "RGBA")))
    lines = rgba[..., 3] == 255
    assert (out[lines, 3] == 255).all()
    assert np.abs(out[lines, :3].astype(int) - (30, 200, 60)).max() <= 40


def test_has_transparency_ignores_a_few_stray_pixels_in_outputs():
    img = np.full((512, 512, 4), 200, np.uint8)
    img[..., 3] = 255
    img[5, 5, 3] = 249  # what the RGBA VAE leaves in normal images
    stray = Image.fromarray(img, "RGBA")
    assert not imaging.has_transparency(stray, output=True)
    assert not imaging.has_transparency(stray)
    img[7, 7, 3] = 60  # an input with even one clearly transparent pixel keeps its alpha
    assert imaging.has_transparency(Image.fromarray(img, "RGBA"))
    assert not imaging.has_transparency(Image.fromarray(img, "RGBA"), output=True)
    img[:20, :20, 3] = 0  # 0.15% of the pixels: real transparency
    assert imaging.has_transparency(Image.fromarray(img, "RGBA"), output=True)


def test_transparent_webp_is_lossless():
    src, _ = _disc()
    clean = transparency.finish_cutout(Image.fromarray(src, "RGBA"))
    back = np.asarray(Image.open(io.BytesIO(imaging.encode(clean, "webp"))).convert("RGBA"))
    assert (back == np.asarray(clean)).all()
    opaque = Image.new("RGB", (64, 64), (10, 20, 30))
    assert imaging.encode(opaque, "webp")[:4] == b"RIFF"


def test_models_without_alpha_get_visible_colours_and_the_alpha_back():
    src, a = _disc()
    clean = transparency.finish_cutout(Image.fromarray(src, "RGBA"))  # hidden pixels are black now
    rgb = np.asarray(transparency.visible_rgb(clean))
    ring = (np.hypot(*np.mgrid[0:160, 0:160] - 80) > 54) & (np.hypot(*np.mgrid[0:160, 0:160] - 80) < 60)
    assert rgb[ring].mean(0)[0] > 150  # the model sees red next to the edge, not black
    back = np.asarray(transparency.with_alpha(Image.fromarray(rgb, "RGB"), clean.getchannel("A")))
    assert (back[..., 3] == np.asarray(clean)[..., 3]).all() and back[back[..., 3] == 0, :3].max() == 0


def test_alpha_stats_are_exact():
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[:5, :, 3] = 255
    rgba[5:7, :, 3] = 128
    st = transparency.alpha_stats(Image.fromarray(rgba, "RGBA"))
    assert st == {"transparent_pct": 30.0, "partial_pct": 20.0, "opaque_pct": 50.0}


def test_cutout_of_a_photo_clears_the_background(monkeypatch):
    class FakeMatter:
        def matte(self, img):
            _, a = _disc()
            return Image.fromarray(np.rint(a * 255).astype(np.uint8), "L")

    src, a = _disc(inner_alpha=255, bg=np.array([40, 90, 230], np.float32))
    photo = Image.fromarray(src[..., :3], "RGB")
    out = np.asarray(transparency.cutout(photo, FakeMatter()))
    assert out[out[..., 3] == 0, :3].max() == 0
    assert _toward_bg(out[..., :3], a, np.array([40, 90, 230], np.float32)) < 0.15


def test_matte_threads_follow_the_container_cpu_limit(monkeypatch):
    real = transparency.Path.read_text

    def fake_read(self, *args, **kw):
        if str(self).replace("\\", "/") == "/sys/fs/cgroup/cpu.max":
            return "400000 100000\n"
        return real(self, *args, **kw)

    monkeypatch.setattr(transparency.Path, "read_text", fake_read)
    assert transparency.available_cpus() <= 4
    assert transparency.default_threads() <= 2


class _FullMatter:
    """A matte that keeps everything (the model sees no background to remove)."""

    def matte(self, img):
        return Image.new("L", img.size, 255)


def test_cutout_of_an_already_transparent_image_keeps_its_clean_colours():
    """The input's own soft edges are straight colours, not a mix with whatever sits under its transparent pixels."""
    for under in ((0, 0, 0), (255, 255, 255), (255, 0, 255)):
        rgba, a = _disc(inner_alpha=255)
        rgba[..., :3] = (100, 150, 200)  # clean, straight colour everywhere visible
        rgba[a <= 0, :3] = under
        out = np.asarray(transparency.cutout(Image.fromarray(rgba, "RGBA"), _FullMatter()))
        vis = out[..., 3] > 0
        assert (out[vis, :3] == (100, 150, 200)).all(), under
    sprite = np.zeros((40, 40, 4), np.uint8)
    sprite[..., :3] = (255, 0, 255)
    sprite[8:32, 8:32] = (20, 20, 20, 255)  # dark 2-px outline ...
    sprite[10:30, 10:30] = (240, 200, 40, 255)  # ... around a yellow body
    out = np.asarray(transparency.cutout(Image.fromarray(sprite, "RGBA"), _FullMatter()))
    assert (out[8:32, 8, :3] == 20).all() and (out[15, 15, :3] == (240, 200, 40)).all()


def test_an_opaque_rgba_input_cuts_out_like_the_same_rgb_image():
    class DiscMatter:
        def matte(self, img):
            _, a = _disc()
            return Image.fromarray(np.rint(a * 255).astype(np.uint8), "L")

    src, _ = _disc(inner_alpha=255, bg=np.array([40, 90, 230], np.float32))
    rgb = Image.fromarray(src[..., :3], "RGB")
    via_rgb = np.asarray(transparency.cutout(rgb, DiscMatter()))
    via_rgba = np.asarray(transparency.cutout(rgb.convert("RGBA"), DiscMatter()))
    assert (via_rgb == via_rgba).all()


def test_big_images_give_the_same_result_tiled_or_whole(monkeypatch):
    import functools
    h, w = 120, 2600  # a soft band wider than one 2048 px window
    xx = np.arange(w)[None, :].repeat(h, 0)
    alpha = np.clip(np.rint(255 * (0.5 + 0.45 * np.sin(xx / 37.0))), 0, 255).astype(np.uint8)
    alpha[:, :40] = 0
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = np.clip(xx * 255 // w, 0, 255)
    rgba[..., 1], rgba[..., 2], rgba[..., 3] = 120, 200, alpha
    img = Image.fromarray(rgba, "RGBA")
    tiled = np.asarray(transparency.finish_cutout(img))
    monkeypatch.setattr(transparency, "_windows", functools.partial(transparency._windows, tile=100000))
    whole = np.asarray(transparency.finish_cutout(img))
    assert np.array_equal(tiled, whole)


def test_keep_mode_passes_colours_and_alpha_through():
    rgb = Image.new("RGB", (8, 8), (9, 99, 199))
    alpha = Image.fromarray(np.array([[0, 1, 251, 255] * 2] * 8, np.uint8), "L")
    out = np.asarray(transparency.with_alpha(rgb, alpha, hidden="keep"))
    assert (out[..., :3] == (9, 99, 199)).all() and (out[..., 3] == np.asarray(alpha)).all()


def test_small_semi_transparent_areas_of_inputs_count():
    img = np.full((1024, 1024, 4), 255, np.uint8)
    img[100:130, 100:130, 3] = 200  # a 30x30 glass badge: 0.09% of the pixels
    assert imaging.has_transparency(Image.fromarray(img, "RGBA"))
    assert not imaging.has_transparency(Image.fromarray(img, "RGBA"), output=True)


def test_previews_of_transparent_images_stay_small():
    src, _ = _disc(size=512, r=180)
    noise = (np.random.default_rng(3).random((512, 512, 3)) * 255).astype(np.uint8)
    src[..., :3] = noise
    img = transparency.finish_cutout(Image.fromarray(src, "RGBA"))
    data, mime = imaging.preview(img, 512, keep_alpha=True)
    assert mime == "image/webp" and len(data) < len(imaging.encode(img, "webp")) / 2


async def test_remove_background_keeps_the_colour_profile(tmp_path):
    import base64
    from types import SimpleNamespace
    from PIL import ImageCms
    from imagegen_mcp.config import Config
    from imagegen_mcp.sdserver import EngineProgress
    from imagegen_mcp.service import ImageService

    cfg = Config.model_validate({"outputs_dir": str(tmp_path / "out"), "models_dir": str(tmp_path / "m")})
    svc = ImageService(cfg)
    svc.state = "ready"
    svc.engine = SimpleNamespace(state="ready", progress=EngineProgress(), vae_tiling_for=lambda w, h: None)
    svc.outputs.mkdir(parents=True)

    class DiscMatter:
        def matte(self, img):
            _, a = _disc(size=img.width)
            return Image.fromarray(np.rint(a * 255).astype(np.uint8), "L")

    svc.matter = DiscMatter()
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    src, _ = _disc(inner_alpha=255)
    buf = io.BytesIO()
    Image.fromarray(src[..., :3], "RGB").save(buf, "PNG", icc_profile=icc)
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    for fmt in ("png", "webp"):
        res = await svc.remove_background(image=url, output_format=fmt)
        assert Image.open(io.BytesIO(res.images[0].data)).info.get("icc_profile") == icc, fmt
