"""Transparent-background support.

Qwen-Image-2.1 has an RGBA VAE: prompts written with the official template
produce a real alpha channel. In practice (stable-diffusion.cpp issue #2024) the
native alpha often keeps opaque white patches around the subject, so by
default the native alpha is intersected with a BiRefNet matte ("hybrid").

Every transparent result is then cleaned up (``finish_cutout``): the alpha is
snapped to exact 0/255 where it is nearly so, the true colour of the soft edge
pixels is estimated so the old background (the decoder's purple fill, or the
photo behind a cut-out) no longer tints the edge, and fully transparent pixels
are cleared so nothing of the source image is left in them.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("imagegen.transparency")

RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."

SNAP_HI = 250  # alpha at or above this becomes fully opaque
SNAP_LO = 2    # alpha at or below this becomes fully transparent


def rgba_prompt(prompt: str) -> str:
    p = prompt.strip()
    if p.startswith("This is an RGBA image"):
        return p
    if p and p[-1] not in ".!?":
        p += "."
    return f"{RGBA_PREFIX}{p}{RGBA_SUFFIX}"


def alpha_stats(img: Image.Image) -> dict:
    """Exact shares of fully transparent (alpha 0), partly transparent and fully opaque (alpha 255) pixels."""
    if img.mode != "RGBA":
        return {"transparent_pct": 0.0, "partial_pct": 0.0, "opaque_pct": 100.0}
    a = np.asarray(img.getchannel("A"))
    clear, solid = float((a == 0).mean() * 100), float((a == 255).mean() * 100)
    return {"transparent_pct": round(clear, 2), "partial_pct": round(100 - clear - solid, 2),
            "opaque_pct": round(solid, 2)}


class Matter:
    """Lazy ONNX Runtime session for a BiRefNet / IS-Net background-removal model."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: Path, model_name: str, device: int | str = "cpu", threads: int = 0):
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.threads = threads
        self._session = None
        self._lock = threading.Lock()
        self.provider = "not loaded"

    @property
    def input_size(self) -> int:
        return 1024

    def _load(self):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if self.threads:
            opts.intra_op_num_threads = self.threads
        # The CPU arena keeps the largest buffers it ever allocated (several GB after a 2048 px matte) for the life
        # of the process; without it the memory is returned after each image, at a small speed cost.
        opts.enable_cpu_mem_arena = False
        providers: list = ["CPUExecutionProvider"]
        if self.device != "cpu":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                # The process sees only this GPU (see __main__), so it is device 0.
                providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            else:
                log.warning("background_removal is set to GPU %s but this image has no CUDA ONNX Runtime; "
                            "using the CPU", self.device)
        sess = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)
        self.provider = sess.get_providers()[0]
        log.info("loaded %s background-removal model on %s", self.model_name, self.provider)
        return sess

    def matte(self, img: Image.Image) -> Image.Image:
        """Return an L-mode alpha matte (255 = foreground) for ``img``."""
        with self._lock:
            if self._session is None:
                self._session = self._load()
            sess = self._session
        rgb = img.convert("RGB")
        size = self.input_size
        x = np.asarray(rgb.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
        x = x / max(float(x.max()), 1e-6)
        if self.model_name.startswith("isnet"):
            x = x - 0.5  # IS-Net: mean 0.5, std 1.0
        else:
            x = (x - self._MEAN) / self._STD
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        name = sess.get_inputs()[0].name
        out = sess.run(None, {name: x})[0]
        pred = out[0, 0] if out.ndim == 4 else out[0]
        if self.model_name.startswith("birefnet"):
            pred = 1.0 / (1.0 + np.exp(-pred))
        lo, hi = float(pred.min()), float(pred.max())
        pred = (pred - lo) / (hi - lo) if hi > lo else np.zeros_like(pred)
        # rounded, not truncated: truncation left the whole subject at 254
        mask = Image.fromarray(np.rint(pred * 255).astype(np.uint8), "L")
        return mask.resize(img.size, Image.Resampling.LANCZOS)


def composite_on(img: Image.Image, color=(255, 255, 255)) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, color)
    bg.paste(img, mask=img.getchannel("A"))
    return bg


def apply_transparency(img: Image.Image, method: str, matter: Matter | None, *, decontaminate: bool = True,
                       hidden: str = "black") -> tuple[Image.Image, dict]:
    """Post-process a generated image into a clean RGBA cut-out."""
    info: dict = {"method": method, "native": alpha_stats(img)}
    if method == "native" or matter is None:
        if method != "native":
            info["note"] = "matte model unavailable, kept the model's native alpha"
        out = img.convert("RGBA")
    else:
        # Matte on the image as the viewer would see it over white.
        matte = np.asarray(matter.matte(composite_on(img)), dtype=np.uint8)
        rgba = np.array(img.convert("RGBA"))
        if method == "hybrid" and (rgba[..., 3] < 16).mean() >= 0.05:
            rgba[..., 3] = np.minimum(rgba[..., 3], matte)
        else:
            # Native alpha is unusable (almost nothing transparent): rely on the matte.
            rgba[..., 3] = matte
            if method == "hybrid":
                info["note"] = "native alpha was nearly opaque, used the matte only"
        out = Image.fromarray(rgba, "RGBA")
    out = finish_cutout(out, decontaminate=decontaminate, hidden=hidden)
    info.update(final=alpha_stats(out), decontaminated=decontaminate, hidden_pixels=hidden)
    return out, info


def cutout(img: Image.Image, matter: Matter, *, decontaminate: bool = True, hidden: str = "black") -> Image.Image:
    """Plain background removal for an arbitrary input image."""
    matte = np.asarray(matter.matte(composite_on(img) if img.mode == "RGBA" else img).convert("L"))
    if img.mode != "RGBA":
        out = img.convert("RGBA")
        out.putalpha(Image.fromarray(matte, "L"))
        return finish_cutout(out, decontaminate=decontaminate, hidden=hidden)
    src = np.asarray(img)
    a_in = src[..., 3]
    rgba = src.copy()
    rgba[..., :3] = fill_hidden(src)  # the colour under the input's transparent pixels is not a background
    rgba[..., 3] = np.minimum(a_in, matte)
    keep = ~_grow(matte < a_in, 1)  # where the alpha is the input's own, its colours are already clean
    return finish_cutout(Image.fromarray(rgba, "RGBA"), decontaminate=decontaminate, hidden=hidden, keep=keep)


def _grow(mask: np.ndarray, r: int) -> np.ndarray:
    """Maximum over a (2r+1) x (2r+1) window."""
    out = mask
    for _ in range(r):
        m = out.copy()
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            s = _shift(out, dy, 0) if dy else out
            m |= _shift(s, 0, dx) if dx else s
        out = m
    return out


def available_cpus() -> int:
    """CPUs this process may use: the container's CPU limit (cgroup quota) when there is one, else the CPU count."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    quota = None
    try:
        q, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]  # cgroup v2: "400000 100000"
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:  # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            if q > 0:
                quota = q / int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        except (OSError, ValueError):
            pass
    return max(1, min(n, int(quota)) if quota else n)


def default_threads() -> int:
    return max(1, min(8, available_cpus() // 2))


# ---------------------------------------------------------------------------------------------------------- clean-up
# The edge colour estimate ports the multi-level foreground estimation of Germer et al. 2020 ("Fast Multi-Level
# Foreground Estimation"), as in pymatting's estimate_foreground_ml (MIT licence, Copyright (c) 2020 Thomas Germer),
# the method rembg uses for its colour decontamination. Here it is plain numpy: the per-pixel 2x2 solve is the same,
# the in-place Gauss-Seidel sweep is a red-black (checkerboard) sweep, and the fine levels solve only the soft edge
# pixels. Big images are solved in 2048 px windows after one shared coarse solve, so tiling does not change the
# result. Cost: about 0.4 s per megapixel of image around the edges.

def snap_alpha(alpha: np.ndarray) -> np.ndarray:
    out = alpha.copy()
    out[out >= SNAP_HI] = 255
    out[out <= SNAP_LO] = 0
    return out


def _box(x: np.ndarray, k: int) -> np.ndarray:
    """Mean over a k x k window, normalised by the in-image count at the borders (cumulative sums, O(N))."""
    if k <= 1:
        return x.astype(np.float32, copy=True)
    r0, r1 = k // 2, k - 1 - k // 2
    a = x[..., None] if x.ndim == 2 else x

    def axis_mean(arr, n, axis):
        c = np.cumsum(arr, axis=axis, dtype=np.float64)
        pad = list(arr.shape)
        pad[axis] = 1
        c = np.concatenate([np.zeros(pad), c], axis=axis)
        idx = np.arange(n)
        hi, lo = np.minimum(idx + r1 + 1, n), np.maximum(idx - r0, 0)
        shape = [1] * arr.ndim
        shape[axis] = n
        return (np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)) / (hi - lo).reshape(shape)

    out = axis_mean(axis_mean(a, a.shape[0], 0), a.shape[1], 1).astype(np.float32)
    return out[..., 0] if x.ndim == 2 else out


def _shift(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """The neighbour at (dy, dx), clamped at the image border."""
    if dy == -1:
        return np.concatenate([x[:1], x[:-1]], 0)
    if dy == 1:
        return np.concatenate([x[1:], x[-1:]], 0)
    if dx == -1:
        return np.concatenate([x[:, :1], x[:, :-1]], 1)
    return np.concatenate([x[:, 1:], x[:, -1:]], 1)


def _erode(a: np.ndarray, r: int) -> np.ndarray:
    """Minimum over a (2r+1) x (2r+1) window."""
    out = a
    for _ in range(r):
        m = out.copy()
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            s = _shift(out, dy, 0) if dy else out
            m = np.minimum(m, _shift(s, 0, dx) if dx else s)
        out = m
    return out


def _resize_nearest(src: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = np.clip(np.arange(h) * src.shape[0] // h, 0, src.shape[0] - 1)
    xs = np.clip(np.arange(w) * src.shape[1] // w, 0, src.shape[1] - 1)
    return src[ys][:, xs]


_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _fb_dense(image: np.ndarray, alpha: np.ndarray, reg=1e-5, small_iters=10, big_iters=2, small_size=32):
    """Foreground and background colours (F, B) for every pixel, coarse to fine."""
    h0, w0, depth = image.shape
    fm, bm = alpha > 0.9, alpha < 0.1
    F_prev = np.zeros((1, 1, depth), np.float32) + (image[fm].mean(0) if fm.any() else 0)
    B_prev = np.zeros((1, 1, depth), np.float32) + (image[bm].mean(0) if bm.any() else 0)
    n_levels = max(1, int(np.ceil(np.log2(max(w0, h0)))))
    for level in range(n_levels + 1):
        w, h = round(w0 ** (level / n_levels)), round(h0 ** (level / n_levels))
        img, a = _resize_nearest(image, h, w), _resize_nearest(alpha, h, w)
        F, B = _resize_nearest(F_prev, h, w).copy(), _resize_nearest(B_prev, h, w).copy()
        a1 = 1.0 - a
        das = [reg + np.abs(a - _shift(a, dy, dx)) for dy, dx in _DIRS]
        dsum = sum(das)
        a00, a11, a01 = a * a + dsum, a1 * a1 + dsum, a * a1
        inv = 1.0 / (a00 * a11 - a01 * a01)
        yy, xx = np.mgrid[0:h, 0:w]
        red = (yy + xx) % 2 == 0
        for _ in range(small_iters if w <= small_size and h <= small_size else big_iters):
            for m in (red, ~red):
                b0, b1 = a[..., None] * img, a1[..., None] * img
                for (dy, dx), da in zip(_DIRS, das):
                    b0 = b0 + da[..., None] * _shift(F, dy, dx)
                    b1 = b1 + da[..., None] * _shift(B, dy, dx)
                F[m] = np.clip((inv * a11)[..., None] * b0 - (inv * a01)[..., None] * b1, 0, 1)[m]
                B[m] = np.clip(-(inv * a01)[..., None] * b0 + (inv * a00)[..., None] * b1, 0, 1)[m]
        F_prev, B_prev = F, B
    return F_prev, B_prev


def _fb_tiled(image: np.ndarray, alpha: np.ndarray, windows: list, reg=1e-5, big_iters=2,
             dense_max_px=1_500_000):
    """Foreground colour for each window's inner box, as one solve over the whole frame would give it: the coarse
    levels (up to ~1.5 MP) are solved once for the frame, and only the fine levels run per window, in the frame's
    level geometry, updating only the soft pixels (opaque pixels keep F = I and clear ones B = I there, which is
    what a dense solve converges to anyway). image: HxWx3 uint8, alpha: HxW uint8, windows: _windows() boxes in
    frame coordinates. Yields (inner box, F of the inner box)."""
    H, W, depth = image.shape
    n = max(1, int(np.ceil(np.log2(max(W, H)))))
    last_dense = n - 1
    while last_dense > 1 and (H * W) ** (last_dense / n) > dense_max_px:
        last_dense -= 1
    hd, wd = round(H ** (last_dense / n)), round(W ** (last_dense / n))
    Fd, Bd = _fb_dense(_resize_nearest(image, hd, wd).astype(np.float32) / 255,
                       _resize_nearest(alpha, hd, wd).astype(np.float32) / 255, reg)
    for (iy0, iy1, ix0, ix1), (y0, y1, x0, x1) in windows:
        F_prev, B_prev, pr0, pc0, ph, pw = Fd, Bd, 0, 0, hd, wd
        for level in range(last_dense + 1, n + 1):
            h, w = round(H ** (level / n)), round(W ** (level / n))
            r, c = np.arange(h), np.arange(w)
            r = r[(r * H // h >= y0) & (r * H // h < y1)]  # this window's rows and columns at this level
            c = c[(c * W // w >= x0) & (c * W // w < x1)]
            img = image[(r * H // h)[:, None], (c * W // w)[None, :]].astype(np.float32) / 255
            a = alpha[(r * H // h)[:, None], (c * W // w)[None, :]].astype(np.float32) / 255
            sr = np.clip(r * ph // h - pr0, 0, F_prev.shape[0] - 1)
            sc = np.clip(c * pw // w - pc0, 0, F_prev.shape[1] - 1)
            F, B = F_prev[sr][:, sc].copy(), B_prev[sr][:, sc].copy()
            hh, ww = len(r), len(c)
            F[a >= 1], B[a <= 0] = img[a >= 1], img[a <= 0]
            ys, xs = np.nonzero((a > 0) & (a < 1))
            if ys.size:
                Ff, Bf, af = F.reshape(-1, depth), B.reshape(-1, depth), a.reshape(-1)
                idx = ys * ww + xs
                nb = [np.clip(ys + dy, 0, hh - 1) * ww + np.clip(xs + dx, 0, ww - 1) for dy, dx in _DIRS]
                a0 = af[idx]
                a1 = 1 - a0
                das = [reg + np.abs(a0 - af[k]) for k in nb]
                dsum = sum(das)
                a00, a11, a01 = a0 * a0 + dsum, a1 * a1 + dsum, a0 * a1
                inv = 1.0 / (a00 * a11 - a01 * a01)
                red = (ys + r[0] + xs + c[0]) % 2 == 0  # the frame's checkerboard, so windows agree at their borders
                i0 = img.reshape(-1, depth)[idx]
                for _ in range(big_iters):
                    for m in (red, ~red):
                        b0, b1 = a0[:, None] * i0, a1[:, None] * i0
                        for k, da in zip(nb, das):
                            b0 = b0 + da[:, None] * Ff[k]
                            b1 = b1 + da[:, None] * Bf[k]
                        Ff[idx[m]] = np.clip((inv * a11)[:, None] * b0 - (inv * a01)[:, None] * b1, 0, 1)[m]
                        Bf[idx[m]] = np.clip(-(inv * a01)[:, None] * b0 + (inv * a00)[:, None] * b1, 0, 1)[m]
            F_prev, B_prev, pr0, pc0, ph, pw = F, B, int(r[0]), int(c[0]), h, w
        yield (iy0, iy1, ix0, ix1), F_prev[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]


def _windows(mask: np.ndarray, pad: int, tile: int = 2048):
    """Cover the bounding box of ``mask`` with tiles of at most ``tile`` px (inner window) plus ``pad`` px of
    context (outer window), so big images are processed in pieces of bounded memory. Up to 2048 px it is one
    window: the bounding box plus the padding."""
    ys, xs = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    if not ys.size:
        return
    h, w = mask.shape
    y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
    for ty in range(y0, y1, tile):
        for tx in range(x0, x1, tile):
            iy1, ix1 = min(ty + tile, y1), min(tx + tile, x1)
            if mask[ty:iy1, tx:ix1].any():
                yield (ty, iy1, tx, ix1), (max(0, ty - pad), min(h, iy1 + pad), max(0, tx - pad), min(w, ix1 + pad))


def _local_key(rgb: np.ndarray, hidden: np.ndarray, k: int = 31) -> np.ndarray:
    """Local mean colour of the fully transparent pixels: the decoder's fill colour, or the photo's background."""
    hid = hidden.astype(np.float32)
    gk = (rgb * hid[..., None]).sum((0, 1)) / max(float(hid.sum()), 1.0)
    w = _box(hid, k)[..., None]
    return (_box(rgb * hid[..., None], k) + 1e-3 * gk) / (w + 1e-3)


def _despill_rim(out: np.ndarray, src_rgb: np.ndarray, ring: int = 1, k_key: int = 31, k_in: int = 5,
                 min_sep: float = 30.0) -> None:
    """In the soft band and a ``ring``-px rim of opaque pixels, remove the part of each colour that points from the
    subject's nearby inside colour toward the local background colour. Only that direction is touched, so details
    of other colours keep their colour. Works in place on ``out``."""
    alpha = out[..., 3]
    hidden = alpha == 0
    rim = (alpha > 0) & (_erode(alpha, ring) < 255)
    if not hidden.any() or not rim.any():
        return
    for (iy0, iy1, ix0, ix1), (y0, y1, x0, x1) in _windows(rim, k_key):
        x = out[y0:y1, x0:x1, :3].astype(np.float32)
        s = src_rgb[y0:y1, x0:x1].astype(np.float32)
        key = _local_key(s, hidden[y0:y1, x0:x1], k_key)
        deep = (_erode(alpha[y0:y1, x0:x1], ring + 1) == 255).astype(np.float32)
        wd = _box(deep, k_in)
        inside = _box(s * deep[..., None], k_in) / (wd[..., None] + 1e-6)
        d = key - inside
        dn2 = (d * d).sum(-1)
        t = np.clip(((x - inside) * d).sum(-1) / (dn2 + 1e-6), 0, 1)
        sel = rim[y0:y1, x0:x1] & (wd > 0) & (dn2 > min_sep ** 2)
        inner = np.zeros_like(sel)
        inner[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = True
        sel &= inner
        sub = out[y0:y1, x0:x1]
        sub[sel, :3] = np.clip(np.rint((x - t[..., None] * d)[sel]), 0, 255).astype(np.uint8)


def fill_hidden(rgba: np.ndarray, band: int = 512) -> np.ndarray:
    """RGB with every fully transparent pixel set to the colour of the nearest visible pixels (push-pull pyramid);
    visible pixels keep their colour. Level 0 is never held as float (there a visible pixel keeps its colour and a
    hidden one takes level 1's colour), and the first reduction runs in row bands, so memory stays small."""
    rgb = rgba[..., :3]
    known = rgba[..., 3] > 0
    if known.all() or not known.any():
        return rgb.copy()

    def down(c, k):
        ph, pw = k.shape[0] % 2, k.shape[1] % 2
        if ph or pw:
            c = np.pad(c, ((0, ph), (0, pw), (0, 0)), mode="edge")
            k = np.pad(k, ((0, ph), (0, pw)), mode="edge")
        return (c[0::2, 0::2] + c[1::2, 0::2] + c[0::2, 1::2] + c[1::2, 1::2],
                k[0::2, 0::2] + k[1::2, 0::2] + k[0::2, 1::2] + k[1::2, 1::2])

    h, w = known.shape
    ph, pw = h % 2, w % 2
    rgbp = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge") if ph or pw else rgb
    kp = np.pad(known, ((0, ph), (0, pw)), mode="edge") if ph or pw else known
    hh, ww = kp.shape[0] // 2, kp.shape[1] // 2
    c = np.empty((hh, ww, 3), np.float32)
    k = np.empty((hh, ww), np.float32)
    for y in range(0, hh, band):
        rows = slice(2 * y, 2 * min(y + band, hh))
        cc, kk = down(rgbp[rows].astype(np.float32) * kp[rows][..., None], kp[rows].astype(np.float32))
        c[y:y + band], k[y:y + band] = cc, kk
    levels = []
    while max(k.shape) > 1:
        levels.append((c, k))
        c, k = down(c, k)
    col = c / np.maximum(k, 1e-6)[..., None]
    for c_l, k_l in reversed(levels):
        up = np.repeat(np.repeat(col, 2, 0), 2, 1)[:k_l.shape[0], :k_l.shape[1]]
        col = np.where(k_l[..., None] > 0, c_l / np.maximum(k_l, 1e-6)[..., None], up)
    col8 = np.clip(np.rint(col), 0, 255).astype(np.uint8)
    del col, levels
    out = rgb.copy()
    for y in range(0, h, 2 * band):
        y1 = min(y + 2 * band, h)
        up = np.repeat(np.repeat(col8[y // 2:(y1 + 1) // 2], 2, 0), 2, 1)[:y1 - y, :w]
        np.copyto(out[y:y1], up, where=~known[y:y1, :, None])
    return out


def finish_cutout(img: Image.Image, *, decontaminate: bool = True, hidden: str = "black",
                  keep: np.ndarray | None = None) -> Image.Image:
    """Snap alpha (>= 250 -> 255, <= 2 -> 0), estimate the true colour of the soft edge pixels and remove the
    background tint from a 1-px rim (decontaminate), then fill fully transparent pixels: black, the nearest visible
    colour (hidden="edge"), or leave them as they are (hidden="keep"). keep: pixels whose colour is already clean
    (e.g. the straight colours of a transparent input) and must not change."""
    rgba = np.array(img.convert("RGBA"))
    src = rgba[..., :3].copy()
    alpha = snap_alpha(rgba[..., 3])
    rgba[..., 3] = alpha
    if decontaminate:
        soft = (alpha > 0) & (alpha < 255)
        windows = list(_windows(soft, 64))
        if windows:
            fy0, fy1 = min(o[0] for _, o in windows), max(o[1] for _, o in windows)  # the frame: bounding box + pad
            fx0, fx1 = min(o[2] for _, o in windows), max(o[3] for _, o in windows)
            local = [((a - fy0, b - fy0, c - fx0, d - fx0), (e - fy0, f - fy0, g - fx0, h - fx0))
                     for (a, b, c, d), (e, f, g, h) in windows]
            for (iy0, iy1, ix0, ix1), F in _fb_tiled(src[fy0:fy1, fx0:fx1], alpha[fy0:fy1, fx0:fx1], local):
                sub = rgba[fy0 + iy0:fy0 + iy1, fx0 + ix0:fx0 + ix1]
                m = soft[fy0 + iy0:fy0 + iy1, fx0 + ix0:fx0 + ix1]
                sub[m, :3] = np.clip(np.rint(F[m] * 255), 0, 255).astype(np.uint8)
        _despill_rim(rgba, src)
        if keep is not None:
            rgba[keep, :3] = src[keep]
    if hidden == "edge":
        rgba[..., :3] = fill_hidden(rgba)
    elif hidden != "keep":
        rgba[alpha == 0, :3] = 0
    return Image.fromarray(rgba, "RGBA")


def visible_rgb(img: Image.Image) -> Image.Image:
    """RGB copy for models that ignore alpha (upscaler, watermark removal): fully transparent pixels take the
    nearest visible colour, so a black or leftover background does not bleed into the edges."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    return Image.fromarray(fill_hidden(np.asarray(img)), "RGB")


def with_alpha(rgb: Image.Image, alpha: Image.Image, *, hidden: str = "black") -> Image.Image:
    """Put an alpha channel back on a model's RGB output: snap it, and clear the fully transparent pixels
    (hidden="keep": leave the colours and the alpha exactly as they are, e.g. for textures with data in alpha)."""
    if hidden == "keep":
        out = rgb.convert("RGB")
        out.putalpha(alpha.convert("L"))
        return out
    rgba = np.array(rgb.convert("RGB").convert("RGBA"))
    a = snap_alpha(np.asarray(alpha.convert("L")))
    rgba[..., 3] = a
    if hidden == "edge":
        rgba[..., :3] = fill_hidden(rgba)
    else:
        rgba[a == 0, :3] = 0
    return Image.fromarray(rgba, "RGBA")
