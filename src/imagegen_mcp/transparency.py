"""Transparent-background support.

Qwen-Image-2.1 has an RGBA VAE: prompts written with the official template
produce a real alpha channel. In practice (stable-diffusion.cpp issue #2024) the
native alpha often keeps opaque white patches around the subject, so by
default the native alpha is intersected with a BiRefNet matte ("hybrid").
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


def rgba_prompt(prompt: str) -> str:
    p = prompt.strip()
    if p.startswith("This is an RGBA image"):
        return p
    if p and p[-1] not in ".!?":
        p += "."
    return f"{RGBA_PREFIX}{p}{RGBA_SUFFIX}"


def alpha_stats(img: Image.Image) -> dict:
    if img.mode != "RGBA":
        return {"transparent_pct": 0.0, "opaque_pct": 100.0}
    a = np.asarray(img.getchannel("A"))
    return {"transparent_pct": round(float((a < 16).mean() * 100), 1),
            "opaque_pct": round(float((a > 239).mean() * 100), 1)}


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
        mask = Image.fromarray((pred * 255).astype(np.uint8), "L")
        return mask.resize(img.size, Image.Resampling.LANCZOS)


def composite_on(img: Image.Image, color=(255, 255, 255)) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, color)
    bg.paste(img, mask=img.getchannel("A"))
    return bg


def apply_transparency(img: Image.Image, method: str, matter: Matter | None) -> tuple[Image.Image, dict]:
    """Post-process a generated image into a clean RGBA cut-out."""
    info: dict = {"method": method, "native": alpha_stats(img)}
    if method == "native" or matter is None:
        if method != "native":
            info["note"] = "matte model unavailable, kept the model's native alpha"
        return img.convert("RGBA"), info

    # Matte on the image as the viewer would see it over white.
    matte = np.asarray(matter.matte(composite_on(img)), dtype=np.uint8)
    rgba = np.array(img.convert("RGBA"))
    if method == "hybrid" and info["native"]["transparent_pct"] >= 5.0:
        rgba[..., 3] = np.minimum(rgba[..., 3], matte)
    else:
        # Native alpha is unusable (almost nothing transparent): rely on the matte.
        rgba[..., 3] = matte
        if method == "hybrid":
            info["note"] = "native alpha was nearly opaque, used the matte only"
    out = Image.fromarray(rgba, "RGBA")
    info["final"] = alpha_stats(out)
    return out, info


def cutout(img: Image.Image, matter: Matter) -> Image.Image:
    """Plain background removal for an arbitrary input image."""
    matte = matter.matte(composite_on(img) if img.mode == "RGBA" else img)
    out = img.convert("RGBA")
    if img.mode == "RGBA":
        a = np.minimum(np.asarray(img.getchannel("A")), np.asarray(matte))
        out.putalpha(Image.fromarray(a, "L"))
    else:
        out.putalpha(matte)
    return out


def default_threads() -> int:
    return max(1, min(8, (os.cpu_count() or 4) // 2))
