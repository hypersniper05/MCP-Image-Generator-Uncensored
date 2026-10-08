"""High level image operations shared by the MCP tools and HTTP routes."""

from __future__ import annotations

import asyncio
import collections
import contextvars
import hashlib
import io
import json
import os
import logging
import math
import re
import secrets
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from PIL import Image

from . import imaging, loras, panorama, transparency
from .config import Config
from .devices import DevicePlan, build_device_plan
from .jobs import JobManager
from .imaging import ImageInputError
from .models import ModelStore
from .schedule import official_sigmas
from .sdserver import EngineError, SdServer

log = logging.getLogger("imagegen.service")

Progress = Callable[[float, str], Awaitable[None]]

# Set per request from the Host header, so links work over LAN, VPN (e.g. Tailscale) or localhost alike.
# Jobs are asyncio tasks, which copy the context of the request that started them.
REQUEST_BASE_URL: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_base_url", default=None)


def base_url_from_headers(headers) -> str | None:
    if not headers:
        return None
    host = (headers.get("x-forwarded-host") or headers.get("host") or "").split(",")[0].strip()
    if not host or re.search(r"[/\\\s@]", host):
        return None
    proto = (headers.get("x-forwarded-proto") or "http").split(",")[0].strip().lower()
    if proto not in ("http", "https"):
        proto = "http"
    return f"{proto}://{host}"


async def _noop_progress(_: float, __: str) -> None:
    return None


class ServiceUnavailable(RuntimeError):
    pass


class PreviewRegistry:
    """sha256(preview bytes) -> full-resolution output file.

    Results that are too large to inline are sent as reduced previews. Clients (e.g. chat UIs) often send
    those previews back as inputs for the next edit; this lets the server use the original instead.
    """

    def __init__(self, outputs: Path, max_entries: int = 5000):
        self.outputs = outputs
        self.path = outputs / ".previews.json"
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._map: collections.OrderedDict[str, str] = collections.OrderedDict()
        try:
            self._map.update(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    def register(self, data: bytes, rel: str) -> None:
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            self._map[digest] = rel
            self._map.move_to_end(digest)
            while len(self._map) > self.max_entries:
                self._map.popitem(last=False)
            snapshot = json.dumps(self._map)
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(snapshot, encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.debug("could not save %s: %s", self.path, exc)

    def lookup(self, data: bytes) -> Path | None:
        with self._lock:
            rel = self._map.get(hashlib.sha256(data).hexdigest())
        if rel is None:
            return None
        p = (self.outputs / rel).resolve()
        return p if p.is_file() and p.is_relative_to(self.outputs.resolve()) else None


UPSCALE_MAX_SIDE = 8192  # results stay within sd-server's /sdcpp/v1/upscale limit, per side
PANORAMA_WRAP_PX = 32  # input columns copied from the opposite edge on each side when upscaling a 360 panorama
SAVED_CACHE_BYTES = 2 * 1024 * 1024  # results up to this size stay in memory; larger ones are re-read from disk
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")  # files listed and served from outputs/


def servable(rel: str) -> bool:
    """Whether a path under outputs/ is listed and served: an image file outside hidden folders (.previews.json,
    .stversions/ ...)."""
    parts = rel.replace("\\", "/").split("/")
    return not any(p.startswith(".") for p in parts) and Path(parts[-1]).suffix.lower() in IMAGE_SUFFIXES


def _is_sphere_file(path: Path) -> bool:
    """True for a 2:1 image file tagged as a full 360 panorama (reads only the header and metadata)."""
    try:
        with Image.open(path) as im:
            return panorama.is_two_to_one(*im.size) and panorama.sphere_xmp(im) is not None
    except Exception:  # noqa: BLE001 - an unreadable file is simply not a panorama
        return False


def _input_notes(images: list[Image.Image]) -> list[str]:
    return [im.info["imagegen_note"] for im in images if im.info.get("imagegen_note")]


def merge_negative(built_in: str, extra: str, cfg_scale: float, notes: list[str]) -> str:
    """The negative prompt to use: the built-in list plus the caller's extra terms (duplicates dropped).

    Guidance of 1 or less ignores the negative prompt, so none is sent then.
    """
    if cfg_scale <= 1:
        if extra.strip():
            notes.append("negative_prompt has no effect at cfg_scale 1; raise cfg_scale to use it")
        return ""
    terms: list[str] = []
    for part in f"{built_in},{extra}".split(","):
        term = part.strip()
        if term and term.lower() not in (t.lower() for t in terms):
            terms.append(term)
    return ", ".join(terms)


def _meminfo() -> dict[str, int]:
    """/proc/meminfo in bytes (empty dict when unavailable)."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    out[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


@dataclass
class SavedImage:
    """A saved result. Finished jobs are kept for a day, so only small files stay in memory; larger ones are read
    back from disk when a result is built (an 8192 px upscale would otherwise hold ~0.4 GB per job)."""
    path: Path
    filename: str
    url: str
    mime: str
    width: int
    height: int
    size_bytes: int
    view_url: str | None = None
    cached: bytes | None = field(default=None, repr=False)

    @property
    def data(self) -> bytes:
        return self.cached if self.cached is not None else self.path.read_bytes()

    @property
    def image(self) -> Image.Image:
        img = Image.open(io.BytesIO(self.data))
        img.load()
        return img


@dataclass
class OpResult:
    images: list[SavedImage]
    info: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class ImageService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.plan: DevicePlan = build_device_plan(cfg)
        self.store = ModelStore(cfg)
        self.engine: SdServer | None = None
        self.matter: transparency.Matter | None = None
        self.watermark_lora: str | None = None  # LoRA path for the engine, set when remove_watermark can run
        self.watermark_error: str | None = None
        self._optional_task: asyncio.Task | None = None
        self.upscaler_name: str | None = None  # model name, set when upscale_image can run
        self.upscaler_path: Path | None = None
        self.upscaler_error: str | None = None
        self._paths: dict[str, Path] = {}
        self.state = "starting"  # starting | downloading | loading | ready | error
        self.error: str | None = None
        self.started_at = time.time()
        self.ready_at: float | None = None
        self.jobs_done = 0
        self.active_jobs = 0
        self._circular_failed = False  # set when sd-server ran a circular request without applying it
        self.jobs = JobManager()
        self.outputs = Path(cfg.outputs_dir)
        self.previews = PreviewRegistry(self.outputs)
        self.loader = imaging.ImageLoader(self.outputs, max_bytes=cfg.server.max_request_mb * 1024 * 1024,
                                          original_for=self.previews.lookup)

    # ------------------------------------------------------------ lifecycle
    def _check_environment(self) -> None:
        """Fail early with an actionable message instead of a confusing error later."""
        for d in (Path(self.cfg.models_dir), self.outputs):
            d.mkdir(parents=True, exist_ok=True)
            probe = d / f".write-test-{uuid.uuid4().hex[:6]}"
            try:
                probe.write_bytes(b"ok")
                probe.unlink()
            except OSError as exc:
                raise RuntimeError(
                    f"{d} is not writable inside the container ({exc}). On Linux, give the folder to the "
                    f"container user, e.g. `sudo chown -R $(id -u):$(id -g) models outputs`.") from exc
        mem = _meminfo()
        need_gb = 14 if self.cfg.is_cpu else 8
        if mem.get("MemTotal") and mem["MemTotal"] / 1e6 < need_gb:
            log.warning("the container sees only %.1f GB of RAM; %d GB or more is recommended for %s mode. "
                        "On Windows raise the WSL memory limit in %%UserProfile%%\\.wslconfig ([wsl2] memory=...).",
                        mem["MemTotal"] / 1e6, need_gb, self.cfg.device)

    async def startup(self) -> None:
        try:
            self._check_environment()
            log.info("device plan: %s", self.plan.description)
            self.state = "downloading"
            paths = await self.store.ensure()
            if self.cfg.transparency.method in ("hybrid", "matte"):
                bg = self.cfg.gpu.background_removal if not self.cfg.is_cpu else "cpu"
                self.matter = transparency.Matter(paths["background_removal"], self.cfg.transparency.matte_model,
                                                  device=bg,
                                                  threads=self.cfg.transparency.threads or transparency.default_threads())
            self._paths = paths
            self._prepare_watermark_lora(paths)
            self._prepare_upscaler(paths)
            # the engine reads these folders from its launch arguments, so they must exist before a later download
            if self.cfg.watermark.enabled:
                (Path(self.cfg.models_dir) / "loras").mkdir(parents=True, exist_ok=True)
            if self.cfg.upscale.enabled:
                (Path(self.cfg.models_dir) / "upscalers").mkdir(parents=True, exist_ok=True)
            self.state = "loading"
            self.engine = SdServer(self.cfg, self.plan, paths)
            if not self.cfg.is_cpu:
                devices = await self.engine.list_devices()
                cuda = [ln for ln in devices.splitlines() if ln.lower().startswith("cuda")]
                if not cuda:
                    raise RuntimeError(
                        "device is 'gpu' but no CUDA device is visible inside the container. Check that the "
                        "NVIDIA Container Toolkit is installed and that the container was started with the gpu "
                        "profile, or set device: cpu in config.yaml.\n" + devices[-2000:])
                log.info("visible devices:\n%s", devices.strip())
            await self.engine.start()
            if self.cfg.generation.warmup:
                await self._warmup()
            self.state = "ready"
            self.ready_at = time.time()
            log.info("ready after %.0f s. MCP endpoint: %s", self.ready_at - self.started_at, self.mcp_url())
            asyncio.create_task(self._cleanup_loop())
            self.engine.start_idle_watch()
            if self.store.pending_optional:
                self._optional_task = asyncio.create_task(self._fetch_optional())
        except Exception as exc:  # noqa: BLE001
            self.state = "error"
            self.error = str(exc)
            log.exception("startup failed: %s", exc)

    async def _fetch_optional(self) -> None:
        """Download optional model files after startup, then enable what needs them."""
        try:
            paths = await self.store.fetch_optional()
        except Exception as exc:  # noqa: BLE001 - optional files must never take the server down
            log.warning("optional downloads failed: %s", exc)
            paths = {}
        self._paths.update(paths)
        if self.cfg.watermark.enabled and self.watermark_lora is None:
            self._prepare_watermark_lora(self._paths, downloading=False)
        if self.cfg.upscale.enabled and self.upscaler_name is None:
            self._prepare_upscaler(self._paths, downloading=False)

    def _prepare_upscaler(self, paths: dict[str, Path], downloading: bool = True) -> None:
        if not self.cfg.upscale.enabled:
            return
        if "upscaler" in paths:
            self.upscaler_path = Path(paths["upscaler"])
            self.upscaler_name = self.upscaler_path.stem
            self.upscaler_error = None
            return
        pending = downloading and any(f.key == "upscaler" for f in self.store.pending_optional)
        self.upscaler_error = ("the upscaler model is still downloading; try again in a minute" if pending else
                               "the upscaler model is not available (download failed or model.auto_download is "
                               "false); see the server log")

    def _prepare_watermark_lora(self, paths: dict[str, Path], downloading: bool = True) -> None:
        if not self.cfg.watermark.enabled:
            return
        lora = paths.get("watermark_lora")
        if lora is None:
            pending = downloading and any(f.key == "watermark_lora" for f in self.store.pending_optional)
            self.watermark_error = ("the watermark-removal LoRA is still downloading; try again in a minute"
                                    if pending else
                                    "the watermark-removal LoRA is not available (download failed or "
                                    "model.auto_download is false); see the server log")
            return
        try:
            layout = loras.mlp_layout(paths["diffusion"])
            self.watermark_lora = loras.prepare_lora(lora, Path(self.cfg.models_dir) / "loras", layout)
            self.watermark_error = None
        except (OSError, ValueError) as exc:
            self.watermark_error = f"the watermark-removal LoRA could not be prepared: {exc}"
            log.warning("%s", self.watermark_error)

    async def _warmup(self) -> None:
        log.info("warming up: loading the weights%s and running a tiny test generation",
                 "" if self.cfg.is_cpu else ", preparing GPU kernels")
        t = time.monotonic()
        body = self._body("a white cube", "", 256, 256, steps=1, seed=1)
        try:
            await self.engine.generate(body, timeout=self.cfg.timeout_seconds)
            log.info("warm-up finished in %.1f s", time.monotonic() - t)
        except Exception as exc:  # noqa: BLE001 - a failed warm-up must not take the server down
            log.warning("warm-up failed (the first request will be slower): %s", exc)

    async def shutdown(self) -> None:
        if self.engine:
            await self.engine.stop()

    async def _cleanup_loop(self) -> None:
        keep = self.cfg.outputs.keep_days
        if keep <= 0:
            return
        while True:
            try:
                cutoff = time.time() - keep * 86400
                for p in self.outputs.rglob("*"):
                    # hidden files such as the repository's .gitkeep are not outputs
                    if p.is_file() and not p.name.startswith(".") and p.stat().st_mtime < cutoff:
                        p.unlink(missing_ok=True)
                dirs = [p for p in self.outputs.rglob("*") if p.is_dir()]
                for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):  # deepest first
                    if not any(d.iterdir()):
                        d.rmdir()
            except OSError as exc:
                log.debug("cleanup: %s", exc)
            await asyncio.sleep(3600)

    def require_ready(self) -> SdServer:
        if self.state == "ready" and self.engine is not None:
            return self.engine
        if self.state == "error":
            raise ServiceUnavailable(f"the image server failed to start: {self.error}")
        if self.state == "downloading":
            st = self.store.status.as_dict()
            raise ServiceUnavailable(
                f"the model is still downloading ({st['percent']}% of {st['total_gb']} GB, current file "
                f"{st['current_file']}). Try again in a few minutes.")
        raise ServiceUnavailable(f"the model is still loading (state: {self.state}). Try again shortly.")

    # ------------------------------------------------------------ helpers
    def base_url(self) -> str:
        """Base URL for links in results: config public_url, else the address the client used."""
        return self.cfg.server.public_url or REQUEST_BASE_URL.get() or f"http://localhost:{self.cfg.server.port}"

    def mcp_url(self) -> str:
        return self.base_url() + self.cfg.server.path

    def _seed(self, seed: int | None) -> int:
        if seed is None or seed < 0:
            return secrets.randbelow(2**31 - 1)
        return int(seed)

    def _body(self, prompt: str, negative: str, width: int, height: int, *, steps: int, seed: int,
              cfg_scale: float | None = None, refs: list[str] | None = None, init: str | None = None,
              mask: str | None = None, strength: float | None = None, lora: tuple[str, float] | None = None,
              ref_max_pixels: int | None = None, circular: tuple[bool, bool] = (False, False)) -> dict:
        g = self.cfg.generation
        scale = self.cfg.default_cfg_scale if cfg_scale is None else cfg_scale
        sample: dict = {"sample_method": g.sampler, "sample_steps": int(steps),
                        "guidance": {"txt_cfg": float(scale), "img_cfg": float(scale)}}
        if g.schedule == "official":
            sample["custom_sigmas"] = official_sigmas(width, height, steps)
        elif g.scheduler:
            sample["scheduler"] = g.scheduler
        body: dict = {
            "prompt": prompt, "negative_prompt": negative or "", "width": int(width), "height": int(height),
            "seed": int(seed), "batch_count": 1, "sample_params": sample, "output_format": "png",
            "embed_image_metadata": False,
            "ref_image_args": f"vae_input_max_pixels={ref_max_pixels or int(g.ref_max_megapixels * 1_000_000)}",
        }
        if self.engine is not None and (tiling := self.engine.vae_tiling_for(width, height)):
            body["vae_tiling_params"] = tiling
        if refs:
            body["ref_images"] = refs
        if init is not None:
            body["init_image"] = init
            body["strength"] = float(strength if strength is not None else 0.75)
        if mask is not None:
            body["mask_image"] = mask
        if lora is not None:  # path relative to models/loras
            body["lora"] = [{"path": lora[0], "multiplier": float(lora[1])}]
        if any(circular):  # read only by an engine built with the sdcpp-circular-json patch
            body["circular_x"], body["circular_y"] = bool(circular[0]), bool(circular[1])
        return body

    def _use_circular(self, method: str, notes: list[str]) -> bool:
        """Whether to generate with the engine's wrap-around mode (generation.tile_method, panorama.wrap_method)."""
        if method == "repair" or self.engine is None:
            return False
        if getattr(self.engine, "supports_circular", False) and not self._circular_failed:
            return True
        if method == "circular":
            notes.append("this engine build cannot generate wrap-around edges directly; used the seam repair pass")
        return False

    def _is_tile_input(self, ref: str, img: Image.Image) -> bool:
        """A seamless tile: tagged by this server, or one of its tile-* results (also those made before the tag)."""
        if panorama.is_tile(img):
            return True
        path = self.loader.local_path(ref) if isinstance(ref, str) else None
        return path is not None and path.name.startswith("tile-") and path.is_relative_to(self.outputs.resolve())

    def _fit_token_budget(self, width: int, height: int, refs: list[Image.Image],
                          notes: list[str]) -> tuple[list[Image.Image], int]:
        """Pixels allowed per reference image so (output + reference pixels) / 256 stays within
        generation.token_budget, and the references reduced to it. Every reference adds tokens to each denoising
        step and to the text encoder. sd-server resizes each reference to its vae_input_max_pixels, so the
        returned per-reference size must also go into the request (ref_max_pixels)."""
        per_ref = int(self.cfg.generation.ref_max_megapixels * 1_000_000)
        budget = self.cfg.generation.token_budget
        if not budget or not refs:
            return refs, per_ref
        room = budget * 256 - width * height
        if len(refs) * per_ref <= room:
            return refs, per_ref
        per_ref = max(256 * 256, room // len(refs))  # keep each reference usable
        out = [imaging.fit_reference(im, per_ref) for im in refs]
        notes.append(f"the input images were reduced to {per_ref / 1e6:.2f} MP each to fit the per-request memory "
                     f"budget ({len(refs)} image(s) with a {width}x{height} result)")
        return out, per_ref

    async def _run(self, body: dict, progress: Progress) -> Image.Image:
        return (await self._run_checked(body, progress))[0]

    async def _run_checked(self, body: dict, progress: Progress) -> tuple[Image.Image, bool]:
        """Run one generation. The flag is True when the body asked for circular_x / circular_y and sd-server
        applied it (it logs "Using circular padding ..." for each such request; an unpatched build ignores the
        keys and logs nothing)."""
        engine = self.require_ready()
        wrap = (bool(body.get("circular_x")), bool(body.get("circular_y")))
        acks = getattr(engine, "circular_acks", 0)
        self.active_jobs += 1
        try:
            pngs = await engine.generate(body, timeout=self.cfg.timeout_seconds, on_progress=progress)
        finally:
            self.active_jobs -= 1
        self.jobs_done += 1
        applied = any(wrap) and getattr(engine, "circular_acks", 0) > acks
        if any(wrap) and not applied:
            self._circular_failed = True
            log.warning("sd-server did not apply circular_x/circular_y; using the seam repair pass from now on")
            wrap = (False, False)
        img = Image.open(io.BytesIO(pngs[0]))
        img.load()
        if self.cfg.outputs.degrid:
            img = await asyncio.to_thread(imaging.degrid, img, wrap=wrap)
        return img, applied

    def _check_disk(self) -> None:
        free = shutil.disk_usage(self.outputs).free
        if free < 512 * 1024 * 1024:
            raise ServiceUnavailable(f"the outputs disk is almost full ({free / 1e6:.0f} MB free); delete old files "
                                     "in ./outputs or lower outputs.keep_days")

    def _save(self, img: Image.Image, fmt: str, kind: str, xmp: bytes | None = None) -> SavedImage:
        self._check_disk()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        name = f"{kind}-{datetime.now(timezone.utc).strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}.{imaging.EXT[fmt]}"
        folder = self.outputs / day
        folder.mkdir(parents=True, exist_ok=True)
        data = imaging.encode(img, fmt, quality=self.cfg.outputs.jpeg_quality, xmp=xmp)
        path = folder / name
        path.write_bytes(data)
        rel = f"{day}/{name}"
        return SavedImage(path=path, filename=rel, url=f"{self.base_url()}/outputs/{rel}", mime=imaging.MIME[fmt],
                          width=img.width, height=img.height, size_bytes=len(data),
                          cached=data if len(data) <= SAVED_CACHE_BYTES else None)

    def _finish_rgb(self, img: Image.Image) -> Image.Image:
        # Outputs of the RGBA VAE are 4-channel even for normal prompts; drop the
        # alpha channel unless the image really has transparency.
        return img.convert("RGB") if not imaging.has_transparency(img, output=True) else img

    def _model_rgb(self, src: Image.Image, alpha_in: bool) -> Image.Image:
        """RGB input for a model that ignores alpha: transparent pixels take the nearest visible colour, so nothing
        dark bleeds into the edges (unless transparency.hidden_pixels is "keep")."""
        if alpha_in and self.cfg.transparency.hidden_pixels != "keep":
            return transparency.visible_rgb(src)
        return src.convert("RGB")

    @property
    def _alpha_opts(self) -> dict:
        t = self.cfg.transparency
        return {"decontaminate": t.decontaminate, "hidden": t.hidden_pixels}

    # ------------------------------------------------------------ operations
    async def generate(self, *, prompt: str, negative_prompt: str = "", width: int | None = None,
                       height: int | None = None, aspect_ratio: str | None = None, size: str | None = None,
                       transparent: bool = False, steps: int | None = None, cfg_scale: float | None = None,
                       seed: int | None = None, output_format: str | None = None, tileable: bool = False,
                       progress: Progress = _noop_progress) -> OpResult:
        if tileable and transparent:
            raise ValueError("tileable and transparent cannot be combined; choose one")
        w, h, notes = imaging.resolve_size(width=width, height=height, aspect_ratio=aspect_ratio, size=size,
                                           default=self.cfg.default_size, max_pixels=self.cfg.max_pixels)
        seed_v = self._seed(seed)
        steps_v = steps or self.cfg.default_steps
        text = transparency.rgba_prompt(prompt) if transparent else prompt
        scale = self.cfg.default_cfg_scale if cfg_scale is None else cfg_scale
        negative = merge_negative(self.cfg.generation.negative_prompt, negative_prompt, scale, notes)

        def stage(lo: float, hi: float, label: str) -> Progress:
            async def cb(frac: float, msg: str) -> None:
                await progress(lo + (hi - lo) * frac, f"{label}: {msg}")
            return cb

        # Tiles: one pass in which the model and the VAE treat opposite edges as neighbours (circular), else the
        # older generate-then-repair method.
        circ = tileable and self._use_circular(self.cfg.generation.tile_method, notes)
        img, circ = await self._run_checked(
            self._body(text, negative, w, h, steps=steps_v, seed=seed_v, cfg_scale=scale, circular=(circ, circ)),
            stage(0.0, 0.7, "image") if tileable and not circ else progress)
        info = {"prompt": prompt, "seed": seed_v, "steps": steps_v, "width": img.width, "height": img.height,
                "cfg_scale": scale}
        if tileable and circ:
            img = img.convert("RGB")
            info["tile_method"] = "circular"
            info["tile_seam"] = panorama.tile_seam_score(img)
            info["tile_wrap"] = panorama.wrap_report(img)
        elif tileable:
            # Fallback: shift by half so the wrap-around edges meet in the middle, repaint a cross-shaped band
            # there, then shift back. It can leave a line or a pattern break at the edges.
            img = img.convert("RGB")
            info["tile_method"] = "repair"
            info["tile_seam_before"] = panorama.tile_seam_score(img)
            rolled = panorama.roll_both(img)
            hard, soft = panorama.cross_mask(rolled.size, self.cfg.panorama.seam_band_fraction)
            try:
                patched = await self._run(
                    self._body(text, negative, rolled.width, rolled.height, steps=max(8, steps_v), seed=seed_v + 1,
                               cfg_scale=scale, init=imaging.to_png_b64(rolled), mask=imaging.to_png_b64(hard),
                               strength=self.cfg.panorama.seam_strength),
                    stage(0.7, 0.98, "seam repair"))
                img = panorama.roll_both(panorama.blend(rolled, patched, soft), back=True)
            except EngineError as exc:
                notes.append(f"tile seam repair failed ({exc}); the edges may not line up")
            info["tile_seam_after"] = panorama.tile_seam_score(img)
            info["tile_wrap"] = panorama.wrap_report(img)
        fmt = (output_format or self.cfg.outputs.format).lower()
        if transparent:
            await progress(0.97, "cleaning up the transparent background")
            img, tinfo = await asyncio.to_thread(transparency.apply_transparency, img,
                                                 self.cfg.transparency.method, self.matter, **self._alpha_opts)
            info["transparency"] = tinfo
            if fmt == "jpeg":
                notes.append("JPEG cannot store transparency, saved as PNG instead")
                fmt = "png"
        else:
            img = self._finish_rgb(img)
        saved = self._save(img, fmt, "transparent" if transparent else "tile" if tileable else "image",
                           xmp=panorama.tile_xmp() if tileable else None)  # later edits/upscales keep it seamless
        return OpResult([saved], info, notes)

    async def edit(self, *, prompt: str, images: list[str], mask: str | None = None, negative_prompt: str = "",
                   width: int | None = None,
                   height: int | None = None, aspect_ratio: str | None = None, size: str | None = None,
                   transparent: bool = False, steps: int | None = None, cfg_scale: float | None = None,
                   seed: int | None = None, output_format: str | None = None, tileable: bool | None = None,
                   progress: Progress = _noop_progress) -> OpResult:
        if not images:
            raise ValueError("provide at least one input image")
        limit = self.cfg.generation.max_reference_images
        if len(images) > limit:
            raise ValueError(f"at most {limit} input images are supported")
        self.require_ready()
        await progress(0.0, f"loading {len(images)} input image(s)")
        loaded = [await self.loader.load(ref) for ref in images]
        first = loaded[0]
        # Editing a seamless tile (e.g. into a height, normal or roughness map) keeps it seamless: the edit runs in
        # the engine's wrap-around mode, at the tile's own size unless a size is given.
        tile = tileable if tileable is not None else self._is_tile_input(images[0], first)
        if tile and transparent:
            raise ValueError("tileable and transparent cannot be combined; choose one")
        if tile and width is None and height is None and aspect_ratio is None and size is None:
            width, height = first.width, first.height
        if mask:
            if len(loaded) >= limit:
                raise ValueError(f"a mask counts as an input image; use at most {limit - 1} images with a mask")
            m = await self.loader.load(mask)
            loaded.append(imaging.normalize_mask(m, first.size))
            k = len(loaded)
            prompt = (f"{prompt.strip().rstrip('.')}. Apply the change only inside the area marked white in the mask "
                      f"<image{k}>; keep everything outside that area of <image1> unchanged.")
        w, h, notes = imaging.resolve_size(
            width=width, height=height, aspect_ratio=aspect_ratio, size=size, default=self.cfg.default_size,
            max_pixels=self.cfg.max_pixels, reference=(first.width, first.height))
        notes += _input_notes(loaded)
        max_ref = int(self.cfg.generation.ref_max_megapixels * 1_000_000)
        fitted, per_ref = self._fit_token_budget(w, h, [imaging.fit_reference(im, max_ref) for im in loaded], notes)
        refs = [imaging.to_png_b64(im) for im in fitted]
        seed_v = self._seed(seed)
        steps_v = steps or self.cfg.default_steps
        text = transparency.rgba_prompt(prompt) if transparent else prompt
        # Edits need guidance: at cfg 1 the model tends to ignore the instruction and return an
        # over-sharpened copy of the input (measured with stable-diffusion.cpp).
        g = self.cfg.generation
        scale = g.edit_cfg_scale if cfg_scale is None else cfg_scale
        negative = merge_negative(g.edit_negative_prompt, negative_prompt, scale, notes)
        circ = tile and self._use_circular(self.cfg.generation.tile_method, [])
        img, circ = await self._run_checked(self._body(text, negative, w, h, steps=steps_v, seed=seed_v,
                                                       cfg_scale=scale, refs=refs, ref_max_pixels=per_ref,
                                                       circular=(circ, circ)), progress)
        info = {"prompt": prompt, "seed": seed_v, "steps": steps_v, "width": img.width, "height": img.height,
                "cfg_scale": scale, "input_images": [f"{im.width}x{im.height}" for im in loaded]}
        if circ:
            info["tileable"] = "seamless (wrap-around edges)"
            info["tile_wrap"] = panorama.wrap_report(img)
        elif tile:
            notes.append("this engine build cannot edit with wrap-around edges, so the result may not tile seamlessly")
        fmt = (output_format or self.cfg.outputs.format).lower()
        if transparent:
            await progress(0.97, "cleaning up the transparent background")
            img, tinfo = await asyncio.to_thread(transparency.apply_transparency, img,
                                                 self.cfg.transparency.method, self.matter, **self._alpha_opts)
            info["transparency"] = tinfo
            if fmt == "jpeg":
                notes.append("JPEG cannot store transparency, saved as PNG instead")
                fmt = "png"
        else:
            img = self._finish_rgb(img)
        saved = self._save(img, fmt, "edit", xmp=panorama.tile_xmp() if circ else None)
        return OpResult([saved], info, notes)

    async def panorama(self, *, prompt: str, image: str | None = None, negative_prompt: str = "",
                       width: int | None = None, steps: int | None = None, cfg_scale: float | None = None,
                       seed: int | None = None, seam_fix: bool | None = None, output_format: str | None = None,
                       progress: Progress = _noop_progress) -> OpResult:
        pw, ph = self.cfg.panorama_size
        if width:  # a multiple of 64, so the height is exactly half and still a multiple of 32
            pw = max(512, min(imaging.MAX_SIDE, int(round(width / 64)) * 64))
            ph = pw // 2
        notes: list[str] = []
        if pw * ph > self.cfg.max_pixels:
            s = math.sqrt(self.cfg.max_pixels / (pw * ph))
            pw = max(512, int(pw * s) // 64 * 64)
            ph = pw // 2
            notes.append(f"reduced the panorama to {pw}x{ph} (server pixel limit)")
        seed_v = self._seed(seed)
        steps_v = steps or self.cfg.default_steps
        refs = None
        per_ref = None
        if image:
            self.require_ready()
            src = await self.loader.load(image)
            notes += _input_notes([src])
            fitted, per_ref = self._fit_token_budget(pw, ph, [imaging.fit_reference(
                src, int(self.cfg.generation.ref_max_megapixels * 1_000_000))], notes)
            refs = [imaging.to_png_b64(im) for im in fitted]
            text = panorama.panorama_from_image_prompt(prompt, pw, ph)
        else:
            text = panorama.panorama_prompt(prompt, pw, ph)

        def stage(lo: float, hi: float, label: str) -> Progress:
            async def cb(frac: float, msg: str) -> None:
                await progress(lo + (hi - lo) * frac, f"{label}: {msg}")
            return cb

        fix = self.cfg.panorama.seam_fix if seam_fix is None else seam_fix
        circ = self._use_circular(self.cfg.panorama.wrap_method, notes)  # x only: no wrap from top to bottom
        g = self.cfg.generation
        if refs:  # extending a photo is an edit: it uses the edit guidance
            pano_cfg = g.edit_cfg_scale if cfg_scale is None else cfg_scale
            pano_neg = merge_negative(g.edit_negative_prompt, negative_prompt, pano_cfg, notes)
        else:
            pano_cfg = self.cfg.default_cfg_scale if cfg_scale is None else cfg_scale
            pano_neg = merge_negative(g.negative_prompt, negative_prompt, pano_cfg, notes)
        img, circ = await self._run_checked(
            self._body(text, pano_neg, pw, ph, steps=steps_v, seed=seed_v, cfg_scale=pano_cfg, refs=refs,
                       ref_max_pixels=per_ref, circular=(circ, False)),
            stage(0.0, 0.75 if fix and not circ else 1.0, "panorama"))
        img = img.convert("RGB")
        before = panorama.seam_score(img)
        info: dict = {"prompt": prompt, "seed": seed_v, "steps": steps_v, "width": img.width, "height": img.height,
                      "seam_score_before": before,
                      "wrap_method": "circular" if circ else "repair" if fix else "none"}
        if fix and not circ:
            rolled = panorama.roll_half(img)
            hard, soft = panorama.seam_mask(rolled.size, self.cfg.panorama.seam_band_fraction)
            fix_steps = max(8, steps_v)
            try:
                patched = await self._run(
                    self._body(text, pano_neg, rolled.width, rolled.height, steps=fix_steps,
                               seed=seed_v + 1, cfg_scale=pano_cfg, init=imaging.to_png_b64(rolled),
                               mask=imaging.to_png_b64(hard), strength=self.cfg.panorama.seam_strength),
                    stage(0.75, 0.98, "seam repair"))
                img = panorama.roll_half(panorama.blend(rolled, patched, soft), back=True)
            except EngineError as exc:
                notes.append(f"seam repair failed ({exc}); applied a simple cross-fade instead")
                img = panorama.crossfade_wrap(img)
            info["seam_score_after"] = panorama.seam_score(img)
        fmt = (output_format or self.cfg.panorama.format).lower()
        xmp = panorama.gpano_xmp(img.width, img.height) if self.cfg.panorama.embed_gpano else None
        saved = self._save(img, fmt, "panorama", xmp=xmp)
        saved.view_url = f"{self.base_url()}/view/{saved.filename}"
        return OpResult([saved], info, notes)

    async def remove_background(self, *, image: str, output_format: str = "png",
                                progress: Progress = _noop_progress) -> OpResult:
        if self.matter is None:
            if self.state != "ready":
                self.require_ready()
            raise ServiceUnavailable("background removal is disabled (transparency.method is 'native')")
        src = await self.loader.load(image)
        await progress(0.3, "running the background-removal model")
        out = await asyncio.to_thread(transparency.cutout, src, self.matter, **self._alpha_opts)
        if icc := imaging.rgb_icc(src):  # the visible pixels are the input's, so keep its colour profile
            out.info["icc_profile"] = icc
        fmt = "webp" if output_format == "webp" else "png"
        saved = self._save(out, fmt, "cutout")
        return OpResult([saved], {"width": out.width, "height": out.height,
                                  "alpha": transparency.alpha_stats(out), "model": self.cfg.transparency.matte_model,
                                  "decontaminated": self.cfg.transparency.decontaminate,
                                  "hidden_pixels": self.cfg.transparency.hidden_pixels}, _input_notes([src]))

    async def remove_watermark(self, *, image: str, steps: int | None = None, seed: int | None = None,
                               output_format: str | None = None, progress: Progress = _noop_progress) -> OpResult:
        wm = self.cfg.watermark
        if self.watermark_lora is None:
            self.require_ready()
            raise ServiceUnavailable("watermark removal is not available: "
                                     + (self.watermark_error or "it is disabled (watermark.enabled is false)"))
        self.require_ready()
        await progress(0.0, "loading the image")
        src = await self.loader.load(image)
        notes = _input_notes([src])
        alpha_in = imaging.has_transparency(src)
        rgb = await asyncio.to_thread(self._model_rgb, src, alpha_in)
        # The model works at about watermark.megapixels whatever the input size; the result goes back to the
        # input's size afterwards.
        target = min(int(wm.megapixels * 1_000_000), self.cfg.max_pixels)
        if self.cfg.generation.token_budget:  # the reference is the same size as the output
            target = min(target, self.cfg.generation.token_budget * 256 // 2)
        w, h = imaging.dims_for((src.width, src.height), target)
        work = rgb.resize((w, h), Image.Resampling.LANCZOS) if (w, h) != rgb.size else rgb
        seed_v = self._seed(seed)
        steps_v = steps or self.cfg.default_steps
        g = self.cfg.generation
        scale = g.edit_cfg_scale
        negative = merge_negative(g.edit_negative_prompt, "", scale, notes)
        out = await self._run(self._body(wm.prompt, negative, w, h, steps=steps_v, seed=seed_v, cfg_scale=scale,
                                         refs=[imaging.to_png_b64(work)], lora=(self.watermark_lora, wm.strength),
                                         ref_max_pixels=max(w * h, int(g.ref_max_megapixels * 1_000_000))),
                              progress)
        out = out.convert("RGB")
        if out.size != (w, h):
            out = out.resize((w, h), Image.Resampling.LANCZOS)
        info: dict = {"seed": seed_v, "steps": steps_v, "cfg_scale": scale, "processed_at": f"{w}x{h}",
                      "lora_strength": wm.strength}
        if wm.restore_unchanged:
            await progress(0.97, "keeping the unchanged areas of the original")
            result, changed = await asyncio.to_thread(imaging.restore_unchanged, rgb, work, out)
            info["changed_percent"] = round(changed * 100, 1)
            if result is None:
                notes.append("the model changed most of the image, so the whole result is used instead of only "
                             "the changed areas")
                result = out.resize(rgb.size, Image.Resampling.LANCZOS) if out.size != rgb.size else out
        else:
            result = out.resize(rgb.size, Image.Resampling.LANCZOS) if out.size != rgb.size else out
        fmt = (output_format or self.cfg.outputs.format).lower()
        if alpha_in:  # the model works in RGB; keep the input's alpha channel
            result = await asyncio.to_thread(transparency.with_alpha, result, src.convert("RGBA").getchannel("A"),
                                             hidden=self.cfg.transparency.hidden_pixels)
            if fmt == "jpeg":
                notes.append("JPEG cannot store transparency, saved as PNG instead")
                fmt = "png"
        if fmt == "webp" and max(result.size) > 16383:
            notes.append("WebP cannot store images over 16383 pixels per side, saved as PNG instead")
            fmt = "png"
        if icc := imaging.rgb_icc(src):  # the untouched pixels are the input's, so keep its color profile
            result.info["icc_profile"] = icc
        info.update(width=result.width, height=result.height)
        saved = self._save(result, fmt, "unwatermarked")
        return OpResult([saved], info, notes)

    async def upscale(self, *, image: str, scale: int = 4, output_format: str | None = None,
                      as_panorama: bool | None = None, as_tileable: bool | None = None,
                      progress: Progress = _noop_progress) -> OpResult:
        if self.upscaler_name is None:
            self.require_ready()
            raise ServiceUnavailable("image upscaling is not available: "
                                     + (self.upscaler_error or "it is disabled (upscale.enabled is false)"))
        engine = self.require_ready()
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        await progress(0.0, "loading the image")
        src = await self.loader.load(image)
        notes = _input_notes([src])
        # A 360 panorama stays one: its left and right edges are upscaled as neighbours, and the result keeps the
        # Photo Sphere metadata and gets a viewer link.
        sphere = panorama.sphere_xmp(src)
        two_to_one = panorama.is_two_to_one(src.width, src.height)
        if as_panorama and not two_to_one:
            raise ValueError(f"panorama=true needs a 2:1 equirectangular image; this one is {src.width}x{src.height}")
        pano = two_to_one and (as_panorama or (as_panorama is None and sphere is not None))
        if as_panorama is None and sphere is not None and not two_to_one:
            notes.append("the image has 360 metadata but is not 2:1, so it was upscaled as a plain image")
        # A seamless tile (tagged by generate_image(tileable=true), or as_tileable) wraps in both directions.
        tile = not pano and (as_tileable if as_tileable is not None else self._is_tile_input(image, src))
        alpha_in = imaging.has_transparency(src)
        rgb = await asyncio.to_thread(self._model_rgb, src, alpha_in)
        target = (src.width * scale, src.height * scale)
        if (longest := max(target)) > UPSCALE_MAX_SIDE:  # integer math: the long side is exactly the limit
            target = (max(1, target[0] * UPSCALE_MAX_SIDE // longest), max(1, target[1] * UPSCALE_MAX_SIDE // longest))
            notes.append(f"the result is limited to {UPSCALE_MAX_SIDE} px per side, so it is {target[0]}x{target[1]}")
        # The model always enlarges 4x, and the engine allows at most 8192 px per side.
        work = rgb
        if max(rgb.size) * 4 > UPSCALE_MAX_SIDE:
            f = UPSCALE_MAX_SIDE / 4 / max(rgb.size)
            work = rgb.resize((max(1, round(rgb.width * f)), max(1, round(rgb.height * f))), Image.Resampling.LANCZOS)
            notes.append(f"the input is larger than {UPSCALE_MAX_SIDE // 4} px per side, so it was reduced to "
                         f"{work.width}x{work.height} before upscaling")
        await progress(0.1, f"upscaling {work.width}x{work.height} 4x with {self.upscaler_name}")
        # Wrapped columns (and rows, for a tile) on each side let the model see across the wrap-around; they are
        # cropped off afterwards. sd-cli has no size limit, so the padded pass may briefly exceed 8192 px.
        pad = min(PANORAMA_WRAP_PX, work.width) if pano or tile else 0
        pad_y = min(PANORAMA_WRAP_PX, work.height) if tile else 0
        model_input = panorama.wrap_pad(work, pad, pad_y) if pad or pad_y else work
        t = time.monotonic()
        self.active_jobs += 1
        try:
            async def waiting(msg: str) -> None:
                await progress(0.1, msg)

            timeout = (self.cfg.generation.cpu_timeout_seconds if self.cfg.is_cpu or self.plan.vae_cuda is None
                       else self.cfg.upscale.timeout_seconds)
            buf = io.BytesIO()
            model_input.save(buf, "PNG", compress_level=1)
            data = await engine.upscale(buf.getvalue(), model=self.upscaler_path,
                                        tile_size=self.cfg.upscale.tile_size, timeout=timeout, on_wait=waiting)
        finally:
            self.active_jobs -= 1
        self.jobs_done += 1
        seconds = round(time.monotonic() - t, 1)
        await progress(0.9, "saving")
        out = Image.open(io.BytesIO(data))
        out.load()
        out = out.convert("RGB")
        if pad or pad_y:
            cut = round(pad * out.width / model_input.width)
            cut_y = round(pad_y * out.height / model_input.height)
            out = out.crop((cut, cut_y, out.width - cut, out.height - cut_y))
        if out.size != target:
            out = out.resize(target, Image.Resampling.LANCZOS)
        fmt = (output_format or (self.cfg.panorama.format if pano else self.cfg.outputs.format)).lower()
        if alpha_in:  # the model is RGB-only: scale the alpha channel separately
            alpha = src.convert("RGBA").getchannel("A").resize(target, Image.Resampling.LANCZOS)
            out = await asyncio.to_thread(transparency.with_alpha, out, alpha,
                                          hidden=self.cfg.transparency.hidden_pixels)
            if fmt == "jpeg":
                notes.append("JPEG cannot store transparency, saved as PNG instead")
                fmt = "png"
        if icc := imaging.rgb_icc(src):
            out.info["icc_profile"] = icc
        info = {"model": self.upscaler_name, "scale": scale, "input": f"{src.width}x{src.height}",
                "width": out.width, "height": out.height, "upscale_seconds": seconds,
                "tile_size": self.cfg.upscale.tile_size}
        xmp = None
        if pano:
            info["panorama"] = "360 (seamless wrap-around edges)"
            if sphere is not None:  # the input's own Photo Sphere packet, with the new pixel size
                xmp = panorama.resize_sphere_xmp(sphere, out.width, out.height)
                if len(xmp) > 60_000:  # a JPEG XMP segment holds about 64 KB
                    xmp = None
            if xmp is None and self.cfg.panorama.embed_gpano:
                xmp = panorama.gpano_xmp(out.width, out.height)
        elif tile:
            info["tileable"] = "seamless (wrap-around edges)"
            xmp = panorama.tile_xmp()
        saved = await asyncio.to_thread(self._save, out, fmt, "upscaled", xmp)
        if pano:
            saved.view_url = f"{self.base_url()}/view/{saved.filename}"
        return OpResult([saved], info, notes)

    # ------------------------------------------------------------ uploads / listing
    def save_upload(self, data: bytes, filename: str = "", hashed: bool = False) -> dict:
        """Store an uploaded image under outputs/uploads/ and return how to reference it.

        hashed: name the file after the sha256 of its bytes (uploads/chat/<hash>.<ext>), so uploading the same
        image again returns the same reference (used by chat proxies that re-send the whole conversation).
        """
        digest = hashlib.sha256(data).hexdigest()[:16]
        self._check_disk()
        if len(data) > self.cfg.server.max_request_mb * 1024 * 1024:
            raise ImageInputError(f"upload is larger than {self.cfg.server.max_request_mb} MB")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as exc:  # noqa: BLE001
            raise ImageInputError(f"not a readable image: {exc}") from exc
        fmt = (img.format or "").lower()
        ext = {"png": "png", "jpeg": "jpg", "webp": "webp"}.get(fmt)
        stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in Path(filename).stem)[:40].strip("-") or "upload"
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        folder = self.outputs / "uploads" / day
        folder.mkdir(parents=True, exist_ok=True)
        if ext is None:  # re-encode other formats (gif, bmp, tiff, ...) as PNG
            buf = io.BytesIO()
            (img.convert("RGBA") if "A" in img.getbands() or img.mode == "P" else img.convert("RGB")).save(buf, "PNG")
            data, ext = buf.getvalue(), "png"
        if hashed:
            folder = self.outputs / "uploads" / "chat"
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{digest}.{ext}"
            rel = f"uploads/chat/{name}"
            if (folder / name).exists():
                (folder / name).touch()  # keep it alive for the outputs.keep_days cleanup
                return {"file": rel, "url": f"{self.base_url()}/outputs/{rel}", "width": img.width,
                        "height": img.height}
        else:
            name = f"{stem}-{uuid.uuid4().hex[:8]}.{ext}"
            rel = f"uploads/{day}/{name}"
        (folder / name).write_bytes(data)
        return {"file": rel, "url": f"{self.base_url()}/outputs/{rel}", "width": img.width, "height": img.height}

    def list_images(self, limit: int = 30) -> dict:
        def entries(root: Path, prefix: str, url_base: str | None):
            out = []
            if not root.exists():
                return out
            for p in root.rglob("*"):
                rel = p.relative_to(root).as_posix()
                if servable(rel) and p.is_file():
                    st = p.stat()
                    out.append({"file": f"{prefix}{rel}", "url": f"{url_base}/{rel}" if url_base else None,
                                "bytes": st.st_size, "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                                .strftime("%Y-%m-%d %H:%M:%S")})
            out.sort(key=lambda e: e["modified"], reverse=True)
            out = out[:limit]
            if url_base:  # 360 panoramas (generated or upscaled) get a viewer link
                for e in out:
                    if _is_sphere_file(root / e["file"][len(prefix):]):
                        e["viewer_url"] = f"{self.base_url()}/view/{e['file']}"
            return out
        return {
            "outputs_and_uploads": entries(self.outputs, "", f"{self.base_url()}/outputs"),
            "inputs_folder": entries(Path("/inputs"), "", None),
            "how_to_use": "Pass the 'file' or 'url' value of an entry as an image to edit_image, generate_panorama "
                          "or remove_background. Upload new images at " + self.base_url() + "/upload",
        }

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        e = self.engine
        return {
            "state": self.state,
            "error": self.error,
            "device": self.cfg.device,
            "placement": self.plan.placement,
            "visible_gpus": self.plan.visible_gpus,
            "model": {"variant": self.cfg.model.variant, "quant": self.cfg.model.quant,
                      "text_encoder": self.cfg.model.text_encoder_quant},
            "download": self.store.status.as_dict(),
            "engine": e.state if e else "not started",
            "engine_progress": vars(e.progress) if e else None,
            "queue": e.queue.snapshot() if e and hasattr(e, "queue") else None,
            "background_removal": ({"model": self.cfg.transparency.matte_model, "provider": self.matter.provider}
                                   if self.matter else None),
            "upscale": ("disabled" if not self.cfg.upscale.enabled else
                        f"ready ({self.upscaler_name})" if self.upscaler_name else
                        self.upscaler_error or "not loaded yet"),
            "watermark_removal": ("disabled" if not self.cfg.watermark.enabled else
                                  "ready" if self.watermark_lora else self.watermark_error or "not loaded yet"),
            "defaults": {"size": "x".join(map(str, self.cfg.default_size)), "steps": self.cfg.default_steps,
                         "cfg_scale": self.cfg.default_cfg_scale,
                         "edit_cfg_scale": self.cfg.generation.edit_cfg_scale,
                         "schedule": self.cfg.generation.schedule,
                         "panorama_size": "x".join(map(str, self.cfg.panorama_size)),
                         "max_megapixels": self.cfg.max_pixels / 1e6},
            "jobs_done": self.jobs_done,
            "active_jobs": self.active_jobs,
            "uptime_s": int(time.time() - self.started_at),
        }
