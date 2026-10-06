"""Supervisor and client for the stable-diffusion.cpp ``sd-server`` process.

``sd-server`` keeps the model loaded between requests and runs one job at a
time. It listens only on 127.0.0.1 inside the container; the MCP server is the
only public entry point.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import functools
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from .config import Config
from .devices import DevicePlan

log = logging.getLogger("imagegen.sdserver")

# Credentials the engine never needs: they are kept out of its environment.
_SECRET_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def child_env(extra: dict[str, str]) -> dict[str, str]:
    """Environment for engine processes: the server's own, without credentials, plus `extra`."""
    env = {k: v for k, v in os.environ.items() if k not in _SECRET_ENV}
    env.update(extra)
    return env

_STEP_RE = re.compile(r"\|\s*(\d+)/(\d+) - ([\d.]+)(s/it|it/s)")
_CUDA_DEV_RE = re.compile(r"Device (\d+): (.+?), compute capability.*?VRAM: (\d+) MiB")

ProgressCallback = Callable[[float, str], Awaitable[None]]


class EngineError(RuntimeError):
    pass


@dataclass
class EngineProgress:
    phase: str = "idle"  # idle | encoding | sampling | decoding
    step: int = 0
    total: int = 0
    sec_per_step: float | None = None
    updated: float = field(default_factory=time.monotonic)


class SdServer:
    def __init__(self, cfg: Config, plan: DevicePlan, model_paths: dict[str, Path]):
        self.cfg = cfg
        self.plan = plan
        self.paths = model_paths
        self.base_url = f"http://127.0.0.1:{cfg.sd_server_port}"
        self.proc: asyncio.subprocess.Process | None = None
        self.state = "stopped"  # stopped | starting | ready | crashed
        self.progress = EngineProgress()
        self.log_tail: collections.deque[str] = collections.deque(maxlen=300)
        self.circular_acks = 0  # "Using circular padding" lines: sd-server applied circular_x / circular_y
        self.cuda_devices: list[str] = []
        self.vram_mib: dict[int, int] = {}  # cudaN -> total VRAM, from the engine's startup log
        self._reader: asyncio.Task | None = None
        self._empty_dir = Path(tempfile.mkdtemp(prefix="imagegen-empty-"))
        # No connection reuse: sd-server (cpp-httplib) closes keep-alive connections after a few
        # requests, and reusing one at that moment fails with "Server disconnected without sending a
        # response". Localhost connections are cheap.
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(120.0, connect=5.0),
                                         limits=httpx.Limits(max_keepalive_connections=0))
        self._restarts = 0
        self._stopping = False
        self._restart_lock = asyncio.Lock()
        # Idle unload: stop the engine (state "unloaded") after this long with no job; ensure_running()
        # starts it again on the next request. 0 disables it.
        self.idle_unload_seconds = cfg.generation.idle_unload_seconds
        self._active = 0                      # jobs currently inside generate() or upscale()
        self._gen_active = 0                  # generations submitted to sd-server and not finished
        self._upscale_lock = asyncio.Lock()   # held while sd-cli upscales; generations wait for it
        self._last_active = time.monotonic()
        self._idle_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ process
    def command(self) -> list[str]:
        p = self.paths
        return [
            str(self.cfg.sd_server_bin),
            "--diffusion-model", str(p["diffusion"]),
            "--vae", str(p["vae"]),
            "--llm", str(p["text_encoder"]),
            "--llm_vision", str(p["vision"]),
            "--listen-ip", "127.0.0.1",
            "--listen-port", str(self.cfg.sd_server_port),
            # sd-server scans these directories on every request; the default "." is "/"
            # in a container, which makes the native API fail with HTTP 500.
            "--lora-model-dir", str(self.lora_dir),
            # Never give sd-server an upscaler model: one in this folder slowed every denoising step by ~45%
            # (smaller graph-cut budget, 3 GB/s GPU->host traffic). upscale() runs sd-cli instead.
            "--hires-upscalers-dir", str(self._empty_dir),
            "--sampling-method", self.cfg.generation.sampler,
            "--cfg-scale", str(self.cfg.generation.cfg_scale),
            "--steps", str(self.cfg.default_steps),
            *self.plan.args,
            *(["--vae-conv-direct"] if self.vae_conv_direct else []),
            *(["--model-args", f"qwen_image_2_1_prefix_cache_type={self.cfg.generation.prefix_cache_type}"]
              if self.cfg.generation.prefix_cache_type not in ("", "auto") else []),
        ]

    @property
    def lora_dir(self) -> Path:
        """models/loras (few-step "turbo" LoRAs and others; requests name them by file name), if it exists."""
        d = Path(self.cfg.models_dir) / "loras"
        return d if d.is_dir() else self._empty_dir

    @property
    def vae_conv_direct(self) -> bool:
        mode = self.cfg.generation.vae_conv_direct
        return mode == "on" or (mode == "auto" and not self.cfg.is_cpu)

    @functools.cached_property
    def supports_circular(self) -> bool:
        """True when this sd-server build reads circular_x / circular_y from the request body (the Dockerfile
        lists the applied engine patches in MCP_PATCHES next to the binary). An unpatched build ignores the keys."""
        try:
            return "sdcpp-circular-json" in (self.cfg.sd_server_bin.parent / "MCP_PATCHES").read_text().split()
        except OSError:
            return False

    def vae_tiling_for(self, width: int, height: int) -> dict | None:
        """Tiled-decode parameters for images above generation.vae_tiling_above_megapixels, else None.

        A full decode of a 4 MP image needs far more memory than the tiles and was slower on a 12 GB card (it failed
        and fell back to tiling); 1024 px tiles decoded 2048x2048 in 15.7 s against 21.4 s with 512 px tiles."""
        g = self.cfg.generation
        if g.vae_tiling == "off" or (g.vae_tiling == "auto" and self.plan.vae_cuda is None):  # CPU: RAM is plentiful
            return None
        if g.vae_tiling == "auto" and width * height <= g.vae_tiling_above_megapixels * 1_000_000:
            return None
        return {"enabled": True, "tile_size_w": g.vae_tile_size, "tile_size_h": g.vae_tile_size,
                "target_overlap": 0.5}

    async def start(self) -> None:
        self._stopping = False
        env = child_env(self.plan.env)
        cmd = self.command()
        log.info("starting sd-server: %s", " ".join(cmd))
        self.state = "starting"
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            cwd=str(self._empty_dir),
        )
        self._reader = asyncio.create_task(self._read_output(self.proc))
        await self._wait_ready()

    async def _read_output(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        buf = b""
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            # progress bars redraw with \r, so split on both \r and \n
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").replace("\x1b[K", "").rstrip()
                if line:
                    self._handle_line(line)
        rc = await proc.wait()
        if not self._stopping:
            self.state = "crashed"
            log.error("sd-server exited with code %s. Last output:\n%s", rc, "\n".join(list(self.log_tail)[-25:]))

    def _handle_line(self, line: str) -> None:
        m = _STEP_RE.search(line)
        if m and self.progress.phase == "sampling":
            step, total, val, unit = int(m.group(1)), int(m.group(2)), float(m.group(3)), m.group(4)
            self.progress.step, self.progress.total = step, total
            self.progress.sec_per_step = val if unit == "s/it" else (1.0 / val if val else None)
            self.progress.updated = time.monotonic()
            return
        if "MB/s" in line and "|" in line:  # tensor loading progress bars
            return
        self.log_tail.append(line)
        if "Using circular padding for convolutions" in line:
            self.circular_acks += 1
        if "generate_image " in line and "completed" not in line:
            self.progress = EngineProgress(phase="encoding")
        elif "get_learned_condition completed" in line:
            self.progress.phase = "sampling"
        elif "sampling completed" in line:
            self.progress.phase = "decoding"
        elif "generate_image completed" in line:
            self.progress.phase = "idle"
        dm = _CUDA_DEV_RE.search(line)
        if dm:
            self.cuda_devices.append(f"cuda{dm.group(1)}: {dm.group(2)}")
            self.vram_mib[int(dm.group(1))] = int(dm.group(3))
        lvl = logging.DEBUG
        if "[ERROR" in line or "GGML_ASSERT" in line or "error:" in line.lower():
            lvl = logging.ERROR
        elif "[WARN" in line:
            lvl = logging.WARNING
        elif any(k in line for k in ("ggml_cuda_init", "Device ", "load_backend", "auto-fit", "listening",
                                      "generate_image completed", "loading diffusion", "loading llm")):
            lvl = logging.INFO
        log.log(lvl, "[sd] %s", line)

    async def _wait_ready(self, timeout: float = 900.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is None or self.proc.returncode is not None:
                tail = "\n".join(list(self.log_tail)[-25:])
                raise EngineError(f"sd-server exited during startup:\n{tail}")
            try:
                r = await self._client.get("/sdcpp/v1/capabilities", timeout=5.0)
                if r.status_code == 200:
                    self.state = "ready"
                    self._last_active = time.monotonic()  # a fresh (re)load starts a new idle window
                    log.info("sd-server is ready")
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
        raise EngineError("sd-server did not become ready in time")

    async def stop(self) -> None:
        self._stopping = True
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 20)
            except asyncio.TimeoutError:
                self.proc.kill()
        self.state = "stopped"
        await self._client.aclose()

    async def ensure_running(self) -> None:
        if self.state == "ready" and self.proc and self.proc.returncode is None:
            return
        async with self._restart_lock:
            if self.state == "ready" and self.proc and self.proc.returncode is None:
                return
            if self.state == "unloaded":
                log.info("reloading sd-server on demand (was unloaded after %ss idle)", self.idle_unload_seconds)
                t = time.monotonic()
                await self.start()
                log.info("sd-server reloaded in %.1f s", time.monotonic() - t)
                return
            if self.state == "crashed":
                self._restarts += 1
                wait = min(30, 2 ** min(self._restarts, 5))
                log.warning("restarting sd-server in %ss (restart #%d)", wait, self._restarts)
                await asyncio.sleep(wait)
                await self.start()
                return
        if self.state != "ready":
            raise EngineError(f"inference engine is {self.state}")

    # ------------------------------------------------------------------ idle unload
    def start_idle_watch(self) -> None:
        """Start the background idle-unload watcher once (no-op if idle_unload_seconds is 0)."""
        if self._idle_task is not None or self.idle_unload_seconds <= 0:
            return
        self._idle_task = asyncio.create_task(self._idle_watch_loop())
        log.info("idle unload enabled: the engine stops after %ss without a job and restarts on demand",
                 self.idle_unload_seconds)

    async def _idle_watch_loop(self) -> None:
        interval = max(10.0, min(self.idle_unload_seconds / 4.0, 60.0))
        while True:
            await asyncio.sleep(interval)
            try:
                if (self.state == "ready" and self._active == 0
                        and time.monotonic() - self._last_active >= self.idle_unload_seconds):
                    await self.unload()
            except Exception:  # noqa: BLE001 - the watcher must never die
                log.exception("idle unload check failed")

    async def unload(self) -> bool:
        """Stop the engine to free its VRAM. Re-checks under the restart lock that it is still idle,
        so it never stops while a job is running or right after one started."""
        async with self._restart_lock:
            idle = time.monotonic() - self._last_active
            if self.state != "ready" or self._active > 0 or idle < self.idle_unload_seconds:
                return False
            self.state = "unloading"
            self._stopping = True               # tells _read_output this exit is intentional
            if self.proc and self.proc.returncode is None:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), 20)
                except asyncio.TimeoutError:
                    self.proc.kill()
                    await self.proc.wait()
            self.state = "unloaded"
            log.info("sd-server unloaded after %.0f s idle (VRAM freed; reloads on the next request)", idle)
            return True

    # ------------------------------------------------------------------ jobs
    async def _get(self, path: str) -> httpx.Response:
        """GET with a few retries: polling is idempotent, so transient transport errors are retried."""
        for attempt in range(4):
            try:
                return await self._client.get(path)
            except httpx.TransportError:
                if attempt == 3:
                    raise
                await asyncio.sleep(0.3 * (attempt + 1))
        raise AssertionError("unreachable")

    async def generate(self, body: dict, *, timeout: float, on_progress: ProgressCallback | None = None) -> list[bytes]:
        """Submit a native img_gen job and wait for the resulting PNG bytes."""
        self._active += 1                     # counted before ensure_running(), so unload() sees it
        self._gen_active += 1
        try:
            return await self._generate(body, timeout=timeout, on_progress=on_progress)
        except httpx.HTTPError as exc:
            raise EngineError(f"lost the connection to the inference engine ({exc.__class__.__name__}: {exc})") \
                from exc
        finally:
            self._active -= 1
            self._gen_active -= 1
            self._last_active = time.monotonic()  # the idle clock starts when the job ends

    async def upscale(self, image_png: bytes, *, model: Path, tile_size: int, timeout: float,
                      wait_limit: float = 3600.0, on_wait: ProgressCallback | None = None) -> bytes:
        """Upscale a PNG with an ESRGAN model by running sd-cli (-M upscale) as a short-lived process on the VAE's
        device: it loads the 33-67 MB model, upscales and exits, so nothing stays in VRAM. It first waits (up to
        wait_limit) for running generations, and new generations wait for it, so the two never share the card;
        ``timeout`` covers only the upscale, and the process is killed when it runs out."""
        async with self._upscale_lock:
            waited = 0.0
            while self._gen_active > 0:
                if waited >= wait_limit:
                    raise EngineError(f"the engine was busy with other images for {int(wait_limit)} s; try again later")
                if on_wait and waited % 5 < 0.5:
                    await on_wait("waiting for the running image job to finish")
                await asyncio.sleep(0.5)
                waited += 0.5
            with tempfile.TemporaryDirectory(prefix="imagegen-upscale-") as d:
                src, dst = Path(d) / "in.png", Path(d) / "out.png"
                src.write_bytes(image_png)
                cmd = [str(self.cfg.sd_server_bin.with_name("sd-cli")), "-M", "upscale", "--upscale-model", str(model),
                       "-i", str(src), "-o", str(dst), "--upscale-tile-size", str(int(tile_size)),
                       "--backend", self.plan.upscale_backend, "-t", str(self.plan.threads or 4)]
                env = child_env(self.plan.env)
                proc = await asyncio.create_subprocess_exec(*cmd, env=env, cwd=d, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.STDOUT)
                try:
                    out, _ = await asyncio.wait_for(proc.communicate(), timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    raise EngineError(f"the upscale did not finish in {int(timeout)} s and was stopped") from None
                if proc.returncode != 0 or not dst.exists():
                    tail = " | ".join(out.decode("utf-8", "replace").strip().splitlines()[-3:])
                    raise EngineError(f"the upscaler failed (exit code {proc.returncode}): {tail[:400]}")
                return dst.read_bytes()

    async def _generate(self, body: dict, *, timeout: float, on_progress: ProgressCallback | None) -> list[bytes]:
        while self._upscale_lock.locked():  # never share the card with a running upscale
            await asyncio.sleep(0.5)
        await self.ensure_running()
        r = await self._client.post("/sdcpp/v1/img_gen", json=body)
        if r.status_code == 429:
            raise EngineError("the generation queue is full, try again later")
        if r.status_code >= 400:
            raise EngineError(f"sd-server rejected the request (HTTP {r.status_code}): {r.text[:500]}")
        job = r.json()
        job_id = job["id"]
        queued_since = time.monotonic()
        started: float | None = None  # the timeout counts running time, not the wait behind other jobs
        last_report = 0.0
        try:
            while True:
                if self.proc is None or self.proc.returncode is not None:
                    raise EngineError("the inference engine crashed while generating; see server logs")
                jr = await self._get(f"/sdcpp/v1/jobs/{job_id}")
                if jr.status_code == 410:
                    raise EngineError("job result expired before it was collected")
                if jr.status_code != 200:
                    raise EngineError(f"job status HTTP {jr.status_code}: {jr.text[:300]}")
                j = jr.json()
                status = j.get("status")
                if status == "completed":
                    images = (j.get("result") or {}).get("images") or []
                    if not images:
                        raise EngineError("the job completed without images")
                    return [base64.b64decode(img["b64_json"]) for img in images]
                if status in ("failed", "cancelled"):
                    err = j.get("error") or {}
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise EngineError(f"generation {status}: {msg or 'unknown error'}")
                now = time.monotonic()
                if status == "queued":
                    # backstop for a stuck engine: nothing ahead of this job has made progress for a whole timeout
                    if now - max(queued_since, self.progress.updated) > timeout:
                        raise EngineError(f"the engine made no progress for {int(timeout)} s while this job was queued")
                elif started is None:
                    started = now
                if started is not None and now - started > timeout:
                    raise EngineError(f"generation timed out after {int(timeout)} s")
                if on_progress and now - last_report >= 1.0:
                    last_report = now
                    await on_progress(*self._describe(status, j))
                await asyncio.sleep(0.5)
        except (asyncio.CancelledError, EngineError):
            # The client went away or we gave up: drop the job if it is still queued.
            try:
                await self._client.post(f"/sdcpp/v1/jobs/{job_id}/cancel", timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
            raise

    def _describe(self, status: str, job: dict) -> tuple[float, str]:
        if status == "queued":
            pos = job.get("queue_position")
            return 0.0, f"queued (position {pos})" if pos is not None else "queued"
        p = self.progress
        if p.phase == "sampling" and p.total:
            eta = ""
            if p.sec_per_step:
                eta = f", ~{int((p.total - p.step) * p.sec_per_step)} s left"
            return 0.05 + 0.9 * p.step / p.total, f"denoising step {p.step}/{p.total}{eta}"
        if p.phase == "decoding":
            return 0.96, "decoding image"
        return 0.02, "encoding prompt"

    async def list_devices(self) -> str:
        env = child_env(self.plan.env)
        proc = await asyncio.create_subprocess_exec(
            str(self.cfg.sd_server_bin), "--list-devices", env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        return out.decode("utf-8", "replace")
