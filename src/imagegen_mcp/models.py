"""Model file registry and a resumable, checksum-verified downloader.

Every file is pinned to a repository revision and a SHA256 so a fresh install
always gets the exact weights this project was tested with. Files live under
the project's ``models/`` folder (mounted at ``/models``)::

    models/
      diffusion_models/   qwen-image-2.1-UC-Q4_K_M.gguf (or Q6_K / Q8_0 / ...)
      text_encoders/      Qwen3VL-8B-Instruct-Q4_K_M.gguf + mmproj-...gguf
      vae/                texture_fix_vae_for_qwen_image_2.1_bf16.safetensors (or the original VAE)
      background_removal/ birefnet-general-lite.onnx
      loras/              watermark_remover_v2_qwen.safetensors (remove_watermark tool)
      upscalers/          4xNomos2_otf_esrgan.safetensors (upscale_image tool)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import Config

log = logging.getLogger("imagegen.models")

DIFFUSION_REPO = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
DIFFUSION_REVISIONS = {
    "uncensored": "1206d38bb47ef93961bfb77bc2c700d43a25860e",  # branch main
    "base": "4db8efdbe3426175988fca2ac50670f35ccb7ec9",  # branch base
}
TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
TEXT_ENCODER_REVISION = "f982a07559d4a2f6c8744d840bf6fccab30eea96"
TEXTURE_FIX_VAE_REPO = "madebyollin/texture-fix-vae-for-qwen-image-2.1"
TEXTURE_FIX_VAE_REVISION = "1f80456b6de9d89043c133ff1f3875449b8bd016"
REMBG_RELEASE = "https://github.com/danielgatis/rembg/releases/download/v0.0.0"
# Civitai model 2969142, version 3364468 ("v1.0 Qwen 2.1"), by saladin.
WATERMARK_LORA_URL = "https://civitai.com/api/download/models/3364468?fileId=3252344"


@dataclass(frozen=True)
class ModelFile:
    key: str
    url: str
    dest: str  # path relative to the models dir
    size: int
    sha256: str | None
    description: str
    license: str
    # Optional files are downloaded in the background after the server is ready (ModelStore.fetch_optional); a
    # failed download disables the feature that needs them instead of stopping or delaying the server.
    optional: bool = False

    @property
    def filename(self) -> str:
        return os.path.basename(self.dest)


def _hf_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


# (variant, quant) -> (filename, size, sha256)
_DIFFUSION = {
    ("uncensored", "BF16"): ("qwen-image-2.1-UC-BF16.gguf", 14230272800, "f151c683a8aed4b310777017ebbbe3f2180f1180f7867115171adb7d50b0762a"),
    ("uncensored", "Q4_0"): ("qwen-image-2.1-UC-Q4_0.gguf", 4151573280, "13f59f20656efc0aa385d03c1fcac1a9dc2ad6e5ccc0ea9bfe1d6ac636f2c5b9"),
    ("uncensored", "Q4_K_M"): ("qwen-image-2.1-UC-Q4_K_M.gguf", 4604558112, "e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41"),
    ("uncensored", "Q5_K_M"): ("qwen-image-2.1-UC-Q5_K_M.gguf", 5221284640, "af0bf278cf16d204fb31c384dc82fd41dca82d976b15fe9305a60c726fd6f821"),
    ("uncensored", "Q6_K"): ("qwen-image-2.1-UC-Q6_K.gguf", 5876556576, "e14bb312109333b3d73b92ad9b9ac8b29b51f2edca1e7b86d1af1981bf1c4ee3"),
    ("uncensored", "Q8_0"): ("qwen-image-2.1-UC-Q8_0.gguf", 7591557920, "cde456c72ea3ecebfc1be783300e972711d875e0c5f1bed33d42b66b156affa8"),
    ("base", "Q4_0"): ("qwen-image-2.1-Q4_0.gguf", 4050899616, "8efd261419f4d60bf0eda8ae86a656e229ed440a248a22e20bf1f1f9606f6124"),
    ("base", "Q4_K_M"): ("qwen-image-2.1-Q4_K_M.gguf", 4604557984, "833439e91bc1152d28f37aa198c7f6f4218b7de95754c2f7a318a2422ab4b2f8"),
    ("base", "Q5_K_M"): ("qwen-image-2.1-Q5_K_M.gguf", 5221284512, "88ce8e90e5b959cce5e248f697d7f6c9c7ca5696c1eac64a10dadb041dd7fd07"),
    ("base", "Q6_K"): ("qwen-image-2.1-Q6_K.gguf", 5876556448, "a3a0d39bb03cda26302fc048b49d019baaea1c381cbda3994f6f2a7826344fb9"),
    ("base", "Q8_0"): ("qwen-image-2.1-Q8_0.gguf", 7591557792, "9a7ec02f4c9d5cf5b78e8efa6ccd81ec0c35a898defbf8e77ac7c02a32fe0d7e"),
}

_TEXT_ENCODER = {
    "Q4_K_M": ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5027784800, "67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2"),
    "Q8_0": ("Qwen3VL-8B-Instruct-Q8_0.gguf", 8709519456, "0d264b3941185d00a74f75c4245521dae088ff1efc90ab8d1754e83f5844adb0"),
    "F16": ("Qwen3VL-8B-Instruct-F16.gguf", 16388044896, "2715c1a097f1943fb88ad59c7c0e9288a7a50c84d60cf18c6227bd9a40520972"),
}

_VISION = {
    "F16": ("mmproj-Qwen3VL-8B-Instruct-F16.gguf", 1159029824, "ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38"),
    "Q8_0": ("mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", 752289728, "c6ba85508d82f42590e6eb77d5340369ab6fecf107a7561d809523d8aa5f3bfd"),
}

_VAE = ("vae/qwen_image_2.1_vae_bf16.safetensors", 675509688, "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9")
# Decoder-only fine-tune of the same VAE: same tensors and latent space, no 2-pixel checkerboard.
_TEXTURE_FIX_VAE = ("texture_fix_vae_for_qwen_image_2.1_bf16.safetensors", 675509688,
                    "05e9af8da4697d1a5118b674c3d90e95f932702d2c5a8900d7d4f05076403cc7")

# name -> (release asset, size, sha256)
_MATTE = {
    "birefnet-general-lite": ("BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", 224005088,
                              "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333"),
    "birefnet-general": ("BiRefNet-general-epoch_244.onnx", 972666916,
                         "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c"),
    "isnet-general-use": ("isnet-general-use.onnx", 178648008,
                          "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"),
}


def diffusion_file(cfg: Config) -> ModelFile:
    name, size, sha = _DIFFUSION[(cfg.model.variant, cfg.model.quant)]
    rev = DIFFUSION_REVISIONS[cfg.model.variant]
    return ModelFile("diffusion", _hf_url(DIFFUSION_REPO, rev, name), f"diffusion_models/{name}", size, sha,
                     f"Qwen-Image-2.1 {cfg.model.variant} {cfg.model.quant} diffusion transformer",
                     "Qwen Research License")


def text_encoder_file(cfg: Config) -> ModelFile:
    name, size, sha = _TEXT_ENCODER[cfg.model.text_encoder_quant]
    return ModelFile("text_encoder", _hf_url(TEXT_ENCODER_REPO, TEXT_ENCODER_REVISION, name),
                     f"text_encoders/{name}", size, sha,
                     f"Qwen3-VL-8B-Instruct {cfg.model.text_encoder_quant} text encoder", "Apache-2.0")


def vision_file(cfg: Config) -> ModelFile:
    name, size, sha = _VISION[cfg.model.vision_quant]
    return ModelFile("vision", _hf_url(TEXT_ENCODER_REPO, TEXT_ENCODER_REVISION, name),
                     f"text_encoders/{name}", size, sha,
                     f"Qwen3-VL-8B vision projector {cfg.model.vision_quant} (needed for editing)", "Apache-2.0")


def vae_file(cfg: Config) -> ModelFile:
    if cfg.model.vae == "texture-fix":
        name, size, sha = _TEXTURE_FIX_VAE
        return ModelFile("vae", _hf_url(TEXTURE_FIX_VAE_REPO, TEXTURE_FIX_VAE_REVISION, name), f"vae/{name}",
                         size, sha, "Texture-Fix VAE for Qwen-Image-2.1 (no checkerboard)", "Qwen Research License")
    path, size, sha = _VAE
    rev = DIFFUSION_REVISIONS["uncensored"]
    return ModelFile("vae", _hf_url(DIFFUSION_REPO, rev, path), path, size, sha,
                     "Qwen-Image-2.1 RGBA VAE", "Qwen Research License")


def matte_file(cfg: Config) -> ModelFile:
    name = cfg.transparency.matte_model
    asset, size, sha = _MATTE[name]
    return ModelFile("background_removal", f"{REMBG_RELEASE}/{asset}", f"background_removal/{name}.onnx",
                     size, sha, f"{name} background-removal model (ONNX)", "MIT")


def watermark_lora_file() -> ModelFile:
    return ModelFile("watermark_lora", WATERMARK_LORA_URL, "loras/watermark_remover_v2_qwen.safetensors", 79744288,
                     "29fe23b60fde79596aa28ee7080a4cd98c2cfe8055f22753ccf51bae502df5a1",
                     "Watermark removal LoRA for Qwen-Image-2.1 (remove_watermark tool)",
                     "Civitai model license (no commercial use)", optional=True)


# name -> (url, file name, size, sha256, license). ESRGAN (RRDBNet) only: the engine has no other upscaler architecture.
_UPSCALERS = {
    "4xNomos2_otf_esrgan": (
        "https://huggingface.co/Phips/4xNomos2_otf_esrgan/resolve/38e10465ba4f7da9be7259e39865ee4f218ec4ed/"
        "4xNomos2_otf_esrgan.safetensors", "4xNomos2_otf_esrgan.safetensors", 33467822,
        "12db878907ed3a52ee97de552dfc0ce7ffd38559ca9760a7803cb2d06a737055", "CC-BY-4.0 (Philip Hofmann)"),
    "RealESRGAN_x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus.pth", 67040989, "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "BSD-3-Clause (Xintao Wang)"),
}


def upscaler_file(cfg: Config) -> ModelFile:
    url, name, size, sha, lic = _UPSCALERS[cfg.upscale.model]
    return ModelFile("upscaler", url, f"upscalers/{name}", size, sha,
                     f"{cfg.upscale.model} 4x image upscaler (upscale_image tool)", lic, optional=True)


def required_files(cfg: Config) -> list[ModelFile]:
    files = [diffusion_file(cfg), text_encoder_file(cfg), vision_file(cfg), vae_file(cfg)]
    if cfg.transparency.method in ("hybrid", "matte"):
        files.append(matte_file(cfg))
    if cfg.watermark.enabled:
        files.append(watermark_lora_file())
    if cfg.upscale.enabled:
        files.append(upscaler_file(cfg))
    return files


class DownloadStatus:
    """Shared progress state, exposed by /health and the server_info tool."""

    def __init__(self) -> None:
        self.state = "pending"  # pending | checking | downloading | ready | error
        self.current: str | None = None
        self.done_bytes = 0
        self.total_bytes = 0
        self.error: str | None = None
        self.files: dict[str, str] = {}

    def as_dict(self) -> dict:
        pct = round(100.0 * self.done_bytes / self.total_bytes, 1) if self.total_bytes else None
        return {"state": self.state, "current_file": self.current, "percent": pct,
                "downloaded_gb": round(self.done_bytes / 1e9, 2), "total_gb": round(self.total_bytes / 1e9, 2),
                "files": self.files, "error": self.error}


def is_hugging_face(url: str) -> bool:
    """True only for https URLs on huggingface.co, the one host the optional HF token is sent to."""
    u = urlparse(url)
    host = (u.hostname or "").lower()
    return u.scheme == "https" and (host == "huggingface.co" or host.endswith(".huggingface.co"))


class _PermanentError(RuntimeError):
    """A download error that retrying cannot fix (e.g. HTTP 401/403/404)."""


class ModelStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = Path(cfg.models_dir)
        self.status = DownloadStatus()
        self._verified_path = self.root / ".verified.json"
        self.pending_optional: list[ModelFile] = []  # missing optional files, for fetch_optional()

    def path(self, f: ModelFile) -> Path:
        return self.root / f.dest

    # --- verification cache: re-hashing 5-15 GB on every start is slow ------
    def _load_verified(self) -> dict:
        try:
            return json.loads(self._verified_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_verified(self, data: dict) -> None:
        try:
            tmp = self._verified_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, self._verified_path)
        except OSError as exc:  # read-only mount is fine
            log.debug("could not write %s: %s", self._verified_path, exc)

    def _is_verified(self, f: ModelFile, p: Path, cache: dict) -> bool:
        st = p.stat()
        if st.st_size != f.size:
            return False
        if not self.cfg.model.verify_checksums or not f.sha256:
            return True
        entry = cache.get(f.dest)
        return bool(entry and entry.get("sha256") == f.sha256 and entry.get("size") == st.st_size
                    and abs(entry.get("mtime", 0) - st.st_mtime) < 2)

    def _mark_verified(self, f: ModelFile, p: Path, cache: dict) -> None:
        st = p.stat()
        cache[f.dest] = {"sha256": f.sha256, "size": st.st_size, "mtime": st.st_mtime}
        self._save_verified(cache)

    @staticmethod
    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            while chunk := fh.read(8 * 1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    async def ensure(self) -> dict[str, Path]:
        """Check every model file and download the missing required ones. Missing optional files are only
        noted in ``pending_optional`` and left out of the result; fetch_optional() gets them later."""
        files = required_files(self.cfg)
        self.status.state = "checking"
        self.status.total_bytes = sum(f.size for f in files)
        cache = self._load_verified()
        paths: dict[str, Path] = {}
        missing: list[ModelFile] = []
        for f in files:
            p = self.path(f)
            paths[f.key] = p
            if p.exists():
                if self._is_verified(f, p, cache):
                    self.status.files[f.dest] = "ok"
                    self.status.done_bytes += f.size
                    continue
                if p.stat().st_size == f.size and f.sha256 and self.cfg.model.verify_checksums:
                    self.status.current = f.dest
                    log.info("verifying checksum of %s (first start only)", f.dest)
                    digest = await asyncio.to_thread(self._sha256, p)
                    if digest == f.sha256:
                        self._mark_verified(f, p, cache)
                        self.status.files[f.dest] = "ok"
                        self.status.done_bytes += f.size
                        continue
                    log.warning("%s has a wrong checksum, downloading it again", f.dest)
                    p.unlink()
                elif p.stat().st_size != f.size:
                    log.warning("%s has size %d, expected %d; downloading again", f.dest, p.stat().st_size, f.size)
                    p.unlink()
            missing.append(f)

        self.pending_optional = []
        for f in [f for f in missing if f.optional]:
            missing.remove(f)
            paths.pop(f.key, None)
            self.status.total_bytes -= f.size
            if self.cfg.model.auto_download:
                self.pending_optional.append(f)
                self.status.files[f.dest] = "pending"
            else:
                log.warning("%s is missing and model.auto_download is false; continuing without it (%s)",
                            f.dest, f.description)
                self.status.files[f.dest] = "missing"
        if missing and not self.cfg.model.auto_download:
            names = ", ".join(f.dest for f in missing)
            raise RuntimeError(f"missing model files and model.auto_download is false: {names}")
        if missing:
            need = sum(f.size for f in missing)
            free = shutil.disk_usage(self.root).free
            if free < need + 512 * 1024 * 1024:
                raise RuntimeError(f"not enough disk space in {self.root}: need {need / 1e9:.1f} GB, "
                                   f"free {free / 1e9:.1f} GB")
            self.status.state = "downloading"
            log.info("downloading %d file(s), %.1f GB total", len(missing), need / 1e9)
            for f in missing:
                await self._download(f, cache)

        self.status.state = "ready"
        self.status.current = None
        return paths

    async def fetch_optional(self) -> dict[str, Path]:
        """Download the optional files ensure() found missing. Failures are logged, not raised: a few quick
        attempts only, so an unreachable host does not keep retrying for minutes."""
        paths: dict[str, Path] = {}
        cache = self._load_verified()
        while self.pending_optional:
            f = self.pending_optional[0]
            try:
                free = shutil.disk_usage(self.root).free
                if free < f.size + 512 * 1024 * 1024:
                    raise RuntimeError(f"not enough disk space in {self.root} for {f.dest} ({free / 1e6:.0f} MB free)")
                self.status.total_bytes += f.size
                await self._download(f, cache, max_attempts=3)
                paths[f.key] = self.path(f)
            except (RuntimeError, OSError) as exc:
                log.warning("%s; continuing without it (%s)", exc, f.description)
                self.status.files[f.dest] = "error"
            finally:
                self.pending_optional.pop(0)
                self.status.current = None
        return paths

    async def _download(self, f: ModelFile, cache: dict, max_attempts: int = 8) -> None:
        dest = self.path(f)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        self.status.current = f.dest
        self.status.files[f.dest] = "downloading"
        headers = {"User-Agent": "imagegen-mcp/1.0"}
        if self.cfg.model.hf_token and is_hugging_face(f.url):
            # httpx drops this header when a download redirects to another host (the file CDN).
            headers["Authorization"] = f"Bearer {self.cfg.model.hf_token}"
        base_done = self.status.done_bytes
        attempts = 0
        while True:
            attempts += 1
            have = part.stat().st_size if part.exists() else 0
            if have > f.size:
                part.unlink()
                have = 0
            h = dict(headers)
            if have:
                h["Range"] = f"bytes={have}-"
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=30.0)) as client:
                    async with client.stream("GET", f.url, headers=h) as resp:
                        if resp.status_code == 416:  # already complete
                            break
                        if resp.status_code not in (200, 206):
                            err = _PermanentError if 400 <= resp.status_code < 500 and resp.status_code not in (
                                408, 429) else RuntimeError
                            raise err(f"HTTP {resp.status_code} for {f.url.split('?')[0]}")
                        mode = "ab" if (have and resp.status_code == 206) else "wb"
                        if mode == "wb":
                            have = 0
                        log.info("downloading %s (%.2f GB)%s", f.dest, f.size / 1e9,
                                 f", resuming at {have / 1e9:.2f} GB" if have else "")
                        last_log = time.monotonic()
                        with open(part, mode) as fh:
                            async for chunk in resp.aiter_bytes(4 * 1024 * 1024):
                                fh.write(chunk)
                                have += len(chunk)
                                self.status.done_bytes = base_done + have
                                if time.monotonic() - last_log > 10:
                                    last_log = time.monotonic()
                                    log.info("  %s: %.1f%% (%.2f / %.2f GB)", f.filename,
                                             100.0 * have / f.size, have / 1e9, f.size / 1e9)
                break
            except (httpx.HTTPError, OSError, RuntimeError) as exc:
                if attempts >= max_attempts or isinstance(exc, _PermanentError):
                    self.status.files[f.dest] = "error"
                    raise RuntimeError(f"download of {f.dest} failed: {exc}") from exc
                wait = min(60, 2 ** attempts)
                log.warning("download error for %s (%s), retrying in %ss", f.dest, exc, wait)
                await asyncio.sleep(wait)

        size = part.stat().st_size
        if size != f.size:
            raise RuntimeError(f"{f.dest}: downloaded {size} bytes, expected {f.size}")
        if f.sha256 and self.cfg.model.verify_checksums:
            log.info("verifying checksum of %s", f.dest)
            digest = await asyncio.to_thread(self._sha256, part)
            if digest != f.sha256:
                part.unlink()
                raise RuntimeError(f"{f.dest}: checksum mismatch (got {digest}, expected {f.sha256})")
        os.replace(part, dest)
        self._mark_verified(f, dest, cache)
        self.status.files[f.dest] = "ok"
        self.status.done_bytes = base_done + f.size
        log.info("finished %s", f.dest)
