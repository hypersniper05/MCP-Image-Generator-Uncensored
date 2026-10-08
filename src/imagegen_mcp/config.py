"""Configuration loading and validation.

All settings live in one YAML file (``config.yaml`` in the project root, mounted
into the container at ``/app/config.yaml``). A few values can be overridden by
environment variables so the same file works in several deployments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GpuRef = Union[int, Literal["cpu"]]

DIFFUSION_QUANTS = ("Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16")
TEXT_ENCODER_QUANTS = ("Q4_K_M", "Q8_0", "F16")
VISION_QUANTS = ("F16", "Q8_0")
MATTE_MODELS = ("birefnet-general-lite", "birefnet-general", "isnet-general-use")

_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX*]\s*(\d+)\s*$")


def parse_size(value: str) -> tuple[int, int]:
    m = _SIZE_RE.match(str(value))
    if not m:
        raise ValueError(f"invalid size {value!r}, expected WIDTHxHEIGHT such as 1024x1024")
    return int(m.group(1)), int(m.group(2))


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(_Section):
    host: str = "0.0.0.0"
    port: int = Field(5005, ge=1, le=65535)
    path: str = "/mcp"
    public_url: str = ""
    max_request_mb: int = Field(64, ge=1, le=1024)
    allowed_hosts: list[str] = Field(default_factory=list)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        return "/" + v.strip("/")

    @field_validator("public_url")
    @classmethod
    def _public_url(cls, v: str) -> str:
        return v.strip().rstrip("/")


class ModelConfig(_Section):
    quant: Literal["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16"] = "Q4_K_M"
    variant: Literal["uncensored", "base"] = "uncensored"
    text_encoder_quant: Literal["Q4_K_M", "Q8_0", "F16"] = "Q4_K_M"
    vision_quant: Literal["F16", "Q8_0"] = "F16"
    # texture-fix: a decoder fine-tune without the stock VAE's 2-pixel checkerboard; original: the stock VAE
    vae: Literal["texture-fix", "original"] = "texture-fix"
    auto_download: bool = True
    verify_checksums: bool = True
    hf_token: str = ""

    @field_validator("quant", mode="before")
    @classmethod
    def _norm_quant(cls, v):
        s = str(v).strip().upper()
        aliases = {"Q4": "Q4_K_M", "4Q": "Q4_K_M", "Q5": "Q5_K_M", "5Q": "Q5_K_M", "Q6": "Q6_K",
                   "6Q": "Q6_K", "Q8": "Q8_0", "8Q": "Q8_0", "Q4_K": "Q4_K_M", "Q5_K": "Q5_K_M"}
        return aliases.get(s, s)

    @field_validator("text_encoder_quant", "vision_quant", mode="before")
    @classmethod
    def _upper(cls, v):
        return str(v).strip().upper()

    @model_validator(mode="after")
    def _check_variant(self):
        if self.variant == "base" and self.quant == "BF16":
            raise ValueError("model.quant BF16 is only published for variant 'uncensored'")
        return self


class GpuConfig(_Section):
    diffusion: list[int] = Field(default_factory=lambda: [0])
    text_encoder: GpuRef = 0
    vae: GpuRef = 0
    background_removal: GpuRef = "cpu"
    split_mode: Literal["layer", "row"] = "layer"
    max_vram_gb: dict[int, float] = Field(default_factory=dict)
    offload: Union[Literal["auto", "none", "cpu", "disk"], dict[str, Literal["gpu", "cpu", "disk"]]] = "auto"
    # true: flash attention in every module (--fa); "diffusion": only the diffusion model (--diffusion-fa)
    flash_attention: Union[bool, Literal["diffusion"]] = True

    @field_validator("offload")
    @classmethod
    def _offload_parts(cls, v):
        if isinstance(v, dict):
            bad = set(v) - {"diffusion", "text_encoder", "vae"}
            if bad:
                raise ValueError(f"gpu.offload: unknown part(s) {sorted(bad)}; use diffusion, text_encoder, vae")
        return v

    @field_validator("diffusion", mode="before")
    @classmethod
    def _diffusion_list(cls, v):
        if isinstance(v, (int, str)):
            v = [v]
        out = []
        for item in v:
            if isinstance(item, str) and item.strip().lower().startswith("cuda"):
                item = item.strip()[4:]
            out.append(int(item))
        if not out:
            raise ValueError("gpu.diffusion needs at least one GPU index")
        if len(set(out)) != len(out):
            raise ValueError("gpu.diffusion lists the same GPU twice")
        return out

    @field_validator("text_encoder", "vae", "background_removal", mode="before")
    @classmethod
    def _gpu_ref(cls, v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "cpu":
                return "cpu"
            if s.startswith("cuda"):
                s = s[4:]
            return int(s)
        return v

    @field_validator("diffusion")
    @classmethod
    def _non_negative(cls, v):
        if any(i < 0 for i in v):
            raise ValueError("GPU indices must be >= 0")
        return v


class CpuConfig(_Section):
    threads: int = Field(0, ge=0)


class GenerationConfig(_Section):
    steps: int = Field(40, ge=1, le=150)
    cpu_steps: int = Field(12, ge=1, le=150)
    # Guidance. Above 1 the negative prompt takes effect and each step costs twice as much.
    cfg_scale: float = Field(4.0, ge=0.0, le=30.0)
    cpu_cfg_scale: float = Field(1.0, ge=0.0, le=30.0)
    edit_cfg_scale: float = Field(4.0, ge=0.0, le=30.0)
    # Built-in negative prompts, used whenever guidance is above 1. A negative_prompt passed to a tool is
    # added to these, not swapped in.
    negative_prompt: str = "blurry, oversaturated, overexposed, high contrast, oversharpened, low quality"
    edit_negative_prompt: str = "blurry, oversaturated, overexposed, high contrast, oversharpened, low quality"
    sampler: str = "euler"
    scheduler: str = ""
    # Size the tool descriptions tell the model to use for final images. "xl" (native 2K) is the sharpest but
    # takes minutes on mid-range GPUs; "large" (~2 MP) or "medium" (~1 MP) suit slower cards.
    recommended_size: Literal["xl", "large", "medium"] = "xl"
    default_size: str = "1024x1024"
    cpu_default_size: str = "512x512"
    panorama_size: str = "2048x1024"
    cpu_panorama_size: str = "1024x512"
    max_megapixels: float = Field(4.2, gt=0.05, le=16.0)
    cpu_max_megapixels: float = Field(1.1, gt=0.05, le=16.0)
    ref_max_megapixels: float = Field(1.05, gt=0.05, le=16.0)
    max_reference_images: int = Field(10, ge=1, le=16)
    timeout_seconds: int = Field(1800, ge=30)
    cpu_timeout_seconds: int = Field(7200, ge=30)
    warmup: bool = True
    wait_seconds: int = Field(50, ge=0, le=3600)
    # Stop the inference engine after this many seconds with no job, freeing all of its VRAM; the next
    # request starts it again (costs one engine load). 0 = keep it loaded forever.
    idle_unload_seconds: int = Field(0, ge=0)
    # VAE decode memory: direct convolution needs about half the memory of the default (im2col) path.
    # auto = on for GPU, off for CPU.
    vae_conv_direct: Literal["auto", "on", "off"] = "auto"
    # Tiled VAE decode with overlap for images that would not fit in the VAE GPU's memory. Without it,
    # stable-diffusion.cpp's out-of-memory fallback stitches two halves and leaves a visible seam.
    vae_tiling: Literal["auto", "on", "off"] = "auto"
    # auto: decode images above vae_tiling_above_megapixels in tiles of vae_tile_size px (image pixels).
    vae_tiling_above_megapixels: float = Field(1.05, ge=0.0, le=16.0)
    vae_tile_size: int = Field(1024, ge=256, le=4096)
    # Prefix KV cache type for Qwen-Image-2.1 edits (the reference-image tokens are cached across steps):
    # "auto" or a ggml type such as "q8_0" (half the memory of f16).
    prefix_cache_type: str = "auto"
    # Upper bound on (output pixels + reference pixels) / 256 per request; references are scaled down evenly to
    # fit. Bounds the memory of multi-image edits. 0 = no limit.
    token_budget: int = Field(24576, ge=0)
    # Noise schedule. "official": Qwen-Image-2.1's own sigmas (shift by image size, last step at 0.02).
    # "sdcpp": stable-diffusion.cpp's flux default, which shifts much harder and loses fine detail.
    schedule: Literal["official", "sdcpp"] = "official"
    # generate_image(tileable=true). "circular": one pass with wrap-around attention positions (RoPE) and
    # wrap-around VAE padding, so the left/right and top/bottom edges are generated as neighbours (needs the
    # sdcpp-circular-json engine patch). "repair": the older two-pass method (generate, shift by half, repaint a
    # cross-shaped band). "auto": circular when the engine supports it, else repair.
    tile_method: Literal["auto", "circular", "repair"] = "auto"

    @field_validator("default_size", "cpu_default_size", "panorama_size", "cpu_panorama_size")
    @classmethod
    def _size(cls, v: str) -> str:
        parse_size(v)
        return v


class OutputConfig(_Section):
    format: Literal["png", "webp", "jpeg"] = "png"
    jpeg_quality: int = Field(95, ge=1, le=100)
    inline: Literal["auto", "full", "preview", "none"] = "auto"
    inline_max_bytes: int = Field(700_000, ge=10_000)
    preview_max_side: int = Field(1024, ge=128)
    keep_days: float = Field(7, ge=0)
    embed_metadata: bool = False
    # Remove the faint 2-pixel diamond grid the Qwen-Image VAE leaves on smooth areas such as skin.
    degrid: bool = True


class TransparencyConfig(_Section):
    method: Literal["hybrid", "native", "matte"] = "hybrid"
    threads: int = Field(0, ge=0)  # CPU threads for the background-removal model (0 = the default)
    matte_model: Literal["birefnet-general-lite", "birefnet-general", "isnet-general-use"] = "birefnet-general-lite"
    # Clean edges: estimate the true colour of soft edge pixels, so the old background (the model's fill colour, or
    # the photo behind a cut-out) does not tint them.
    decontaminate: bool = True
    # What fully transparent pixels hold: "black" (smallest file, nothing of the source image left), "edge" (the
    # nearest visible colour; better for game engines and mipmaps) or "keep" (leave them as they are, and pass the
    # alpha of upscale/watermark inputs through unchanged: for textures that store data in the alpha channel).
    hidden_pixels: Literal["black", "edge", "keep"] = "black"


class PanoramaConfig(_Section):
    # How the 360 left/right edges are made to meet: "circular" generates them as neighbours in one pass (x axis
    # only; needs the sdcpp-circular-json engine patch), "repair" repaints a band over the seam afterwards
    # (seam_fix), "auto" = circular when supported. Default "repair" until circular is checked at panorama sizes.
    wrap_method: Literal["auto", "circular", "repair"] = "repair"
    seam_fix: bool = True
    seam_band_fraction: float = Field(0.125, gt=0.01, le=0.5)
    seam_strength: float = Field(0.6, gt=0.0, le=1.0)
    format: Literal["jpeg", "png", "webp"] = "jpeg"
    embed_gpano: bool = True


class WatermarkConfig(_Section):
    # The remove_watermark tool: an edit with a watermark-removal LoRA (downloaded to models/loras).
    enabled: bool = True
    prompt: str = "Remove the watermarks"  # the LoRA author's prompt
    strength: float = Field(1.0, ge=0.0, le=2.0)
    # The model works at about this size (larger images are processed scaled down, smaller ones scaled up);
    # with restore_unchanged the result is put back at the input's own size.
    megapixels: float = Field(1.05, ge=0.25, le=4.2)
    # Keep the original pixels wherever the model did not change the image, so only the watermark areas are
    # replaced and the rest keeps the input's full resolution and detail.
    restore_unchanged: bool = True


class UpscaleConfig(_Section):
    # The upscale_image tool: an ESRGAN super-resolution model run by a short-lived sd-cli process per request
    # (downloaded to models/upscalers).
    enabled: bool = True
    # 4xNomos2_otf_esrgan: Real-ESRGAN x4plus fine-tuned for photos (CC BY 4.0, 33 MB).
    # RealESRGAN_x4plus: the original general model (BSD-3-Clause, 67 MB).
    model: Literal["4xNomos2_otf_esrgan", "RealESRGAN_x4plus"] = "4xNomos2_otf_esrgan"
    # Tile size in input pixels. Larger tiles are a little faster and need more VRAM.
    tile_size: int = Field(128, ge=32, le=512)
    timeout_seconds: int = Field(900, ge=30)


class Config(_Section):
    device: Literal["gpu", "cpu"] = "gpu"
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    cpu: CpuConfig = Field(default_factory=CpuConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    transparency: TransparencyConfig = Field(default_factory=TransparencyConfig)
    panorama: PanoramaConfig = Field(default_factory=PanoramaConfig)
    watermark: WatermarkConfig = Field(default_factory=WatermarkConfig)
    upscale: UpscaleConfig = Field(default_factory=UpscaleConfig)

    models_dir: Path = Path("/models")
    outputs_dir: Path = Path("/outputs")
    sd_server_bin: Path = Path("/opt/sdcpp/sd-server")
    sd_server_port: int = 7861

    @field_validator("device", mode="before")
    @classmethod
    def _device(cls, v):
        s = str(v).strip().lower()
        return {"cuda": "gpu", "nvidia": "gpu"}.get(s, s)

    # --- derived values -------------------------------------------------
    @property
    def is_cpu(self) -> bool:
        return self.device == "cpu"

    @property
    def default_steps(self) -> int:
        return self.generation.cpu_steps if self.is_cpu else self.generation.steps

    @property
    def default_cfg_scale(self) -> float:
        return self.generation.cpu_cfg_scale if self.is_cpu else self.generation.cfg_scale

    @property
    def default_size(self) -> tuple[int, int]:
        return parse_size(self.generation.cpu_default_size if self.is_cpu else self.generation.default_size)

    @property
    def panorama_size(self) -> tuple[int, int]:
        return parse_size(self.generation.cpu_panorama_size if self.is_cpu else self.generation.panorama_size)

    @property
    def max_pixels(self) -> int:
        mp = self.generation.cpu_max_megapixels if self.is_cpu else self.generation.max_megapixels
        return int(mp * 1_000_000)

    @property
    def timeout_seconds(self) -> int:
        return self.generation.cpu_timeout_seconds if self.is_cpu else self.generation.timeout_seconds


ENV_OVERRIDES = {
    "IMAGEGEN_DEVICE": ("device",),
    "IMAGEGEN_QUANT": ("model", "quant"),
    "IMAGEGEN_PORT": ("server", "port"),
    "IMAGEGEN_PUBLIC_URL": ("server", "public_url"),
    "IMAGEGEN_LOG_LEVEL": ("server", "log_level"),
    "HF_TOKEN": ("model", "hf_token"),
    "IMAGEGEN_MODELS_DIR": ("models_dir",),
    "IMAGEGEN_OUTPUTS_DIR": ("outputs_dir",),
    "IMAGEGEN_SD_SERVER_BIN": ("sd_server_bin",),
}


def _apply_env(data: dict) -> dict:
    for env, keys in ENV_OVERRIDES.items():
        val = os.environ.get(env)
        if val is None or val == "":
            continue
        node = data
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = val
    return data


def default_config_path() -> Path:
    return Path(os.environ.get("IMAGEGEN_CONFIG", "/app/config.yaml"))


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else default_config_path()
    data: dict = {}
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"{p}: top level must be a mapping")
        data = loaded or {}
    return Config.model_validate(_apply_env(data))
