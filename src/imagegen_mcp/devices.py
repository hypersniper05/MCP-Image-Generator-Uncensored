"""Translate the ``device`` / ``gpu`` config sections into stable-diffusion.cpp
arguments and environment variables.

GPU indices in the config are the indices printed by ``nvidia-smi`` (PCI bus
order). Only the GPUs that the config uses are made visible to the inference
process (``CUDA_VISIBLE_DEVICES``), so other GPUs on a shared machine are never
touched. Inside the process the visible GPUs are renumbered ``cuda0..cudaN`` in
the same order, and the per-module ``--backend`` assignment refers to them.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

from .config import Config


@dataclass
class DevicePlan:
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    visible_gpus: list[int] = field(default_factory=list)
    placement: dict[str, str] = field(default_factory=dict)
    description: str = ""
    vae_cuda: int | None = None  # index of the VAE's GPU inside the engine process (cudaN), None = CPU
    threads: int = 0  # CPU threads given to the engine (-t)

    @property
    def upscale_backend(self) -> str:
        """Device for the image upscaler (run by sd-cli): the VAE's GPU, or the CPU."""
        return "cpu" if self.vae_cuda is None else f"cuda{self.vae_cuda}"


def physical_cpu_threads() -> int:
    """Best effort count of physical cores (ggml runs best on physical cores)."""
    try:
        cores = set()
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            phys = core = None
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if core is not None:
                        cores.add((phys, core))
                    phys = core = None
        if cores:
            return len(cores)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def gpu_uuids() -> dict[int, str]:
    """nvidia-smi index (PCI bus order) -> GPU UUID, or {} when nvidia-smi is not available."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    res = {}
    for line in out.splitlines():
        idx, _, uuid = line.partition(",")
        if idx.strip().isdigit() and uuid.strip().startswith("GPU-"):
            res[int(idx)] = uuid.strip()
    return res


def _gpu_name(idx: int, remap: dict[int, int]) -> str:
    return f"cuda{remap[idx]}"


def build_device_plan(cfg: Config) -> DevicePlan:
    plan = DevicePlan()
    threads = cfg.cpu.threads or physical_cpu_threads()

    if cfg.is_cpu:
        plan.args = ["--backend", "cpu", "-t", str(threads)]
        plan.threads = threads
        # Hide every GPU so a CUDA build never creates a context on shared cards.
        plan.env = {"CUDA_VISIBLE_DEVICES": "-1"}
        plan.placement = {"diffusion": "cpu", "text_encoder": "cpu", "vae": "cpu"}
        plan.description = f"CPU only ({threads} threads)"
        return plan

    g = cfg.gpu
    used: set[int] = set(g.diffusion)
    for ref in (g.text_encoder, g.vae):
        if ref != "cpu":
            used.add(int(ref))
    for idx in g.max_vram_gb:
        if idx not in used:
            raise ValueError(f"gpu.max_vram_gb lists GPU {idx}, which no model part is assigned to")
    visible = sorted(used)
    remap = {phys: i for i, phys in enumerate(visible)}

    def ref_name(ref) -> str:
        return "cpu" if ref == "cpu" else _gpu_name(int(ref), remap)

    diffusion = "&".join(_gpu_name(i, remap) for i in g.diffusion)
    # The image upscaler (ESRGAN) is not covered by auto-fit and would otherwise run on the first visible GPU;
    # keep it with the VAE, which is the other image-sized stage.
    backend = f"diffusion={diffusion},te={ref_name(g.text_encoder)},vae={ref_name(g.vae)}"
    args = ["--backend", backend, "-t", str(threads)]

    if len(g.diffusion) > 1:
        args += ["--split-mode", f"diffusion={g.split_mode}"]
    if g.max_vram_gb:
        budget = ",".join(f"{_gpu_name(i, remap)}={v:g}" for i, v in sorted(g.max_vram_gb.items()))
        args += ["--max-vram", budget]
    if isinstance(g.offload, dict):
        # Per part: where the weights live. Compute still runs on the part's device.
        module = {"diffusion": "diffusion", "text_encoder": "te", "vae": "vae"}
        spec = [f"{module[part]}={where}" for part, where in g.offload.items() if where != "gpu"]
        args += ["--params-backend", ",".join(spec)] if spec else ["--auto-fit", "off"]
    elif g.offload == "none":
        args += ["--auto-fit", "off"]
    elif g.offload == "cpu":
        args += ["--offload-to-cpu"]
    elif g.offload == "disk":
        args += ["--params-backend", "disk"]
    if g.flash_attention == "diffusion":
        args += ["--diffusion-fa"]
    elif g.flash_attention:
        args += ["--fa"]  # every module; the text encoder's long multi-image prompts need it most

    plan.args = args
    plan.threads = threads
    plan.visible_gpus = visible
    plan.vae_cuda = None if g.vae == "cpu" else remap[int(g.vae)]
    uuids = gpu_uuids()
    allowed = [x.strip() for x in os.environ.get("IMAGEGEN_CONTAINER_GPUS", "").split(",") if x.strip()]
    if allowed and uuids:  # the container was started with CUDA_VISIBLE_DEVICES: treat it as an allow-list
        outside = [i for i in visible if uuids.get(i) not in allowed and str(i) not in allowed]
        if outside:
            raise ValueError(f"config.yaml uses GPU {outside}, but the container is limited to {allowed} "
                             "(CUDA_VISIBLE_DEVICES in docker-compose.override.yml)")
    plan.env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        # Pin by UUID when nvidia-smi can tell us the UUIDs: unlike an index, it cannot drift to another card.
        "CUDA_VISIBLE_DEVICES": ",".join(uuids.get(i, str(i)) for i in visible),
    }
    plan.placement = {
        "diffusion": " + ".join(f"GPU {i}" for i in g.diffusion),
        "text_encoder": "cpu" if g.text_encoder == "cpu" else f"GPU {g.text_encoder}",
        "vae": "cpu" if g.vae == "cpu" else f"GPU {g.vae}",
    }
    plan.description = (
        f"GPU mode: diffusion on {plan.placement['diffusion']}, text encoder on "
        f"{plan.placement['text_encoder']}, VAE on {plan.placement['vae']} "
        f"(visible GPUs {visible} -> {[f'cuda{i}' for i in range(len(visible))]})"
    )
    return plan
