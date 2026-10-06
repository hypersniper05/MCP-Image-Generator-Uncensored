from pathlib import Path

import pytest

from imagegen_mcp.config import Config, load_config
from imagegen_mcp.devices import build_device_plan
from imagegen_mcp.models import diffusion_file, required_files

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_config_is_valid():
    path = ROOT / "config.example.yaml"  # the start scripts copy it to the local, untracked config.yaml
    assert path.is_file()
    cfg = load_config(path)
    assert cfg.device == "gpu"
    assert cfg.model.quant == "Q4_K_M"
    assert cfg.server.host == "0.0.0.0"
    assert diffusion_file(cfg).filename == "qwen-image-2.1-UC-Q4_K_M.gguf"


def test_shipped_config_is_portable():
    """The published example must work on a one-GPU machine: no local GPU pinning or tuning."""
    shipped = load_config(ROOT / "config.example.yaml").model_dump()
    assert shipped == Config().model_dump()  # every value is the code default
    assert shipped["gpu"]["diffusion"] == [0] and shipped["gpu"]["max_vram_gb"] == {}


@pytest.mark.parametrize("alias,expected", [("Q4", "Q4_K_M"), ("6Q", "Q6_K"), ("q8", "Q8_0"), ("Q6_K", "Q6_K")])
def test_quant_aliases(alias, expected):
    cfg = Config.model_validate({"model": {"quant": alias}})
    assert cfg.model.quant == expected


def test_quant_files():
    for q, name in [("Q6_K", "qwen-image-2.1-UC-Q6_K.gguf"), ("Q8_0", "qwen-image-2.1-UC-Q8_0.gguf")]:
        cfg = Config.model_validate({"model": {"quant": q}})
        f = diffusion_file(cfg)
        assert f.filename == name and f.sha256 and f.size > 4e9
        assert "/resolve/1206d38b" in f.url


def test_base_variant_rejects_bf16():
    with pytest.raises(Exception):
        Config.model_validate({"model": {"quant": "BF16", "variant": "base"}})


def test_unknown_keys_rejected():
    with pytest.raises(Exception):
        Config.model_validate({"gpu": {"diffusoin": [0]}})


def test_env_override(monkeypatch, tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("device: gpu\nmodel:\n  quant: Q4_K_M\n")
    monkeypatch.setenv("IMAGEGEN_DEVICE", "cpu")
    monkeypatch.setenv("IMAGEGEN_QUANT", "Q6")
    cfg = load_config(p)
    assert cfg.device == "cpu" and cfg.model.quant == "Q6_K"


def test_cpu_plan_hides_gpus():
    plan = build_device_plan(Config.model_validate({"device": "cpu", "cpu": {"threads": 6}}))
    assert plan.args[:2] == ["--backend", "cpu"]
    assert "-t" in plan.args and "6" in plan.args
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "-1"


def test_single_gpu_plan():
    plan = build_device_plan(Config.model_validate({"gpu": {"diffusion": [0], "text_encoder": 0, "vae": 0}}))
    assert plan.env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"}
    assert "diffusion=cuda0,te=cuda0,vae=cuda0" in plan.args and "upscaler=" not in " ".join(plan.args)
    assert "--fa" in plan.args and "--diffusion-fa" not in plan.args


def test_split_parts_across_gpus_are_remapped():
    cfg = Config.model_validate({"gpu": {"diffusion": [2], "text_encoder": 1, "vae": "cpu",
                                         "max_vram_gb": {2: 10, 1: 8}, "offload": "cpu"}})
    plan = build_device_plan(cfg)
    # physical GPUs 1 and 2 become cuda0 and cuda1 inside the process
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert "diffusion=cuda1,te=cuda0,vae=cpu" in plan.args and plan.upscale_backend == "cpu"
    i = plan.args.index("--max-vram")
    assert plan.args[i + 1] == "cuda0=8,cuda1=10"
    assert "--offload-to-cpu" in plan.args
    assert plan.visible_gpus == [1, 2]


def test_multi_gpu_diffusion_split():
    cfg = Config.model_validate({"gpu": {"diffusion": ["cuda0", 2], "text_encoder": "cuda1", "vae": 0,
                                         "split_mode": "row"}})
    plan = build_device_plan(cfg)
    assert "diffusion=cuda0&cuda2,te=cuda1,vae=cuda0" in plan.args
    assert plan.args[plan.args.index("--split-mode") + 1] == "diffusion=row"


def test_max_vram_for_unused_gpu_is_an_error():
    with pytest.raises(ValueError):
        build_device_plan(Config.model_validate({"gpu": {"max_vram_gb": {3: 5}}}))


def test_required_files_include_matte_only_when_needed():
    files = {f.key: f for f in required_files(Config())}
    assert set(files) == {"diffusion", "text_encoder", "vision", "vae", "background_removal", "watermark_lora",
                          "upscaler"}
    assert files["watermark_lora"].optional and not files["diffusion"].optional
    assert files["watermark_lora"].dest == "loras/watermark_remover_v2_qwen.safetensors"
    keys = {f.key for f in required_files(Config.model_validate({"transparency": {"method": "native"},
                                                                  "watermark": {"enabled": False},
                                                                  "upscale": {"enabled": False}}))}
    assert not keys & {"background_removal", "watermark_lora", "upscaler"}


def test_cpu_defaults():
    cfg = Config.model_validate({"device": "cpu"})
    assert cfg.default_size == (512, 512)
    assert cfg.default_steps == 12
    assert cfg.panorama_size == (1024, 512)


def test_per_part_offload():
    plan = build_device_plan(Config.model_validate({"gpu": {"offload": {"text_encoder": "cpu", "vae": "disk"}}}))
    assert plan.args[plan.args.index("--params-backend") + 1] == "te=cpu,vae=disk"
    plan = build_device_plan(Config.model_validate({"gpu": {"offload": {"diffusion": "gpu"}}}))
    assert "--auto-fit" in plan.args
    with pytest.raises(Exception):
        Config.model_validate({"gpu": {"offload": {"clip": "cpu"}}})


def test_vae_tiling_decision():
    from imagegen_mcp.sdserver import SdServer

    def engine(cfg_dict, vram):
        cfg = Config.model_validate(cfg_dict)
        e = SdServer(cfg, build_device_plan(cfg), {})
        if vram:
            e.vram_mib[0] = vram
        return e

    big = engine({}, 32607)  # RTX 5090
    assert big.vae_conv_direct
    assert big.vae_tiling_for(1024, 1024) is None
    t = big.vae_tiling_for(2720, 1536)  # 4.2 MP: the case that produced the W/2 seam
    assert t == {"enabled": True, "tile_size_w": 1024, "tile_size_h": 1024, "target_overlap": 0.5}
    small = engine({}, 12288)  # RTX 3080 Ti
    assert small.vae_tiling_for(1024, 1024) is None
    assert small.vae_tiling_for(1536, 1536)["tile_size_w"] == 1024  # above 1.05 MP: 1024 px tiles
    custom = engine({"generation": {"vae_tile_size": 512, "vae_tiling_above_megapixels": 4.0}}, 12288)
    assert custom.vae_tiling_for(1536, 1536) is None and custom.vae_tiling_for(2048, 2048)["tile_size_h"] == 512
    cpu = engine({"device": "cpu"}, None)
    assert not cpu.vae_conv_direct and cpu.vae_tiling_for(2048, 1024) is None
    forced = engine({"generation": {"vae_tiling": "off"}}, 12288)
    assert forced.vae_tiling_for(2048, 2048) is None


def test_official_schedule():
    from imagegen_mcp.schedule import official_mu, official_sigmas

    # Reference values from diffusers' FlowMatchEulerDiscreteScheduler with Qwen-Image-2.1's config.
    assert round(official_mu(1024, 1024), 4) == 0.6935
    assert round(official_mu(2048, 2048), 4) == 1.3129
    s = official_sigmas(1024, 1024, 20)
    assert len(s) == 21 and s[0] == 1.0 and s[-2] == 0.02 and s[-1] == 0.0
    assert s[1] == 0.972237 and s[10] == 0.639030
    assert official_sigmas(2048, 2048, 40)[38] == 0.102230
    assert all(a > b for a, b in zip(s, s[1:]))
    assert official_sigmas(512, 512, 1) == [1.0, 0.0]


def test_vae_choice():
    from imagegen_mcp.models import vae_file

    tf = vae_file(Config.model_validate({}))
    assert tf.dest == "vae/texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
    assert "madebyollin/texture-fix-vae-for-qwen-image-2.1/resolve/1f80456b" in tf.url and tf.sha256.startswith("05e9af8d")
    orig = vae_file(Config.model_validate({"model": {"vae": "original"}}))
    assert orig.dest == "vae/qwen_image_2.1_vae_bf16.safetensors" and orig.sha256.startswith("bb21f747")


def test_gpus_are_pinned_by_uuid(monkeypatch):
    from imagegen_mcp import devices
    monkeypatch.setattr(devices, "gpu_uuids", lambda: {0: "GPU-aaa", 1: "GPU-bbb", 2: "GPU-ccc"})
    plan = build_device_plan(Config.model_validate({"gpu": {"diffusion": [2], "text_encoder": 2, "vae": 2}}))
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "GPU-ccc"
    assert plan.args[plan.args.index("--backend") + 1] == "diffusion=cuda0,te=cuda0,vae=cuda0"
    monkeypatch.setattr(devices, "gpu_uuids", lambda: {})  # nvidia-smi unavailable: fall back to indices
    assert build_device_plan(Config.model_validate({"gpu": {"diffusion": [2], "vae": 2, "text_encoder": 2}})
                             ).env["CUDA_VISIBLE_DEVICES"] == "2"


def test_prefix_cache_type_argument():
    from imagegen_mcp.sdserver import SdServer
    paths = {"diffusion": "d", "vae": "v", "text_encoder": "t", "vision": "x"}
    cfg = Config.model_validate({"generation": {"prefix_cache_type": "q8_0"}})
    cmd = SdServer(cfg, build_device_plan(cfg), paths).command()
    assert cmd[cmd.index("--model-args") + 1] == "qwen_image_2_1_prefix_cache_type=q8_0"
    assert "--model-args" not in SdServer(Config(), build_device_plan(Config()), paths).command()


def test_flash_attention_modes():
    def args(v):
        return build_device_plan(Config.model_validate({"gpu": {"flash_attention": v}})).args
    assert "--fa" in args(True) and "--diffusion-fa" not in args(True)
    assert "--diffusion-fa" in args("diffusion") and "--fa" not in args("diffusion")
    assert "--fa" not in args(False) and "--diffusion-fa" not in args(False)


def test_container_gpu_allow_list(monkeypatch):
    from imagegen_mcp import devices
    monkeypatch.setattr(devices, "gpu_uuids", lambda: {1: "GPU-bbb", 2: "GPU-ccc"})
    monkeypatch.setenv("IMAGEGEN_CONTAINER_GPUS", "GPU-ccc")
    ok = {"gpu": {"diffusion": [2], "text_encoder": 2, "vae": 2}}
    assert build_device_plan(Config.model_validate(ok)).env["CUDA_VISIBLE_DEVICES"] == "GPU-ccc"
    with pytest.raises(ValueError, match="limited to"):
        build_device_plan(Config.model_validate({"gpu": {"diffusion": [2], "text_encoder": 1, "vae": 2}}))
