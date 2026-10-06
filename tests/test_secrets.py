"""The optional Hugging Face token never leaves the download code."""
import json

from imagegen_mcp import models, sdserver
from imagegen_mcp.config import Config
from imagegen_mcp.server import create_server
from imagegen_mcp.service import ImageService

TOKEN = "dummy-value-for-tests-only"


def test_token_is_only_for_hugging_face_over_https():
    assert models.is_hugging_face("https://huggingface.co/org/repo/resolve/abc/file.gguf")
    assert models.is_hugging_face("https://cdn-lfs.huggingface.co/x")
    for url in ("http://huggingface.co/x", "https://huggingface.co.evil.example/x",
                "https://evil.example/huggingface.co/x", "https://civitai.com/api/download/models/1",
                "https://github.com/danielgatis/rembg/releases/download/v0.0.0/x.onnx"):
        assert not models.is_hugging_face(url), url


def test_engine_processes_do_not_get_the_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", TOKEN)
    env = sdserver.child_env({"CUDA_VISIBLE_DEVICES": "0"})
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and TOKEN not in json.dumps(env)


async def test_status_and_tools_never_show_the_token(tmp_path):
    cfg = Config.model_validate({"outputs_dir": str(tmp_path / "out"), "models_dir": str(tmp_path / "m"),
                                 "model": {"hf_token": TOKEN}})
    svc = ImageService(cfg)
    assert TOKEN not in json.dumps(svc.status(), default=str)
    mcp = create_server(svc)
    result = await mcp.call_tool("server_status", {})
    assert TOKEN not in json.dumps(result, default=str)
    tools = await mcp.list_tools()
    assert TOKEN not in json.dumps([t.model_dump() for t in tools], default=str)
