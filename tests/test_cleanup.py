"""The outputs.keep_days cleanup deletes old results but never the repository's hidden files."""
import asyncio
import os
import time

from imagegen_mcp.config import Config
from imagegen_mcp.service import ImageService


async def test_cleanup_removes_old_outputs_but_keeps_gitkeep(tmp_path):
    svc = ImageService(Config.model_validate({"outputs_dir": str(tmp_path / "out"), "models_dir": str(tmp_path / "m"),
                                              "outputs": {"keep_days": 1}}))
    out = svc.outputs
    (out / "2026-01-01").mkdir(parents=True)
    old, new, keep = out / "2026-01-01" / "image-old.png", out / "image-new.png", out / ".gitkeep"
    for p in (old, new, keep):
        p.write_bytes(b"x")
    long_ago = time.time() - 30 * 86400
    for p in (old, keep):
        os.utime(p, (long_ago, long_ago))
    task = asyncio.create_task(svc._cleanup_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    assert not old.exists() and not (out / "2026-01-01").exists()  # old result and its empty folder are gone
    assert new.exists() and keep.exists()
