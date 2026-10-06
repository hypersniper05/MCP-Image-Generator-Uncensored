"""Entry point: ``python -m imagegen_mcp [--config PATH] [--check]``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="imagegen-mcp", description=__doc__)
    ap.add_argument("--config", help="path to config.yaml (default: $IMAGEGEN_CONFIG or /app/config.yaml)")
    ap.add_argument("--check", action="store_true", help="validate the config, print the device plan and exit")
    ap.add_argument("--download-only", action="store_true", help="download/verify the model files and exit")
    args = ap.parse_args(argv)

    from .config import load_config

    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    _setup_logging(cfg.server.log_level)
    log = logging.getLogger("imagegen")

    # A container-level CUDA_VISIBLE_DEVICES (docker-compose.override.yml) is an allow-list for the GPUs in
    # config.yaml; devices.py checks it. This process then sets its own value below.
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        os.environ.setdefault("IMAGEGEN_CONTAINER_GPUS", os.environ["CUDA_VISIBLE_DEVICES"])
    # The background-removal model runs inside this process: expose only its GPU.
    if not cfg.is_cpu and cfg.gpu.background_removal != "cpu":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu.background_removal)
    else:
        # nothing in this process uses a GPU; sd-server gets its own pinned CUDA_VISIBLE_DEVICES (devices.py)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    from .devices import build_device_plan
    from .models import required_files

    if args.check:
        plan = build_device_plan(cfg)
        print(f"device: {cfg.device}\n{plan.description}\nsd-server args: {' '.join(plan.args)}\nenv: {plan.env}")
        for f in required_files(cfg):
            print(f"model file: {f.dest} ({f.size / 1e9:.2f} GB)")
        return 0

    if args.download_only:
        from .models import ModelStore

        store = ModelStore(cfg)

        async def _all():
            await store.ensure()
            await store.fetch_optional()

        asyncio.run(_all())
        failed = [f for f, st in store.status.files.items() if st in ("error", "missing")]
        if failed:
            print(f"optional model files not available: {', '.join(failed)}", file=sys.stderr)
        log.info("all model files are present and verified")
        return 0

    import uvicorn

    from .server import build_app
    from .service import ImageService

    service = ImageService(cfg)
    _, app = build_app(service)

    async def run() -> None:
        config = uvicorn.Config(app, host=cfg.server.host, port=cfg.server.port, log_level="warning",
                                timeout_keep_alive=75, proxy_headers=False)
        server = uvicorn.Server(config)
        startup = asyncio.create_task(service.startup())
        log.info("MCP server listening on http://%s:%d%s (no auth)", cfg.server.host, cfg.server.port,
                 cfg.server.path)
        try:
            await server.serve()  # uvicorn handles SIGINT/SIGTERM
        finally:
            startup.cancel()
            await service.shutdown()

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
