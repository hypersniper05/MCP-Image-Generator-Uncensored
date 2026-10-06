"""End-to-end test of a running Image Gen MCP server.

    pip install "mcp==2.3.0" pillow
    python scripts/smoke_test.py http://localhost:5005/mcp [--quick] [--out DIR]

Calls the main tools (generate_image, edit_image, generate_panorama, remove_background, get_job,
server_status) over Streamable HTTP (modern and legacy handshakes), checks the results, and saves the
returned images to --out (default ./smoke_out).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

from mcp import Client
from PIL import Image


def _save(result, out: Path, name: str) -> list[dict]:
    images = (result.structured_content or {}).get("images", [])
    for i, c in enumerate(result.content):
        if c.type == "image":
            ext = c.mime_type.split("/")[1].replace("jpeg", "jpg")
            (out / f"{name}_inline{i}.{ext}").write_bytes(base64.b64decode(c.data))
    for i, item in enumerate(images):
        with urllib.request.urlopen(item["url"], timeout=60) as r:
            data = r.read()
        ext = item["file"].rsplit(".", 1)[-1]
        (out / f"{name}_{i}.{ext}").write_bytes(data)
        item["_image"] = Image.open(io.BytesIO(data))
    return images


def _check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _check.failed += 1


_check.failed = 0


async def run(url: str, out: Path, quick: bool, mode: str, steps: int | None) -> None:
    progress: list[str] = []

    async def on_progress(p: float, total: float | None, message: str | None) -> None:
        progress.append(f"{p:.0f}/{total or 100:.0f} {message or ''}")

    async with Client(url, mode=mode, read_timeout_seconds=7200) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        print(f"[{mode}] tools: {sorted(tools)}")
        _check({"generate_image", "edit_image", "generate_panorama", "remove_background", "server_status"} <= tools,
               "all tools listed")

        status = await client.call_tool("server_status", {})
        st = status.structured_content or json.loads(status.content[0].text)
        print(f"  status: state={st['state']} device={st['device']} placement={st['placement']}")
        _check(st["state"] == "ready", "server is ready")
        small = st["device"] == "cpu" or quick

        async def call(name: str, args: dict, label: str):
            progress.clear()
            t = time.monotonic()
            if steps and name != "remove_background":
                args = {**args, "steps": steps}
            r = await client.call_tool(name, args, progress_callback=on_progress)
            polls = 0
            # Long jobs return a job_id; keep fetching it with get_job until the image is ready.
            while not r.is_error and (r.structured_content or {}).get("status") == "running":
                polls += 1
                r = await client.call_tool("get_job", {"job_id": r.structured_content["job_id"]},
                                           progress_callback=on_progress)
            dt = time.monotonic() - t
            if polls:
                print(f"    ({label} returned a job_id and was fetched with get_job {polls} time(s))")
            text = next((c.text for c in r.content if c.type == "text"), "")
            print(f"  {label}: {dt:.1f} s, {len(progress)} progress updates" + (f", last: {progress[-1]}" if progress else ""))
            if r.is_error:
                print("    error:", text)
            _check(not r.is_error, f"{label} succeeded")
            return r

        size = {"width": 512, "height": 512} if small else {"aspect_ratio": "16:9"}
        r = await call("generate_image", {"prompt": "A red vintage bicycle leaning against a sunny yellow wall, "
                                          "photorealistic", "seed": 7, **size}, "generate_image")
        imgs = _save(r, out, f"{mode}_generate")
        _check(bool(imgs) and imgs[0]["_image"].size[0] % 32 == 0, "generated image saved and downloadable")
        _check(bool(progress), "progress notifications received")
        first_url = imgs[0]["url"] if imgs else None

        r = await call("generate_image", {"prompt": "a cute cartoon robot mascot, full body, flat colors",
                                          "transparent": True, "seed": 3, "width": 512, "height": 512},
                       "generate_image transparent")
        imgs = _save(r, out, f"{mode}_transparent")
        if imgs:
            im = imgs[0]["_image"]
            alpha = im.getchannel("A") if im.mode == "RGBA" else None
            clear = sum(alpha.histogram()[:16]) / (im.width * im.height) if alpha else 0
            _check(im.mode == "RGBA" and clear > 0.2, f"transparent PNG has an alpha channel ({clear:.0%} clear)")

        if first_url:
            r = await call("edit_image", {"prompt": "Make the bicycle bright blue, keep everything else the same",
                                          "images": [first_url], "seed": 11, **({"width": 512, "height": 512} if small else {})},
                           "edit_image (1 image, chained by URL)")
            _save(r, out, f"{mode}_edit1")

            buf = io.BytesIO()
            Image.new("RGB", (256, 256), (30, 144, 255)).save(buf, "PNG")
            ball = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            r = await call("edit_image", {"prompt": "Put a blue ball with the color of <image2> in the basket of the "
                                          "bicycle in <image1>", "images": [first_url, ball], "seed": 5,
                                          **({"width": 512, "height": 512} if small else {})},
                           "edit_image (2 images, URL + data URL)")
            _save(r, out, f"{mode}_edit2")

            r = await call("remove_background", {"image": first_url}, "remove_background")
            imgs = _save(r, out, f"{mode}_cutout")
            _check(bool(imgs) and imgs[0]["_image"].mode == "RGBA", "cutout is RGBA")

        if not quick:
            r = await call("generate_panorama", {"prompt": "a quiet alpine lake surrounded by snowy mountains at "
                                                 "sunrise", "seed": 21, **({"width": 1024} if small else {})},
                           "generate_panorama")
            imgs = _save(r, out, f"{mode}_panorama")
            if imgs:
                im = imgs[0]["_image"]
                _check(im.width == 2 * im.height, f"panorama is 2:1 ({im.width}x{im.height})")
                _check("viewer_url" in imgs[0], "panorama has a viewer link")
                info = (r.structured_content or {}).get("info", {})
                print(f"    seam score before/after: {info.get('seam_score_before')} / {info.get('seam_score_after')}")

        r = await client.call_tool("edit_image", {"prompt": "x", "images": ["not an image"]})
        _check(r.is_error, "bad input returns a tool error instead of crashing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://localhost:5005/mcp")
    ap.add_argument("--quick", action="store_true", help="small images, skip the panorama")
    ap.add_argument("--out", default="smoke_out")
    ap.add_argument("--modes", default="auto,legacy", help="protocol negotiation modes to test")
    ap.add_argument("--steps", type=int, default=None, help="override steps (e.g. 4 for a fast CPU test)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, mode in enumerate(args.modes.split(",")):
        asyncio.run(run(args.url, out, args.quick or i > 0, mode.strip(), args.steps))
    print(f"\n{'all checks passed' if not _check.failed else f'{_check.failed} check(s) FAILED'}; images in {out}/")
    return 1 if _check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
