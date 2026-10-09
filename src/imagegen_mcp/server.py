"""MCP server (Streamable HTTP) exposing the image tools, plus a few plain HTTP routes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.apps import Apps
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from PIL import Image
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response

from . import __version__, imaging, panorama, viewer, webui
from .config import Config
from .imaging import ImageInputError
from .jobs import Job
from .sdserver import EngineError
from .service import (REQUEST_BASE_URL, ImageService, OpResult, ServiceUnavailable, base_url_from_headers,
                      servable)

log = logging.getLogger("imagegen.server")

# Usage guidance the model sees. Most clients send it with every request, so it stays short. Every tool description
# has the same shape: what the tool does, when to use it (and when not, with the tool to use instead), one example call
# (models follow an example better than a description of one), short rules, and what it returns. Each rule is said
# once. The descriptions are filled in with this server's configured defaults (see _guide), so they stay true when
# config.yaml changes. Sources: the official Qwen-Image-2.1 settings, community testing, and A/B runs of this server
# (see the README). Parameter descriptions carry no numbers that depend on the config; the tool description has them.
# Keep every description under 2,048 characters: some clients cut longer ones, and the job rule is at the end.
JOB_LINE = ('If it returns "Not finished yet" with a job_id, call get_job with that job_id. Never call this tool again '
            "to get that image: it would render it twice.")
IMAGE_INPUT = "this server's link or file path exactly as given (with the date folder), a URL, a data URL or base64"
EXAMPLE_FILE = "2026-09-23/image-021530-ab12cd34.png"  # a result, as results and list_images give it
EXAMPLE_UPLOAD = "uploads/2026-09-23/dog-1a2b3c4d.jpg"  # an upload from the /upload page
EXAMPLE_JOB = "c0ffee123456"


def _example(**args) -> str:
    return "Example: " + json.dumps(args)


def _describe(what: str, use: str, example: str, returns: str, avoid: str = "", rules: tuple[str, ...] = (),
              job: bool = False) -> str:
    """A tool description in the shape every tool shares (see above)."""
    return "\n".join([what, f"Use when: {use}", *([f"Not for: {avoid}"] if avoid else []), example, *rules,
                      f"Returns: {returns}", *([JOB_LINE] if job else [])])


AspectRatio = Literal["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "9:21", "2:1", "1:2", "5:4", "4:5"]
SizeTier = Literal["small", "medium", "large", "xl"]
Fmt = Literal["png", "webp", "jpeg"]
Steps = Annotated[int | None, Field(ge=1, le=100, description="Denoising steps. Leave unset.")]
Size = Annotated[Literal["small", "medium", "large", "xl"] | None, Field(description=(
    "Pixel count: small ~0.26 MP, medium ~1 MP, large ~2 MP, xl ~4 MP (native 2K, sharpest). aspect_ratio sets the "
    "shape. Follow the tool description."))]
Guidance = Annotated[float | None, Field(ge=0, le=20, description="Guidance scale. Leave unset unless the user asks.")]
Negative = Annotated[str, Field(description=(
    "Extra things to avoid, comma-separated. Usually leave empty; ignored when cfg_scale is 1 or less."))]
Seed = Annotated[int | None, Field(description=(
    "Omit for new images and variations. Pass an earlier seed only to reproduce that exact image."))]
EditSeed = Annotated[int | None, Field(description="Leave unset. Never pass the seed that produced the input image.")]
Width = Annotated[int | None, Field(ge=256, le=4096, description=(
    "Width in px (multiple of 32); overrides size. Alone, the height follows aspect_ratio."))]
Height = Annotated[int | None, Field(ge=256, le=4096, description=(
    "Height in px (multiple of 32); overrides size. Alone, the width follows aspect_ratio."))]
EditWidth = Annotated[int | None, Field(ge=256, le=4096, description=(
    "Output width in px (multiple of 32); overrides size. Leave unset to keep the shape of <image1>."))]
EditHeight = Annotated[int | None, Field(ge=256, le=4096, description=(
    "Output height in px (multiple of 32); overrides size. Leave unset to keep the shape of <image1>."))]
OutputFormat = Annotated[Fmt | None, Field(description="Saved file format (default png).")]


def _guide(cfg: Config) -> dict[str, str]:
    """Server instructions and tool descriptions with this server's defaults filled in."""
    g = cfg.generation
    steps = cfg.default_steps
    t2i_cfg = cfg.default_cfg_scale
    cpu = cfg.is_cpu
    max_mp = cfg.max_pixels / 1e6
    wm = cfg.watermark

    def fmt(v: float) -> str:
        return f"{v:g}"

    if cpu:
        why_steps = f"lowered for CPU speed; the official setting is {g.steps}"
    else:
        why_steps = "the official setting" if steps == 40 else "this server's setting"
    draft = " Use 25-30 only if the user asks for a quick draft." if steps > 30 else ""
    steps_line = f"- steps: leave unset ({steps}, {why_steps}).{draft}"

    def cfg_line(value: float, edit: bool) -> str:
        if value <= 1:
            why = "because this server runs on a CPU" if cpu else "because this server is configured for speed"
            extra = " For edits consider cfg_scale 4: at 1 the model often ignores the instruction." if edit else ""
            return (f"- cfg_scale: leave unset ({fmt(value)} here, {why}); 3-5 is cleaner and enables the negative "
                    f"prompt but takes twice as long per step.{extra}")
        why = ("at 1 the model often ignores the instruction and returns a copy" if edit
               else "1 is twice as fast but hazier")
        return f"- cfg_scale: leave unset ({fmt(value)}, the tested best value; {why})."

    def neg_line(built_in: str, value: float) -> str:
        if not built_in:
            return "- negative_prompt: only used when cfg_scale is above 1; then list specific things to avoid."
        if value > 1:
            return (f'- negative_prompt: the server already applies "{built_in}"; pass extras only when the user '
                    "wants something kept out.")
        return (f"- negative_prompt: only used when cfg_scale is above 1 (not the default here); then the server "
                f'applies "{built_in}".')

    rec = g.recommended_size
    unset_mp = f"about {cfg.default_size[0] * cfg.default_size[1] / 1e6:.1g} MP"
    if not cpu and rec == "large" and max_mp >= 2.0:
        size_line = ('- size: pass size="large" (about 2 MP) for final images unless the user asks for a draft or '
                     'speed. "xl" (native 2K, about 4 MP) is sharper but takes several minutes here: use it only when '
                     'the user asks for maximum quality. Unset gives "medium" (about 1 MP), a draft size.')
        edit_size = (f'Leave size unset for normal edits ({unset_mp}); pass "large" or "xl" only for extra detail '
                     "(slower).")
        example_size = {"size": "large"}
    elif not cpu and rec == "medium":
        size_line = ('- size: leave size unset ("medium", about 1 MP) for normal images; larger sizes take several '
                     'minutes on this GPU. Pass "large" (about 2 MP) or "xl" (native 2K) only when the user asks for '
                     "more detail.")
        edit_size = "Leave size unset."
        example_size = {}
    elif max_mp >= 4.0:
        size_line = ('- size: pass size="xl" (native 2K, about 4 MP) unless the user asks for a draft or speed; '
                     'unset gives "medium" (about 1 MP), 4x faster but softer.')
        edit_size = f'For the most detail pass size="xl" (slower); otherwise leave size unset ({unset_mp}).'
        example_size = {"size": "xl"}
    else:
        size_line = (f"- size: this server allows at most about {max_mp:.1f} MP. Leave size unset (default "
                     f'{cfg.default_size[0]}x{cfg.default_size[1]}); "medium" is the largest useful tier and takes '
                     "several times longer.")
        edit_size = "Leave size unset."
        example_size = {}
    if cpu:
        timing = "Generation takes minutes on this CPU server."
    elif rec == "xl":
        timing = "Generation takes 30 seconds to a few minutes."
    else:
        timing = "Generation takes one to several minutes."
    next_tools = ", ".join(["edit_image", *(["upscale_image"] if cfg.upscale.enabled else [])]) + " or remove_background"

    generate = _describe(
        "Create a new image from a text prompt.",
        "the user wants a new picture: any size or aspect ratio, optionally with a transparent background "
        "(transparent=true) or as a seamless texture (tileable=true).",
        _example(prompt="A vertical painting of a red fox in a snowy forest at dusk, warm light from a cabin window, "
                        "soft brushwork", **example_size, aspect_ratio="2:3"),
        f"the image, its link and its file path; pass the file path to {next_tools} to keep working on it.",
        avoid="changing an existing image (edit_image) or a 360 panorama (generate_panorama).",
        rules=(
            "- prompt: a detailed description, not keywords: subject, setting, composition, lighting, colors and "
            "medium (photo with camera and lens, painting, 3D render, ...). Put text to render in double quotes. No "
            "resolutions or 4K/8K words: use size and aspect_ratio.",
            size_line,
            steps_line,
            cfg_line(t2i_cfg, edit=False),
            neg_line(g.negative_prompt, t2i_cfg),
            '- Another version of an earlier image ("one more like it"): call generate_image again with your earlier '
            "prompt, changed only as asked, and omit seed (never pass the earlier seed).",
        ), job=True)

    edit = _describe(
        "Edit one image or combine up to 10 images.",
        "the user wants to change an existing picture: add, remove or replace objects or text, change style, "
        "background, lighting or pose, or merge elements from several images.",
        _example(prompt="Put the dog from <image2> on the sofa in <image1>, keep everything else unchanged",
                 images=[EXAMPLE_FILE, EXAMPLE_UPLOAD]),
        "the edited image, its link and its file path (pass it as <image1> to edit it further).",
        avoid='another version of an earlier image ("one more like it": call generate_image again with the earlier '
              "prompt and no seed)"
              + ("; removing watermarks (remove_watermark)" if wm.enabled else "")
              + "; only cutting out the subject (remove_background)"
              + ("; only enlarging (upscale_image)" if cfg.upscale.enabled else "") + ".",
        rules=(
            '- prompt: start with the operation ("Replace the ...", "Remove the ...", "Add a ... to ..."), then "keep '
            'everything else unchanged" (describing the kept parts in detail makes them drift). Put new text in double '
            "quotes. <image1> is the picture being edited.",
            "- size: leave aspect_ratio, width and height unset to keep the shape of <image1>; change them only to "
            f"extend the scene. {edit_size}",
            steps_line,
            cfg_line(g.edit_cfg_scale, edit=True),
            neg_line(g.edit_negative_prompt, g.edit_cfg_scale),
            "- seed: leave unset. Never reuse the seed that produced the input image: it gives over-saturated, "
            "fragmented results.",
        ), job=True)

    pano = _describe(
        "Create a 360-degree equirectangular panorama (2:1) with Photo Sphere metadata.",
        "the user wants a 360 view, a VR scene or a spherical panorama, from a prompt or by extending a photo "
        "(image).",
        _example(prompt="a misty pine forest at sunrise, a wooden cabin with smoke from the chimney in front, a calm "
                        "lake behind, tall pines left and right, mossy ground, pale sky, soft golden light"),
        "the panorama, its link, its file path and a link to an interactive 360 viewer. It takes a minute or more.",
        avoid='a normal wide image (generate_image with aspect_ratio "21:9").',
        rules=(
            "- prompt: describe the scene in every direction: the ground, the horizon, the sky or ceiling, and what is "
            "in front, left, right and behind. Do not write \"360\", \"panorama\" or \"equirectangular\" (the server "
            "adds that).",
            f"- Leave steps, cfg_scale, width and seam_fix unset (defaults here: {steps} steps, "
            f"{cfg.panorama_size[0]}x{cfg.panorama_size[1]}; the wrap seam is repaired automatically).",
        ), job=True)

    background = _describe(
        "Cut out the main subject of an image onto a transparent background.",
        "the user wants the background removed from an existing image (a photo, a result or an upload).",
        _example(image=EXAMPLE_FILE),
        "a PNG with an alpha channel (WebP if asked), its link and its file path.",
        avoid="a new image with a transparent background (generate_image with transparent=true).")

    if wm.restore_unchanged:
        wm_effect = "only the areas the model changed are replaced and the rest stays as it was, at the input's size"
    else:
        wm_effect = f"the whole image is redrawn at about {fmt(wm.megapixels)} MP and returned at the input's size"
    watermark = _describe(
        "Remove watermarks from an image: logos, stamps, signatures, copyright lines and overlaid or tiled text.",
        "the user wants overlaid watermarks, logos or stamped text removed from a picture. It uses a model trained "
        "for this.",
        _example(image=EXAMPLE_FILE),
        f"the cleaned image, its link and its file path: {wm_effect}. It takes about as long as edit_image.",
        avoid="other changes to the picture (edit_image).", job=True)

    upscale = _describe(
        "Enlarge an image 2x or 4x with an AI super-resolution model (ESRGAN): sharper detail than plain resizing, "
        "same content.",
        "the user wants a bigger or sharper version of a finished image, e.g. for print or a wallpaper.",
        _example(image=EXAMPLE_FILE, scale=4),
        "the enlarged image (at most 8192 px per side), its link and its file path. Transparency is kept, and a 360 "
        "panorama stays a seamless 360 panorama with its viewer link. It takes seconds to a minute.",
        avoid="changing the content (edit_image).", job=True)

    get_job = _describe(
        "Get the result of an image job.",
        'a tool replied "Not finished yet" with a job_id. If a call was cut off before you saw its job_id, call '
        "get_job without arguments to list the recent jobs.",
        _example(job_id=EXAMPLE_JOB),
        "the image(s) as soon as the job is done (it waits a short time first); if the job is still running, its "
        "status and the next call to make. Results are kept for 24 hours.",
        avoid="starting a new image: it never starts a render.")

    cancel = _describe(
        "Cancel an image job.",
        "the user wants to stop a queued or running job.",
        _example(job_id=EXAMPLE_JOB),
        "the job's status. If the job already started, it finishes in the background and the result is discarded.")

    list_images = _describe(
        "List the recent images on this server, newest first: results, uploads (from the /upload page, under "
        "uploads/) and inputs-folder files.",
        "you need an image's file path, e.g. the user uploaded one at /upload (take the newest uploads/ entry).",
        _example(),
        "entries with 'file' and 'url'; pass either as an input image to the other tools.")

    view = _describe(
        "Look at an image stored on this server at full size.",
        "you need to see an upload or an image not shown in this chat.",
        _example(image=EXAMPLE_FILE),
        "the original image with its size and link.",
        avoid="images attached in the chat (you already see them).")

    status = _describe(
        "Report the server's state: model, device, download and loading progress, running jobs and default settings.",
        "a tool says the server is not ready, or the user asks about the server.",
        _example(),
        'the state ("starting", "downloading", "loading", "ready" or "error") and the details.')

    instructions = "\n".join([
        "Local image generation and editing with Qwen-Image-2.1.",
        "Input images: this server's link or file path exactly as a result or list_images gave it, with its date "
        'folder (e.g. "2026-09-23/image-021530-ab12cd34.png"), or a URL, data URL or base64. If the user says they '
        "uploaded an image (at /upload), call list_images and use the newest uploads/ entry.",
        f'{timing} If a call returns "Not finished yet", call get_job with its job_id; never call the same tool again '
        "to get that image (it would render it twice). If a call was cut off before you saw a job_id, call "
        "get_job with no arguments.",
    ])
    return {"instructions": instructions, "generate": generate, "edit": edit, "panorama": pano,
            "background": background, "watermark": watermark, "upscale": upscale, "get_job": get_job,
            "cancel_job": cancel, "list_images": list_images, "view_image": view, "server_status": status}


# Optional request headers (e.g. injected by an MCP bridge) that adapt results to a client.
INLINE_HEADER = "x-imagegen-inline"  # "data-uri-text": images as data-URI text lines (llama.cpp server-side MCP)
MAX_WAIT_HEADER = "x-imagegen-max-wait"  # how long a call waits before returning a job_id (also ?max_wait=N)
MAX_WAIT_LIMIT = 280  # seconds; longer waits run into common client limits (300 s) with little gain
INLINE_MAX_HEADER = "x-inline-max-bytes"  # raise the inline image budget for clients without message limits
VIEWER_PREVIEW_BYTES = 100_000  # in-chat viewer images: hosts may drop tool results over ~150,000 characters
# Image tools declare the in-chat viewer; hosts without MCP Apps ignore it. "ui/resourceUri" is the older flat key.
UI_META = {"ui": {"resourceUri": viewer.URI}, "ui/resourceUri": viewer.URI}


def _header(ctx, name: str) -> str:
    try:
        headers = ctx.headers if ctx is not None else None
    except Exception:  # noqa: BLE001 - no HTTP request (stdio, in-process tests)
        headers = None
    return (headers.get(name) or "").strip() if headers else ""


_B64_RUN = re.compile(r"[A-Za-z0-9+/=_-]{120,}")


def _clip(msg: str, limit: int = 600) -> str:
    """Keep error text short: never echo base64 payloads or huge paths back into the model's context."""
    msg = _B64_RUN.sub(lambda m: m.group(0)[:24] + f"...({len(m.group(0))} chars)", msg)
    return msg if len(msg) <= limit else msg[:limit] + f"...({len(msg)} chars)"


def _uint(value: str) -> int | None:
    """A plain non-negative ASCII integer, or None (junk values such as '²' are ignored)."""
    return int(value) if value.isascii() and value.isdigit() and len(value) < 10 else None


def _label(args: dict) -> str:
    """A short description of a request for job listings: the start of the prompt, or the input image's name (never
    inline image data)."""
    p = args.get("prompt")
    if isinstance(p, str) and p.strip():
        p = " ".join(p.split())
        return p[:80] + ("..." if len(p) > 80 else "")
    src = args.get("image") or (args.get("images") or [None])[0]
    if isinstance(src, str) and not src.startswith("data:") and len(src) < 300:
        return "image " + src.rstrip("/").rsplit("/", 1)[-1][:60]
    return "an uploaded image" if src else ""


def _query(ctx, name: str) -> str:
    try:
        return (ctx.request_context.request.query_params.get(name) or "").strip()
    except Exception:  # noqa: BLE001 - no HTTP request (stdio, in-process tests)
        return ""


def fingerprint(kind: str, args: dict) -> str:
    """A stable key for "the same request": the tool and its arguments, with long values (inline image data)
    replaced by their hash."""
    def norm(v):
        if isinstance(v, str) and len(v) > 256:
            return "sha256:" + hashlib.sha256(v.encode("utf-8", "replace")).hexdigest()
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v
    body = json.dumps({"kind": kind, **{k: norm(v) for k, v in sorted(args.items())}}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def _about(seconds: float) -> str:
    if seconds < 90:
        return f"about {max(10, int(round(seconds / 10)) * 10)} s"
    return f"about {int(round(seconds / 60))} min"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def create_server(service: ImageService) -> MCPServer:
    cfg: Config = service.cfg
    guide = _guide(cfg)

    # ------------------------------------------------------------------ in-chat viewer (MCP Apps)
    apps = Apps()
    apps.add_html_resource(viewer.URI, viewer.HTML, name="Image viewer",
                           description="Shows an image job's progress and the finished image in the chat.",
                           prefers_border=False)

    @apps.tool(resource_uri=viewer.URI, visibility=["app"], name="job_status", title="Job status (image viewer)",
               description="Used only by the in-chat image viewer. To get an image, call get_job instead.",
               annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def job_status(job_id: Annotated[str, Field(description="The job_id.")]) -> CallToolResult:
        job = service.jobs.get(job_id)
        nxt = json.dumps({"job_id": job_id.strip()})
        if job is None:  # e.g. a reopened conversation after a restart or 24 h: not an error for the viewer
            return CallToolResult(content=[TextContent(type="text", text=(
                "This job is no longer on the server (results are kept for 24 hours). The image links in the chat "
                "still work, and list_images shows the saved images."))],
                structured_content={"job_id": job_id, "status": "expired"})
        st = job.summary()
        if job.status == "running":
            return CallToolResult(content=[TextContent(type="text", text=(
                f"Status: {_status_line(job)}. Not finished yet. To get the image, call get_job with {nxt} (it waits "
                "for the job). Do NOT re-render the image again."))], structured_content=st)
        if job.status != "completed":
            return _error(job.error or f"job {job.status}")
        # Small previews only (the viewer shows them inline; the links have the full files), built off the event loop
        # once per job. Not marked delivered: the model may still fetch the result with get_job.
        if job.viewer_previews is None:
            job.viewer_previews = await asyncio.to_thread(
                lambda: [_fit(s, VIEWER_PREVIEW_BYTES, chat_safe=False) for s in job.result.images])
        blocks, images, lines = [], [], []
        for s, (data, mime) in zip(job.result.images, job.viewer_previews):
            blocks.append(ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType=mime))
            images.append({"url": s.url, "file": s.filename, "width": s.width, "height": s.height,
                           "viewer_url": s.view_url})
            lines.append(f"Image ready: {s.url} (file for edit_image and other tools: {s.filename})")
        lines.append(f"The inline image is a small preview; get_job with {nxt} returns the full result.")
        return CallToolResult(content=[*blocks, TextContent(type="text", text="\n".join(lines))],
                              structured_content={**st, "images": images, "delivered": job.delivered})

    mcp = MCPServer(name="imagegen-mcp", title="Image Gen MCP", version=__version__,
                    instructions=guide["instructions"], log_level=cfg.server.log_level, extensions=[apps])

    # ------------------------------------------------------------------ results
    def _fit(s, budget: int, chat_safe: bool) -> tuple[bytes, str]:
        """A preview of ``s`` that fits ``budget`` bytes. chat_safe: JPEG/PNG only (llama.cpp cannot decode WebP)."""
        side = cfg.outputs.preview_max_side
        make = imaging.chat_preview if chat_safe else (
            lambda im, sd: imaging.preview(im, sd, keep_alpha=im.mode == "RGBA"))
        img = s.image  # decoded once: SavedImage.image may read the file from disk on every access
        data, mime = make(img, side)
        while len(data) > budget and side > 256:
            side = int(side * 0.75)
            data, mime = make(img, side)
        return data, mime

    def _result(op: OpResult, *, data_uri_text: bool = False, inline_max: int | None = None) -> CallToolResult:
        """Build the tool result.

        Normal clients get ImageContent. data_uri_text (llama.cpp server-side MCP, which drops ImageContent)
        puts each image on its own text line as a data URI, which the llama.cpp web UI shows as an image.
        """
        image_blocks = []
        structured_images = []
        mode = cfg.outputs.inline
        # Many clients reject a single SSE message over 1 MiB (the official Python SDK does), so by default
        # the inline data of all images together stays under inline_max_bytes; the link has the full file.
        budget = (inline_max or cfg.outputs.inline_max_bytes) // max(1, len(op.images))
        for s in op.images:
            if mode == "none":
                inline = "none"
            else:
                if data_uri_text:
                    if s.size_bytes <= budget and s.mime in ("image/png", "image/jpeg"):
                        data, mime, inline = s.data, s.mime, "full"
                    else:
                        (data, mime), inline = _fit(s, budget, chat_safe=True), "preview"
                    b64 = base64.b64encode(data).decode()
                    image_blocks.append(TextContent(type="text", text=f"data:{mime};base64,{b64}"))
                else:
                    if mode == "full" or (mode == "auto" and s.size_bytes <= budget):
                        data, mime, inline = s.data, s.mime, "full"
                    else:
                        (data, mime), inline = _fit(s, budget, chat_safe=False), "preview"
                    image_blocks.append(ImageContent(type="image", data=base64.b64encode(data).decode(),
                                                     mimeType=mime))
                if inline == "preview":
                    # Clients may send the preview back as an input: map it to the full-resolution file.
                    service.previews.register(data, s.filename)
            item = {"url": s.url, "file": s.filename, "width": s.width, "height": s.height, "mime_type": s.mime,
                    "bytes": s.size_bytes, "inline": inline}
            if s.view_url:
                item["viewer_url"] = s.view_url
            structured_images.append(item)
        lines = []
        for item in structured_images:
            if data_uri_text:
                # No markdown syntax in these lines: the llama.cpp UI only draws the image when the whole
                # tool result is classified as plain text.
                lines.append(f"Image ready: {item['width']}x{item['height']} "
                             f"{item['mime_type'].split('/')[1].upper()}. It is already shown to the user in the chat.")
                lines.append(f"File name for edit_image and other tools: {item['file']}")
                lines.append(f"Full-resolution link: {item['url']}")
                if item.get("viewer_url"):
                    lines.append(f"Interactive 360 viewer: {item['viewer_url']}")
                lines.append("To show it in your reply, embed the full-resolution link as a markdown image.")
            else:
                lines.append(f"Saved {item['width']}x{item['height']} {item['mime_type'].split('/')[1].upper()}: "
                             f"{item['url']}")
                if item.get("viewer_url"):
                    lines.append(f"Open in the 360 viewer: {item['viewer_url']}")
                if item["inline"] == "preview":
                    lines.append("(the inline image is a reduced preview; the link has the full resolution file)")
                lines.append(f"(to edit it further, pass \"{item['file']}\" as an input image)")
        info = dict(op.info)
        if isinstance(info.get("prompt"), str) and len(info["prompt"]) > 400:
            info["prompt"] = info["prompt"][:400] + "..."
        if info:
            lines.append("Details: " + json.dumps(info, separators=(", ", ": ")))
        lines += [f"Note: {n}" for n in op.notes]
        text = TextContent(type="text", text="\n".join(lines))
        content = [text, *image_blocks] if data_uri_text else [*image_blocks, text]
        return CallToolResult(content=content,
                              structured_content={"images": structured_images, "info": op.info, "notes": op.notes})

    def _error(msg: str) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=msg)], is_error=True)

    def _format_error(exc: BaseException) -> str:
        if isinstance(exc, ServiceUnavailable):
            msg = str(exc)
        elif isinstance(exc, (ImageInputError, ValueError)):
            msg = f"Invalid input: {exc}"
        elif isinstance(exc, EngineError):
            msg = f"Generation failed: {exc}"
        else:
            msg = f"Internal error: {exc.__class__.__name__}: {exc}"
        return _clip(msg)

    service.jobs.format_error = _format_error
    mcp.build_result = _result  # exposed for tests

    # ------------------------------------------------------------------ jobs
    def _progress(ctx: Context):
        async def cb(frac: float, message: str) -> None:
            try:
                await ctx.report_progress(round(frac * 100, 1), 100, message)
            except Exception:  # noqa: BLE001 - progress is best effort
                pass
        return cb

    def _budget(ctx: Context | None) -> tuple[int, str]:
        """How long a call waits before it returns a job_id: chosen by the server (generation.wait_seconds), or by
        the user for their client with ?max_wait=N on the endpoint URL or the X-Imagegen-Max-Wait header. The model
        does not choose it: it cannot know the client's time limit."""
        wait, source = cfg.generation.wait_seconds, "default"
        for value, src in ((_query(ctx, "max_wait"), "url"), (_header(ctx, MAX_WAIT_HEADER), "header")):
            n = _uint(value)
            if n is not None:
                wait, source = n, src
        return max(0, min(wait, MAX_WAIT_LIMIT)), source

    def _status_line(job: Job) -> str:
        place, eta = job.place()
        if job.status != "running":
            return job.status
        if place:
            text = f"{_ordinal(place)} in line"
        elif job.progress > 0:
            text = f"working, {round(job.progress * 100)}% done"
        else:
            text = "starting"
        if eta is not None:
            text += f", {_about(eta)} left" if eta > 5 else ", should finish soon"
        return text

    def _pending(jobs: list[Job], budget: int, reused: bool = False) -> CallToolResult:
        """The reply while jobs are still running: a normal result (not an error) that tells the model exactly what
        to do next, in the text and in the structured content (some clients show the model only one of them)."""
        n = max(j.polls for j in jobs) + 1
        for j in jobs:
            j.announced = True  # the client has the id now: an identical call later is a new request
        if len(jobs) == 1:
            job = jobs[0]
            args: dict = {"job_id": job.id, "poll": n}
            if reused:
                ago = _about(time.time() - job.created).replace("about ", "")
                first = (f"This exact request is already running as job {job.id} (started {ago} ago), so no second "
                         "render was started.")
            else:
                first = "Not finished yet. This is normal: nothing failed and the job keeps running on the server."
            status = f"Status: {_status_line(job)}."
        else:
            args = {"job_ids": [j.id for j in jobs], "poll": n}
            first = "Not finished yet. This is normal: nothing failed and the jobs keep running on the server."
            status = "Status: " + "; ".join(f"{j.id} {_status_line(j)}" for j in jobs) + "."
        call = json.dumps(args)
        lines = [first,
                 f"NEXT STEP: call get_job with {call}. It waits up to {budget} s and returns the image as soon as it "
                 "is ready.",
                 "Do NOT re-render the image again.",
                 status]
        summaries = [j.summary() for j in jobs]
        structured: dict = {"status": "running", "reused": reused, "wait_budget_seconds": budget,
                            "next_step": f"Call get_job with {call}. Do NOT re-render the image again.",
                            "next": {"tool": "get_job", "arguments": args}}
        if len(jobs) == 1:
            st = summaries[0]
            structured.update(job_id=st["job_id"], kind=st["kind"], progress_percent=st["progress_percent"],
                              queue_position=st["queue_position"], eta_seconds=st["eta_seconds"],
                              elapsed_seconds=st["elapsed_s"], message=st["message"])
        else:
            structured["jobs"] = summaries
        return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))], structured_content=structured)

    def _finished(job: Job, ctx: Context | None, share: float = 1.0) -> CallToolResult:
        """The finished job's reply. share: this job's part of the inline image budget when one reply carries several
        jobs (all images of one reply together stay under the client's message limit)."""
        job.announced = True
        if job.status == "completed":
            cap = _uint(_header(ctx, INLINE_MAX_HEADER))
            total = max(10_000, min(cap, 64 * 1024 * 1024)) if cap is not None else cfg.outputs.inline_max_bytes
            r = _result(job.result, data_uri_text=_header(ctx, INLINE_HEADER).lower() == "data-uri-text",
                        inline_max=max(1, int(total * share)))
            r.structured_content["job_id"] = job.id
            r.structured_content["status"] = "completed"
            job.delivered = True
            return r
        if job.status == "failed":
            return _error(f"{job.error or 'the job failed'}\nDo not retry automatically: tell the user what happened "
                          "and ask before trying again.")
        return _error(f"job {job.id} was {job.status}")

    async def _collect(job: Job, ctx: Context | None, reused: bool = False) -> CallToolResult:
        budget, _ = _budget(ctx)
        done = await service.jobs.wait(job, budget, _progress(ctx) if ctx is not None else None)
        return _finished(job, ctx) if done else _pending([job], budget, reused)

    def _client(ctx: Context | None) -> str:
        """Who sent the request, for the job log: address (and forwarded-for) plus User-Agent."""
        addr = "?"
        try:
            req = ctx.request_context.request
            addr = req.client.host if req is not None and req.client else "?"
        except Exception:  # noqa: BLE001 - no HTTP request (stdio, in-process tests)
            pass
        fwd, ua = _header(ctx, "x-forwarded-for"), _header(ctx, "user-agent")
        return f"{addr}{f' (forwarded for {fwd})' if fwd else ''} ua={ua[:80]!r}"

    def _caller(ctx: Context | None) -> str:
        """Who is asking, so a retry matches its own job and not another client's: the first forwarded-for address
        (or the peer address) and the User-Agent."""
        addr = _header(ctx, "x-forwarded-for").split(",")[0].strip()
        if not addr:
            try:
                req = ctx.request_context.request
                addr = req.client.host if req is not None and req.client else ""
            except Exception:  # noqa: BLE001 - no HTTP request (stdio, in-process tests)
                addr = ""
        return f"{addr}|{_header(ctx, 'user-agent')}"

    async def _start(kind: str, factory, ctx: Context, args: dict, detail: str = "",
                     require_engine: bool = True) -> CallToolResult:
        """Start a job, or attach to the identical request that is already running (a retry after the client gave
        up), then wait for it up to the server's wait budget."""
        if require_engine:
            try:
                service.require_ready()
            except ServiceUnavailable as exc:
                return _error(str(exc))
        _remember_base_url(ctx)
        budget, source = _budget(ctx)
        key = fingerprint(kind, {**args, "_caller": _caller(ctx)})
        job = service.jobs.find_reusable(key)
        if job is not None:
            log.info("job %s %s: identical request attached to the running job (wait %ss from %s) from %s",
                     job.id, kind, budget, source, _client(ctx))
            return await _collect(job, ctx, reused=True)
        job = service.jobs.submit(kind, factory, key=key, label=_label(args))
        log.info("job %s %s%s wait=%ss(%s) from %s", job.id, kind, f" [{detail}]" if detail else "", budget, source,
                 _client(ctx))
        return await _collect(job, ctx)

    def _remember_base_url(ctx: Context | None) -> None:
        try:
            REQUEST_BASE_URL.set(base_url_from_headers(ctx.headers if ctx is not None else None))
        except Exception:  # noqa: BLE001 - fall back to the configured/default base URL
            pass

    # ------------------------------------------------------------------ tools
    @mcp.tool(description=guide["generate"],
              meta=UI_META, annotations=ToolAnnotations(title="Generate image", readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=False, openWorldHint=False))
    async def generate_image(
        prompt: Annotated[str, Field(description="Detailed description of the image; text to render in double quotes.")],
        ctx: Context,
        aspect_ratio: Annotated[AspectRatio | None, Field(description="Aspect ratio, e.g. 16:9. Ignored when both width and height are given.")] = None,
        size: Size = None,
        width: Width = None,
        height: Height = None,
        transparent: Annotated[bool, Field(description="Transparent background (alpha channel). Describe only the subject, with no background, frame, badge or 'app icon' words.")] = False,
        tileable: Annotated[bool, Field(description="Seamless repeating texture: describe a surface that fills the frame. Edits and upscales of it stay seamless when its file is passed first. Not with transparent.")] = False,
        negative_prompt: Negative = "",
        steps: Steps = None,
        cfg_scale: Guidance = None,
        seed: Seed = None,
        output_format: OutputFormat = None,
    ) -> CallToolResult:
        args = dict(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height,
                    aspect_ratio=aspect_ratio, size=size, transparent=transparent, steps=steps, cfg_scale=cfg_scale,
                    seed=seed, output_format=output_format, tileable=tileable)
        return await _start("generate", lambda progress: service.generate(**args, progress=progress), ctx, args,
            f"size={size} {width}x{height} ar={aspect_ratio} steps={steps} transparent={transparent} tileable={tileable}")

    @mcp.tool(description=guide["edit"],
              meta=UI_META, annotations=ToolAnnotations(title="Edit or combine images", readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=False, openWorldHint=True))
    async def edit_image(
        prompt: Annotated[str, Field(description="The edit instruction: the operation, then 'keep everything else unchanged'. Refer to inputs as <image1>, <image2>, ...")],
        images: Annotated[list[str], Field(min_length=1, max_length=10, description=f"1-10 input images: {IMAGE_INPUT}. The first is the one edited and sets the output shape.")],
        ctx: Context,
        mask: Annotated[str | None, Field(description="Optional mask image: white = change, black = keep (transparent counts as black). A guide, not a hard pixel mask: areas outside it can shift slightly. Counts as one of the 10 inputs.")] = None,
        aspect_ratio: Annotated[AspectRatio | None, Field(description="Output aspect ratio (default: same as the first image).")] = None,
        size: Size = None,
        width: EditWidth = None,
        height: EditHeight = None,
        transparent: Annotated[bool, Field(description="Put the edited result on a transparent background (prompt e.g. 'Extract the logo from the shirt').")] = False,
        negative_prompt: Negative = "",
        steps: Steps = None,
        cfg_scale: Guidance = None,
        seed: EditSeed = None,
        output_format: OutputFormat = None,
        tileable: Annotated[bool | None, Field(description="Leave unset: a tile from this server passed as <image1> is detected and stays seamless at the same size (e.g. for height or normal maps). true: treat <image1> as a tile from elsewhere. false: a normal edit.")] = None,
    ) -> CallToolResult:
        args = dict(prompt=prompt, images=images, mask=mask, negative_prompt=negative_prompt, width=width,
                    height=height, aspect_ratio=aspect_ratio, size=size, transparent=transparent, steps=steps,
                    cfg_scale=cfg_scale, seed=seed, output_format=output_format, tileable=tileable)
        return await _start("edit", lambda progress: service.edit(**args, progress=progress), ctx, args,
            f"{len(images)} image(s) mask={mask is not None} size={size} {width}x{height} ar={aspect_ratio} steps={steps}"
            + (f" tileable={tileable}" if tileable is not None else ""))

    @mcp.tool(description=guide["panorama"],
              meta=UI_META, annotations=ToolAnnotations(title="Generate 360 panorama", readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=False, openWorldHint=True))
    async def generate_panorama(
        prompt: Annotated[str, Field(description="The scene only, in every direction. No '360' or 'panorama' words.")],
        ctx: Context,
        image: Annotated[str | None, Field(description="Optional photo to extend into a full 360 panorama.")] = None,
        width: Annotated[int | None, Field(ge=512, le=4096, description="Width, rounded to a multiple of 64; height = width/2. Leave unset.")] = None,
        seam_fix: Annotated[bool | None, Field(description="Repair the left/right wrap seam (default on).")] = None,
        negative_prompt: Negative = "",
        steps: Steps = None,
        cfg_scale: Guidance = None,
        seed: Seed = None,
        output_format: Annotated[Fmt | None, Field(description="jpeg (default, best viewer support), png or webp.")] = None,
    ) -> CallToolResult:
        args = dict(prompt=prompt, image=image, negative_prompt=negative_prompt, width=width, steps=steps,
                    cfg_scale=cfg_scale, seed=seed, seam_fix=seam_fix, output_format=output_format)
        return await _start("panorama", lambda progress: service.panorama(**args, progress=progress), ctx, args,
            f"width={width} from_image={image is not None} seam_fix={seam_fix} steps={steps}")

    @mcp.tool(description=guide["background"],
              meta=UI_META, annotations=ToolAnnotations(title="Remove background", readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=True, openWorldHint=True))
    async def remove_background(
        image: Annotated[str, Field(description=f"Image to cut out: {IMAGE_INPUT}.")],
        ctx: Context,
        output_format: Annotated[Literal["png", "webp"], Field(description="png (default) or webp; both keep the alpha channel.")] = "png",
    ) -> CallToolResult:
        if service.matter is None:
            return _error("background removal is not available yet (models are still loading) or it is disabled "
                          "(transparency.method is 'native')")
        args = dict(image=image, output_format=output_format)
        return await _start("remove_background", lambda progress: service.remove_background(**args, progress=progress),
                            ctx, args, require_engine=False)

    if cfg.watermark.enabled:
        @mcp.tool(description=guide["watermark"],
                  meta=UI_META, annotations=ToolAnnotations(title="Remove watermarks", readOnlyHint=False, destructiveHint=False,
                                              idempotentHint=False, openWorldHint=True))
        async def remove_watermark(
            image: Annotated[str, Field(description=f"Image to clean: {IMAGE_INPUT}.")],
            ctx: Context,
            seed: EditSeed = None,
            output_format: OutputFormat = None,
        ) -> CallToolResult:
            args = dict(image=image, seed=seed, output_format=output_format)
            return await _start("remove_watermark", lambda progress: service.remove_watermark(**args, progress=progress),
                                ctx, args)

    if cfg.upscale.enabled:
        @mcp.tool(description=guide["upscale"],
                  meta=UI_META, annotations=ToolAnnotations(title="Upscale image", readOnlyHint=False, destructiveHint=False,
                                              idempotentHint=True, openWorldHint=True))
        async def upscale_image(
            image: Annotated[str, Field(description=f"Image to enlarge: {IMAGE_INPUT}.")],
            ctx: Context,
            scale: Annotated[Literal[2, 4], Field(description="Enlargement factor: 4 (default) or 2.")] = 4,
            output_format: Annotated[Fmt | None, Field(description="Saved file format (default png; jpeg for 360 panoramas, much smaller for very large results).")] = None,
            panorama: Annotated[bool | None, Field(description="Leave unset (360 panoramas are detected). true: treat a 2:1 image as one. false: plain upscale.")] = None,
            tileable: Annotated[bool | None, Field(description="Leave unset (this server's seamless tiles are detected). true: treat it as a tile. false: plain upscale.")] = None,
        ) -> CallToolResult:
            args = dict(image=image, scale=scale, output_format=output_format, as_panorama=panorama,
                        as_tileable=tileable)
            return await _start("upscale", lambda progress: service.upscale(**args, progress=progress), ctx, args,
                f"scale={scale}"
                + (f" panorama={panorama}" if panorama is not None else "")
                + (f" tileable={tileable}" if tileable is not None else ""))

    @mcp.tool(description=guide["get_job"],
              annotations=ToolAnnotations(title="Get job result", readOnlyHint=True, openWorldHint=False))
    async def get_job(
        ctx: Context,
        job_id: Annotated[str | None, Field(description="The job_id from a 'Not finished yet' reply. Omit job_id and job_ids to list the jobs of the last 30 minutes.")] = None,
        job_ids: Annotated[list[str] | None, Field(max_length=20, description="Several job_ids at once.")] = None,
        poll: Annotated[int | None, Field(description="Optional counter from the last reply's next step.")] = None,
    ) -> CallToolResult:
        ids = list(dict.fromkeys(i.strip() for i in [*([job_id] if job_id else []), *(job_ids or [])] if i.strip()))
        if not ids:
            recent = service.jobs.recent()
            lines = [f"{j.id}: {j.kind}" + (f' "{j.label}"' if j.label else "")
                     + f", {_status_line(j) if j.status == 'running' else j.status}"
                     + (", already delivered" if j.delivered else "")
                     + f", started {_about(time.time() - j.created).replace('about ', '')} ago" for j in recent]
            text = ("Jobs of the last 30 minutes (newest first). Call get_job with a job_id to get its image:\n"
                    + "\n".join(lines)) if lines else "No jobs in the last 30 minutes."
            return CallToolResult(content=[TextContent(type="text", text=text)],
                                  structured_content={"jobs": [j.summary() for j in recent]})
        jobs = [service.jobs.get(i) for i in ids]
        missing = [i for i, j in zip(ids, jobs) if j is None]
        jobs = [j for j in jobs if j is not None]
        if not jobs:
            recent = ", ".join(j.id for j in service.jobs.recent()[:10]) or "none"
            return _error(f"unknown job_id {', '.join(map(repr, missing))} (finished jobs are kept for 24 hours). "
                          f"Recent jobs: {recent}. Call get_job without a job_id to list them.")
        for j in jobs:
            j.polls += 1
        if len(jobs) == 1 and not missing:
            return await _collect(jobs[0], ctx)
        budget, _ = _budget(ctx)
        open_jobs = [j for j in jobs if not j.done.is_set()]
        if open_jobs and len(open_jobs) == len(jobs):  # nothing finished yet: wait for the first one
            waiters = [asyncio.ensure_future(j.done.wait()) for j in open_jobs]
            try:
                await asyncio.wait(waiters, timeout=budget, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for w in waiters:
                    w.cancel()
        done = [j for j in jobs if j.done.is_set()]
        still = [j for j in jobs if not j.done.is_set()]
        content, entries = [], []
        n_img = sum(len(j.result.images) for j in done if j.status == "completed") or 1
        for j in done:
            r = _finished(j, ctx, share=len(j.result.images) / n_img if j.status == "completed" else 1.0)
            content += [TextContent(type="text", text=f"Job {j.id}:")] + list(r.content)
            entry = {"job_id": j.id, "status": j.status}
            if j.status == "completed":
                entry.update(images=r.structured_content["images"], info=r.structured_content["info"])
            else:
                entry["error"] = j.error
            entries.append(entry)
        structured: dict = {"jobs": entries + [j.summary() for j in still]}
        if missing:
            content.append(TextContent(type="text", text=f"Unknown job_id(s): {', '.join(missing)}."))
            structured["unknown"] = missing
        if still:
            p = _pending(still, budget)
            content += list(p.content)
            structured.update(next=p.structured_content["next"], next_step=p.structured_content["next_step"])
        return CallToolResult(content=content, structured_content=structured)

    @mcp.tool(description=guide["cancel_job"],
              annotations=ToolAnnotations(title="Cancel job", readOnlyHint=False, destructiveHint=True,
                                          idempotentHint=True, openWorldHint=False))
    async def cancel_job(job_id: Annotated[str, Field(description="The job_id from a reply, or from get_job with no arguments.")]) -> dict:
        job = service.jobs.cancel(job_id)
        if job is None:
            return {"error": f"unknown job_id {job_id!r}"}
        await asyncio.sleep(0)
        return job.summary()

    @mcp.tool(description=guide["list_images"],
              annotations=ToolAnnotations(title="List images", readOnlyHint=True, openWorldHint=False))
    async def list_images(
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=200, description="Maximum entries (results and uploads share the list).")] = 30,
    ) -> dict:
        _remember_base_url(ctx)
        return service.list_images(limit)

    @mcp.tool(description=guide["view_image"],
              annotations=ToolAnnotations(title="View image", readOnlyHint=True, openWorldHint=False))
    async def view_image(
        image: Annotated[str, Field(description="File path or link from a result or list_images, or its bare file name.")],
        ctx: Context,
    ) -> CallToolResult:
        _remember_base_url(ctx)
        path = await asyncio.to_thread(service.loader.local_path, image)
        if path is None:
            return _error(_clip(f"no image {image!r} on this server. Pass a file path or link from a result or "
                                "list_images. Images attached in the chat are not on this server; use the chat's "
                                "own handle (e.g. img_1) for those."))
        limit = cfg.server.max_request_mb * 1024 * 1024
        if path.stat().st_size > limit:
            return _error(f"{path.name} is larger than {cfg.server.max_request_mb} MB; open its link instead")

        def read() -> tuple[bytes, str, int, int]:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                fmt, size = im.format, im.size
                if fmt in ("PNG", "JPEG"):
                    return data, f"image/{fmt.lower()}", *size
                buf = io.BytesIO()  # WebP, GIF, ...: many clients cannot decode them, so send lossless PNG
                im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else "RGB").save(buf, "PNG")
                return buf.getvalue(), "image/png", *size

        try:
            data, mime, width, height = await asyncio.to_thread(read)
        except Exception as exc:  # noqa: BLE001 - not an image, unreadable file
            return _error(_clip(f"{path.name} is not a readable image: {exc}"))
        info = {"file": path.name, "width": width, "height": height, "mime_type": mime, "bytes": len(data)}
        outputs = Path(cfg.outputs_dir).resolve()
        if path.is_relative_to(outputs):
            rel = path.relative_to(outputs).as_posix()
            info["file"] = rel
            if _output_file(rel) is not None:  # only files the /outputs route serves get a link
                info["url"] = f"{service.base_url()}/outputs/{rel}"
        text = f"{info['file']}: {width}x{height} {mime.split('/')[1].upper()}, {len(data)} bytes (original size)"
        if info.get("url"):
            text += f"\nLink: {info['url']}"
        return CallToolResult(content=[ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType=mime),
                                       TextContent(type="text", text=text)], structured_content=info)

    @mcp.tool(description=guide["server_status"],
              annotations=ToolAnnotations(title="Server status", readOnlyHint=True, openWorldHint=False))
    async def server_status() -> dict:
        st = service.status()
        st["running_jobs"] = [j.summary() for j in service.jobs.jobs.values() if j.status == "running"]
        return st

    # ------------------------------------------------------------------ HTTP routes
    outputs_root = Path(cfg.outputs_dir).resolve()

    def _output_file(rel: str) -> Path | None:
        """An image under outputs/. Hidden files (e.g. .previews.json) and other file types are never served."""
        if not servable(rel):
            return None
        try:
            p = (outputs_root / rel).resolve()
            ok = outputs_root in p.parents and servable(p.relative_to(outputs_root).as_posix()) and p.is_file()
        except (OSError, ValueError, RuntimeError):  # NUL byte, name too long, symlink loop: not served
            return None
        return p if ok else None

    @mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(request: Request) -> Response:
        st = service.status()
        want_ready = request.query_params.get("ready") in ("1", "true")
        ok = st["state"] != "error" and (st["state"] == "ready" or not want_ready)
        return JSONResponse({"status": st["state"], "error": st["error"], "download": st["download"]},
                            status_code=200 if ok else 503)

    @mcp.custom_route("/outputs/{path:path}", methods=["GET"], include_in_schema=False)
    async def outputs(request: Request) -> Response:
        p = _output_file(request.path_params["path"])
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        # CORP lets pages served with COEP (e.g. the llama.cpp web UI) show these images.
        return FileResponse(p, headers={"Cache-Control": "public, max-age=86400",
                                        "Cross-Origin-Resource-Policy": "cross-origin"})

    @mcp.custom_route("/view/{path:path}", methods=["GET"], include_in_schema=False)
    async def view(request: Request) -> Response:
        rel = request.path_params["path"]
        if _output_file(rel) is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        html = panorama.VIEWER_HTML.replace("__IMAGE_URL__", json.dumps(f"/outputs/{rel}"))
        return HTMLResponse(html)

    page = webui.PAGE.replace("__MCP_PATH__", cfg.server.path)

    @mcp.custom_route("/", methods=["GET"], include_in_schema=False)
    async def index(request: Request) -> Response:
        if "text/html" in request.headers.get("accept", ""):
            return HTMLResponse(page)
        st = service.status()
        return JSONResponse({"name": "imagegen-mcp", "version": __version__, "mcp_endpoint": cfg.server.path,
                             "transport": "streamable-http", "state": st["state"], "device": st["device"],
                             "model": st["model"], "placement": st["placement"]})

    @mcp.custom_route("/upload", methods=["GET"], include_in_schema=False)
    async def upload_page(request: Request) -> Response:
        return HTMLResponse(page)

    @mcp.custom_route("/upload", methods=["POST"], include_in_schema=False)
    async def upload(request: Request) -> Response:
        limit = cfg.server.max_request_mb * 1024 * 1024
        if int(request.headers.get("content-length") or 0) > limit:
            return JSONResponse({"error": f"file is larger than {cfg.server.max_request_mb} MB"}, status_code=413)
        try:
            # ?hash=1: content-addressed name, so re-uploading the same bytes returns the same reference.
            hashed = request.query_params.get("hash") in ("1", "true")
            if request.headers.get("content-type", "").startswith("multipart/form-data"):
                form = await request.form(max_part_size=limit)
                results = []
                for _, item in form.multi_items():
                    if hasattr(item, "read"):
                        data = await item.read()
                        results.append(await asyncio.to_thread(service.save_upload, data, item.filename or "",
                                                               hashed))
                if not results:
                    return JSONResponse({"error": "no file in the form"}, status_code=400)
                REQUEST_BASE_URL.set(base_url_from_headers(request.headers))
                results = [{**r, "url": f"{service.base_url()}/outputs/{r['file']}"} for r in results]
                return JSONResponse(results[0] if len(results) == 1 else {"files": results})
            data = await request.body()
            REQUEST_BASE_URL.set(base_url_from_headers(request.headers))
            res = await asyncio.to_thread(service.save_upload, data, request.query_params.get("name", ""), hashed)
            return JSONResponse(res)
        except (ImageInputError, ServiceUnavailable) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @mcp.custom_route("/api/status", methods=["GET"], include_in_schema=False)
    async def api_status(request: Request) -> Response:
        st = service.status()
        st["running_jobs"] = [j.summary() for j in service.jobs.jobs.values() if j.status == "running"]
        return JSONResponse(st)

    @mcp.custom_route("/api/images", methods=["GET"], include_in_schema=False)
    async def api_images(request: Request) -> Response:
        REQUEST_BASE_URL.set(base_url_from_headers(request.headers))
        return JSONResponse(service.list_images(60))

    return mcp


def build_app(service: ImageService):
    cfg = service.cfg
    mcp = create_server(service)
    if cfg.server.allowed_hosts:
        hosts = list(cfg.server.allowed_hosts)
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=hosts + [h if ":" in h else f"{h}:*" for h in hosts],
            allowed_origins=[f"http://{h}" for h in hosts] + [f"https://{h}" for h in hosts]
            + [f"http://{h}:*" for h in hosts if ":" not in h])
    else:
        # Reachable from any host name or IP on the network (no auth, trusted LAN use).
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = mcp.streamable_http_app(
        streamable_http_path=cfg.server.path,
        stateless_http=True,
        json_response=False,
        max_request_body_size=cfg.server.max_request_mb * 1024 * 1024,
        transport_security=security,
        host=cfg.server.host,
    )
    # Browser-based MCP clients (e.g. web UIs connecting directly) need CORS. There are no cookies or
    # credentials to protect: the server has no auth by design and should only run on trusted networks.
    app = CORSMiddleware(app, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                         allow_headers=["*"], expose_headers=["Mcp-Session-Id", "Mcp-Protocol-Version"],
                         max_age=86400)
    return mcp, app
