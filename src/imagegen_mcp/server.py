"""MCP server (Streamable HTTP) exposing the image tools, plus a few plain HTTP routes."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from PIL import Image
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response

from . import __version__, imaging, panorama, webui
from .config import Config
from .imaging import ImageInputError
from .jobs import Job
from .sdserver import EngineError
from .service import (REQUEST_BASE_URL, ImageService, OpResult, ServiceUnavailable, base_url_from_headers,
                      servable)

log = logging.getLogger("imagegen.server")

# Usage guidance the model sees. The descriptions are filled in with this server's configured defaults
# (see _guide), so they stay true when config.yaml changes. Sources: the official Qwen-Image-2.1 settings,
# community testing, and A/B runs of this server (see the README). Parameter descriptions carry no numbers
# that depend on the config; they point at the tool description instead.
INSTRUCTIONS = """\
Image generation and editing with Qwen-Image-2.1 (local, GGUF).
- generate_image: text-to-image in any size or aspect ratio; transparent=true gives a PNG with an alpha channel.
- edit_image: edit or combine 1-10 input images (refer to them as <image1>, <image2>, ... in the prompt).
  Use it to add, remove, replace or restyle elements or text, or to merge elements from several images.
- generate_panorama: 360-degree equirectangular (2:1) panoramas with Photo Sphere metadata and a viewer link.
- remove_background: cut out the subject of an image into a transparent PNG.
__WATERMARK_LINE____UPSCALE_LINE__- view_image: look at an image stored on this server (a result or upload) at its original size.
Input images can be data URLs, base64, http(s) URLs, or this server's images: pass the full link, or the file
path exactly as a result or list_images gave it, including its date folder (e.g. "2026-09-23/image-021530-ab12cd34.png").
Users can upload their own images on the server's web page (/upload); list_images shows what is available.
__TIMING__ If a call returns a "Still working" message with a job_id instead of an image, the job is still
running: call get_job with that job_id to wait for the result.

Quality guide:
- Prompts: detailed and explicit beats short keyword lists. Describe the subject, setting, composition,
  lighting, colors and medium or camera, and put text to render in double quotes. Never put resolutions,
  aspect ratios or words like 4K/8K in the prompt; use size and aspect_ratio.
__SIZE_LINE__
__STEPS_LINE__
__CFG_LINE__
__NEG_LINE__
- Another version of an earlier image ("one more like it", "same idea without the signs"): call generate_image
  again with the earlier prompt, changed only as asked, and omit seed. edit_image is for changing a picture.
- Edits: never reuse the seed that produced the input image; leave seed unset."""

UPSCALE_DESCRIPTION = (
    "Enlarge an image 2x or 4x with an AI super-resolution model (ESRGAN). It adds real-looking fine detail and "
    "sharp edges instead of the blur of plain resizing, and keeps the content, composition and colors as they are. "
    "Use it on finished images (generated, edited or uploaded) when the user wants a bigger or sharper version, for "
    "print or a wallpaper. The result is at most 8192 px per side: larger inputs are reduced first. Transparency is "
    "kept. A 360 panorama (for example from generate_panorama) stays a 360 panorama: the wrap-around edges stay "
    "seamless, the result keeps the Photo Sphere metadata and comes with a 360 viewer link. It takes seconds to "
    "about a minute, and longer for very large images. If a call returns a job_id, call get_job with it.")

AspectRatio = Literal["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "9:21", "2:1", "1:2", "5:4", "4:5"]
SizeTier = Literal["small", "medium", "large", "xl"]
Fmt = Literal["png", "webp", "jpeg"]
Steps = Annotated[int | None, Field(ge=1, le=100, description=(
    "Denoising steps. Leave unset: the default in the tool description is the best-quality setting. Use 25-30 "
    "only when the user asks for a quick draft, and never lower it on your own to save time. Above 50 is slower "
    "and rarely helps."))]
Size = Annotated[Literal["small", "medium", "large", "xl"] | None, Field(description=(
    "Resolution tier, i.e. the pixel count (aspect_ratio sets the shape): small ~0.26 MP, medium ~1 MP, large "
    "~2 MP, xl ~4 MP = native 2K, the sharpest. The tool description says which to use; sizes above the "
    "server's limit are scaled down."))]
WaitSeconds = Annotated[int | None, Field(ge=0, le=3600, description=(
    "Seconds to wait for the image before returning a job_id to poll with get_job. Leave unset for the "
    "server's default (usually 50)."))]
JobWait = Annotated[int | None, Field(ge=0, le=3600, description=(
    "Seconds to wait for the job to finish before returning its progress again. Leave unset for the server's "
    "default (usually 50)."))]
Guidance = Annotated[float | None, Field(ge=0, le=20, description=(
    "Guidance scale. Leave unset unless the user asks; the tool description gives this server's default. 3-5 "
    "gives cleaner, more coherent images that follow the prompt closely and uses the negative prompt; 1 is about "
    "twice as fast but hazier, follows the prompt less and ignores the negative prompt."))]
Negative = Annotated[str, Field(description=(
    "Extra things to avoid, comma-separated (e.g. 'text, signs, watermark'). They are added to the server's "
    "built-in negative prompt, so leave this empty unless the user wants something specific kept out. Neither "
    "is used when cfg_scale is 1 or less."))]
Seed = Annotated[int | None, Field(description=(
    "Random seed. Omit it for new images and for variations of an earlier one. Pass an earlier result's seed only "
    "to reproduce that exact image with the same prompt and settings."))]
EditSeed = Annotated[int | None, Field(description=(
    "Leave unset (random). Never pass the seed that produced the input image: re-editing with the same seed "
    "gives over-saturated, fragmented results."))]
Width = Annotated[int | None, Field(ge=256, le=4096, description=(
    "Width in pixels, rounded to a multiple of 32. With only width, the height follows aspect_ratio (square if "
    "unset). Overrides size. Sizes above the server's limit are scaled down."))]
Height = Annotated[int | None, Field(ge=256, le=4096, description=(
    "Height in pixels, rounded to a multiple of 32. With only height, the width follows aspect_ratio (square if "
    "unset). Overrides size. Sizes above the server's limit are scaled down."))]
EditWidth = Annotated[int | None, Field(ge=256, le=4096, description=(
    "Output width in pixels, rounded to a multiple of 32. With only width, the height follows aspect_ratio or "
    "the shape of <image1>. Overrides size. Leave unset to keep the shape of <image1>."))]
EditHeight = Annotated[int | None, Field(ge=256, le=4096, description=(
    "Output height in pixels, rounded to a multiple of 32. With only height, the width follows aspect_ratio or "
    "the shape of <image1>. Overrides size. Leave unset to keep the shape of <image1>."))]
JOB_LINE = ('If the call returns a "Still working" message with a job_id instead of an image, call get_job with '
            "that job_id until the image is ready.")


def _guide(cfg: Config) -> dict[str, str]:
    """Tool descriptions with this server's defaults filled in."""
    g = cfg.generation
    steps = cfg.default_steps
    t2i_cfg = cfg.default_cfg_scale
    cpu = cfg.is_cpu
    max_mp = cfg.max_pixels / 1e6

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
            why = ("because this server runs on a CPU, where guidance doubles the time per step" if cpu
                   else "because this server is configured for speed")
            extra = " For edits consider cfg_scale 4: at 1 the model often ignores the instruction." if edit else ""
            return (f"- cfg_scale: the default here is {fmt(value)} {why}. 3-5 gives cleaner, more coherent results "
                    f"and enables the negative prompt, at twice the time per step.{extra}")
        why = ("edits need guidance: at 1 the model often ignores the instruction and returns a copy of the input"
               if edit else "3-5 gives cleaner, more coherent images, better text, and uses the negative prompt; "
               "1 is twice as fast but hazier")
        return f"- cfg_scale: leave unset ({fmt(value)}, the tested best value; {why})."

    def neg_line(built_in: str, value: float) -> str:
        if not built_in:
            return "- negative_prompt: only used when cfg_scale is above 1; then list specific things to avoid."
        when = ("whenever cfg_scale is above 1 (the default here)" if value > 1
                else "only when cfg_scale is above 1 (not the default here)")
        return (f"- negative_prompt: {when}, this built-in list is applied: \"{built_in}\". Pass negative_prompt "
                "only for specific unwanted things (e.g. \"text, watermark\"); they are added to the list.")

    rec = g.recommended_size
    if not cpu and rec == "large" and max_mp >= 2.0:
        size_line = ("- size: pass size=\"large\" (about 2 MP) for final images unless the user asks for a quick draft,\n"
                     "  a preview or speed. \"xl\" (native 2K, about 4 MP) is sharper still but takes several minutes on\n"
                     "  this server: use it only when the user asks for maximum quality or detail. Leaving size unset\n"
                     "  gives \"medium\" (about 1 MP), a draft size. size sets the pixel count and aspect_ratio the\n"
                     "  shape, so combine them (e.g. size=\"large\", aspect_ratio=\"2:3\" for a poster).")
        instr_size = ("- size: for new images pass size=\"large\" (about 2 MP) unless the user asks for a draft; use \"xl\"\n"
                      "  (native 2K, several minutes here) only when the user asks for maximum quality.")
        edit_size = "Leave size unset for normal edits; pass size=\"large\" or \"xl\" only for extra detail (slower)."
    elif not cpu and rec == "medium":
        size_line = ("- size: leave size unset (\"medium\", about 1 MP) for normal images; this server's GPU is slower,\n"
                     "  so larger sizes take several minutes. Pass size=\"large\" (about 2 MP) or \"xl\" (native 2K, the\n"
                     "  sharpest) only when the user asks for more detail or maximum quality. size sets the pixel\n"
                     "  count and aspect_ratio the shape, so combine them.")
        instr_size = ("- size: leave it unset (about 1 MP) for normal images; pass \"large\" or \"xl\" (native 2K) only\n"
                      "  when the user asks for more detail, since they take several minutes on this server.")
        edit_size = "Leave size unset."
    elif max_mp >= 4.0:
        size_line = ("- size: pass size=\"xl\" (native 2K, about 4 MP) on every call unless the user asks for a quick\n"
                     "  draft, a preview or speed. Leaving size unset gives \"medium\" (about 1 MP): about 4x faster but\n"
                     "  visibly softer. size sets the pixel count and aspect_ratio the shape, so combine them\n"
                     "  (e.g. size=\"xl\", aspect_ratio=\"2:3\" for a poster).")
        instr_size = ("- size: for new images pass size=\"xl\" (about 4 MP, the native 2K) unless the user asks for a\n"
                      "  draft or speed; leaving it unset gives \"medium\" (about 1 MP), a draft size.")
        edit_size = "For the most detail pass size=\"xl\" (slower); otherwise leave size unset."
    else:
        size_line = (f"- size: this server allows at most about {max_mp:.1f} MP. Leave size unset for its default "
                     f"({cfg.default_size[0]}x{cfg.default_size[1]}); \"medium\" is the largest useful tier and takes "
                     "several times longer. aspect_ratio sets the shape.")
        instr_size = f"- size: this server allows at most about {max_mp:.1f} MP; leave size unset unless asked."
        edit_size = "Leave size unset."
    if cpu:
        timing = "Generation takes minutes on this CPU server (many minutes for large images)."
    elif rec == "xl":
        timing = ("Generation takes about 30 seconds to a few minutes on a GPU (size \"xl\", edits with several\n"
                  "inputs and panoramas take longest).")
    else:
        timing = ("Generation takes one to several minutes on this server's GPU (larger sizes, edits with several\n"
                  "inputs and panoramas take longest).")
    variation = ("- Another version of an earlier image (\"one more like it\", \"same idea again but without the signs\"):\n"
                 "  call generate_image again with the prompt from your earlier call, changed only as the user asks,\n"
                 "  and omit seed (never pass the earlier seed). Put newly unwanted things in negative_prompt. Do not\n"
                 "  use edit_image for this; edit_image changes an existing picture.")

    generate = f"""Generate a new image from a text prompt, in any size or aspect ratio, optionally with a transparent background.

How to get the best image:
- prompt: write a detailed, explicit description, not a few keywords. Cover the subject and what it is doing,
  the setting and background, composition and framing, lighting, colors and mood, and the medium (photo with
  camera and lens, oil painting, 3D render, flat illustration, ...). Put any text that should appear in the image
  in double quotes. A good opening is "A wide photograph of ..." or "A vertical poster of ...". Do not write
  resolutions, aspect ratios or words like 4K/8K/HD in the prompt; use size and aspect_ratio instead.
{size_line}
{steps_line}
{cfg_line(t2i_cfg, edit=False)}
{neg_line(g.negative_prompt, t2i_cfg)}
- transparent=true: describe only the subject (e.g. "a red fox sitting, full body, soft studio light"), with no
  background, backdrop, frame, tile, badge or "app icon" words. The server adds the transparency wording and
  cleans up the alpha channel.
- tileable=true: a seamless texture whose opposite edges continue into each other (patterns, wallpapers, game
  textures). Describe a surface or pattern that fills the whole frame, e.g. "moss-covered cobblestones, top-down".
  Edits of the result (e.g. a height, normal or roughness map from edit_image) and upscales stay seamless and keep
  its size automatically: pass the tile's file as the first image.
- seed: omit it. Pass the seed of an earlier result only to reproduce that exact image with the same prompt.
{variation}
{JOB_LINE}"""

    wm_hint = (" To remove watermarks, use remove_watermark instead: it is trained for that and changes the rest of "
               "the image less.") if cfg.watermark.enabled else ""
    edit = f"""Edit one image or combine several: add, remove or replace objects or text, change style, background, lighting or pose, or merge elements from up to 10 images into one.{wm_hint}

How to get the best result:
- prompt: start with the operation ("Replace the ...", "Remove the ...", "Add a ... to ...", "Change the style
  to ..."), then say what must stay the same in one short clause, e.g. "keep everything else unchanged".
  Describing the parts to keep in detail makes them drift more. Quote any text exactly. Refer to the inputs as
  <image1>, <image2>, ... and say what each one is for, e.g. "Put the dog from <image2> on the sofa in <image1>".
  <image1> is the picture being edited.
- size: leave aspect_ratio, width and height unset so the result keeps the shape of <image1>; a different shape
  shifts or zooms the content. Change it only to extend the scene (outpainting). {edit_size}
{steps_line}
{cfg_line(g.edit_cfg_scale, edit=True)}
{neg_line(g.edit_negative_prompt, g.edit_cfg_scale)}
- seed: leave unset. Never reuse the seed that produced the input image: re-editing with the same seed gives
  over-saturated, fragmented results.
- mask: for a local change pass a mask image (white = area to change, black = keep). It guides the model (it is
  sent as an extra reference image with an instruction); it is not a hard pixel mask, so areas outside it can
  still shift slightly.
- transparent=true extracts a subject onto a transparent background, e.g. prompt "Extract the logo".
- images: data URLs, base64, http(s) URLs, or images from this server (earlier results or uploads): pass the full
  link, or the file path exactly as the result or list_images gave it, including its folder
  (e.g. "2026-09-23/image-021530-ab12cd34.png"). Large photos are scaled down automatically. If the user says they
  uploaded an image, call list_images and use the newest uploads/ entry.
{JOB_LINE}"""

    pano = f"""Generate a 360-degree equirectangular panorama (2:1) with Photo Sphere metadata, plus a link to an interactive viewer.

How to get the best result:
- prompt: describe the whole environment around the viewer in every direction: the ground, the horizon, the sky
  or ceiling, and what is in front, to the left and right, and behind. Describe only the scene; do not write
  "360", "panorama" or "equirectangular" (the server adds that). Lighting, time of day, weather and style help.
- image: optional photo to extend into a full 360 panorama.
- Leave steps, cfg_scale and width unset (defaults here: {steps} steps, {cfg.panorama_size[0]}x{cfg.panorama_size[1]}).
  The server repairs the left/right wrap seam automatically.
{neg_line(g.negative_prompt, t2i_cfg)}
Panoramas take a minute or more. {JOB_LINE}"""

    if t2i_cfg > 1:
        instr_cfg = (f"- cfg_scale: leave unset ({fmt(t2i_cfg)} for new images, {fmt(g.edit_cfg_scale)} for edits, "
                     "the tested best values).")
    else:
        instr_cfg = (f"- cfg_scale: new images default to {fmt(t2i_cfg)} here, a speed setting (3-5 is cleaner at "
                     f"twice the time per step); edits default to {fmt(g.edit_cfg_scale)}.")
    instr_neg = ("- negative_prompt: a built-in list is applied whenever cfg_scale is above 1; pass only extra, "
                 "specific things to avoid.")
    wm = cfg.watermark
    if wm.restore_unchanged:
        wm_effect = ("the rest of the image stays as it was: only the areas the model changed are replaced, and the "
                     "result keeps the input's size")
    else:
        wm_effect = (f"the whole image is redrawn at about {fmt(wm.megapixels)} MP (very close to the input) and "
                     "returned at the input's size")
    watermark = (f"Remove watermarks from an image: logos, stamps, signatures, copyright lines and semi-transparent or "
                 f"tiled text laid over the picture. It uses a model trained for this, so {wm_effect}. Call it with "
                 f"just the image (no prompt needed); it takes about as long as an edit_image call. {JOB_LINE}")
    wm_line = (f"- remove_watermark: remove watermarks, logos and text stamped over a photo or picture;\n"
               f"  {wm_effect}.\n") if wm.enabled else ""
    up_line = ("- upscale_image: enlarge an image 2x or 4x with an AI super-resolution model (sharper detail than plain\n"
               "  resizing, same content), up to 8192 px per side.\n") if cfg.upscale.enabled else ""
    instructions = (INSTRUCTIONS.replace("__WATERMARK_LINE__", wm_line).replace("__UPSCALE_LINE__", up_line)
                    .replace("__TIMING__", timing).replace("__SIZE_LINE__", instr_size)
                    .replace("__STEPS_LINE__", steps_line).replace("__CFG_LINE__", instr_cfg)
                    .replace("__NEG_LINE__", instr_neg))
    return {"instructions": instructions, "generate": generate, "edit": edit, "panorama": pano,
            "watermark": watermark}


# Optional request headers (e.g. injected by an MCP bridge) that adapt results to a client.
INLINE_HEADER = "x-imagegen-inline"  # "data-uri-text": images as data-URI text lines (llama.cpp server-side MCP)
MAX_WAIT_HEADER = "x-imagegen-max-wait"  # upper bound for wait_seconds, in seconds
INLINE_MAX_HEADER = "x-inline-max-bytes"  # raise the inline image budget for clients without message limits


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


def create_server(service: ImageService) -> MCPServer:
    cfg: Config = service.cfg
    guide = _guide(cfg)
    mcp = MCPServer(name="imagegen-mcp", title="Image Gen MCP", version=__version__,
                    instructions=guide["instructions"], log_level=cfg.server.log_level)

    # ------------------------------------------------------------------ results
    def _fit(s, budget: int, chat_safe: bool) -> tuple[bytes, str]:
        """A preview of ``s`` that fits ``budget`` bytes. chat_safe: JPEG/PNG only (llama.cpp cannot decode WebP)."""
        side = cfg.outputs.preview_max_side
        make = imaging.chat_preview if chat_safe else (
            lambda im, sd: imaging.preview(im, sd, keep_alpha=im.mode == "RGBA"))
        data, mime = make(s.image, side)
        while len(data) > budget and side > 256:
            side = int(side * 0.75)
            data, mime = make(s.image, side)
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

    def _pending(job: Job) -> CallToolResult:
        st = job.summary()
        text = (f"Still working: job {job.id} is at {st['progress_percent']}% ({job.message}). It keeps running in "
                f"the background. Call the get_job tool of this image server (some clients list it with a prefix, "
                f'e.g. imagegen_get_job) with job_id="{job.id}" to wait for and fetch the image.')
        return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=st)

    async def _collect(job: Job, ctx: Context | None, wait_seconds: int | None) -> CallToolResult:
        wait = cfg.generation.wait_seconds if wait_seconds is None else wait_seconds
        cap = _header(ctx, MAX_WAIT_HEADER)
        if cap.isdigit():  # the client knows its own request deadline (e.g. llama.cpp timeout_ms)
            wait = min(wait, int(cap))
        done = await service.jobs.wait(job, wait, _progress(ctx) if ctx is not None else None)
        if not done:
            return _pending(job)
        if job.status == "completed":
            inline_max = _header(ctx, INLINE_MAX_HEADER)
            r = _result(job.result, data_uri_text=_header(ctx, INLINE_HEADER).lower() == "data-uri-text",
                        inline_max=max(10_000, min(int(inline_max), 64 * 1024 * 1024)) if inline_max.isdigit()
                        else None)
            r.structured_content["job_id"] = job.id
            return r
        return _error(job.error or f"job {job.status}")

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

    def _submit(kind: str, factory, ctx: Context | None, detail: str = "") -> Job:
        job = service.jobs.submit(kind, factory)
        log.info("job %s %s%s from %s", job.id, kind, f" [{detail}]" if detail else "", _client(ctx))
        return job

    async def _start(kind: str, factory, ctx: Context, wait_seconds: int | None, detail: str = "") -> CallToolResult:
        try:
            service.require_ready()
        except ServiceUnavailable as exc:
            return _error(str(exc))
        _remember_base_url(ctx)
        return await _collect(_submit(kind, factory, ctx, detail), ctx, wait_seconds)

    def _remember_base_url(ctx: Context | None) -> None:
        try:
            REQUEST_BASE_URL.set(base_url_from_headers(ctx.headers if ctx is not None else None))
        except Exception:  # noqa: BLE001 - fall back to the configured/default base URL
            pass

    # ------------------------------------------------------------------ tools
    @mcp.tool(description=guide["generate"],
              annotations=ToolAnnotations(title="Generate image", readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=False, openWorldHint=False))
    async def generate_image(
        prompt: Annotated[str, Field(description="Detailed description of the image: subject, setting, composition, lighting, colors, and medium or camera; text to render in double quotes. No resolutions or '4K' words.")],
        ctx: Context,
        aspect_ratio: Annotated[AspectRatio | None, Field(description="Aspect ratio, e.g. 16:9. Ignored when both width and height are given.")] = None,
        size: Size = None,
        width: Width = None,
        height: Height = None,
        transparent: Annotated[bool, Field(description="Transparent background: returns a PNG/WebP with an alpha channel. Describe only the subject in the prompt, with no background, tile or badge words.")] = False,
        tileable: Annotated[bool, Field(description="Seamless tiling texture: the left and right edges and the top and bottom edges continue into each other, so the image repeats without visible seams (patterns, wallpapers, game textures). Generated in one pass with wrap-around edges, so it takes no longer than a normal image. Cannot be combined with transparent.")] = False,
        negative_prompt: Negative = "",
        steps: Steps = None,
        cfg_scale: Guidance = None,
        seed: Seed = None,
        output_format: Annotated[Fmt | None, Field(description="File format of the saved image (png default).")] = None,
        wait_seconds: WaitSeconds = None,
    ) -> CallToolResult:
        return await _start("generate", lambda progress: service.generate(
            prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, aspect_ratio=aspect_ratio,
            size=size, transparent=transparent, steps=steps, cfg_scale=cfg_scale, seed=seed,
            output_format=output_format, tileable=tileable, progress=progress), ctx, wait_seconds,
            f"size={size} {width}x{height} ar={aspect_ratio} steps={steps} transparent={transparent} tileable={tileable}")

    @mcp.tool(description=guide["edit"],
              annotations=ToolAnnotations(title="Edit or combine images", readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=False, openWorldHint=True))
    async def edit_image(
        prompt: Annotated[str, Field(description="The edit instruction: start with the operation, then 'keep everything else unchanged'. Refer to inputs as <image1>, <image2>, ... e.g. 'Put the cat from <image2> on the sofa in <image1>, keep everything else unchanged', 'Remove the people in the background', 'Change the style to a watercolor painting'.")],
        images: Annotated[list[str], Field(min_length=1, max_length=10, description="1-10 input images: data URLs, base64, http(s) URLs, or this server's links or file paths exactly as given (with the date folder). The first image is the one being edited and sets the output shape.")],
        ctx: Context,
        mask: Annotated[str | None, Field(description="Optional mask image: white = area to change, black = keep (transparent pixels count as black). It is sent to the model as an extra reference image and the prompt is extended to change only the white area, so it guides the edit but does not lock the black area pixel for pixel. Counts as one of the 10 inputs.")] = None,
        aspect_ratio: Annotated[AspectRatio | None, Field(description="Output aspect ratio (default: same as the first image).")] = None,
        size: Size = None,
        width: EditWidth = None,
        height: EditHeight = None,
        transparent: Annotated[bool, Field(description="Produce a transparent background (e.g. 'extract the logo').")] = False,
        negative_prompt: Negative = "",
        steps: Steps = None,
        cfg_scale: Guidance = None,
        seed: EditSeed = None,
        output_format: Annotated[Fmt | None, Field(description="File format of the saved image (png default).")] = None,
        tileable: Annotated[bool | None, Field(description="Leave unset: when the first image is a seamless tile made by this server (generate_image tileable=true, or an earlier tileable edit or upscale), the edit keeps it seamless and at the same size, e.g. for height, normal or roughness maps of a texture. true: treat the first image as a seamless tile (e.g. a texture from elsewhere). false: a normal edit.")] = None,
        wait_seconds: WaitSeconds = None,
    ) -> CallToolResult:
        return await _start("edit", lambda progress: service.edit(
            prompt=prompt, images=images, mask=mask, negative_prompt=negative_prompt, width=width, height=height,
            aspect_ratio=aspect_ratio, size=size, transparent=transparent, steps=steps, cfg_scale=cfg_scale,
            seed=seed, output_format=output_format, tileable=tileable, progress=progress), ctx, wait_seconds,
            f"{len(images)} image(s) mask={mask is not None} size={size} {width}x{height} ar={aspect_ratio} steps={steps}"
            + (f" tileable={tileable}" if tileable is not None else ""))

    @mcp.tool(description=guide["panorama"],
              annotations=ToolAnnotations(title="Generate 360 panorama", readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=False, openWorldHint=True))
    async def generate_panorama(
        prompt: Annotated[str, Field(description="The scene only, in every direction: the ground, the horizon, the sky or ceiling, and what is in front, to the left and right, and behind the viewer, e.g. 'a misty pine forest at sunrise, a wooden cabin with smoke from the chimney in front, a lake behind, mossy ground'. Do not write '360', 'panorama' or 'equirectangular'; the server adds that.")],
        ctx: Context,
        image: Annotated[str | None, Field(description="Optional photo to extend into a full 360 panorama.")] = None,
        width: Annotated[int | None, Field(ge=512, le=4096, description="Panorama width, rounded to a multiple of 64; the height is exactly width/2 (equirectangular). Leave unset for the tested default; sizes above the server's limit are scaled down.")] = None,
        seam_fix: Annotated[bool | None, Field(description="Repair the left/right wrap seam with a second masked pass (default on).")] = None,
        negative_prompt: Negative = "",
        steps: Steps = None,
        cfg_scale: Guidance = None,
        seed: Seed = None,
        output_format: Annotated[Fmt | None, Field(description="jpeg (default, best viewer support), png or webp.")] = None,
        wait_seconds: WaitSeconds = None,
    ) -> CallToolResult:
        return await _start("panorama", lambda progress: service.panorama(
            prompt=prompt, image=image, negative_prompt=negative_prompt, width=width, steps=steps,
            cfg_scale=cfg_scale, seed=seed, seam_fix=seam_fix, output_format=output_format, progress=progress),
            ctx, wait_seconds, f"width={width} from_image={image is not None} seam_fix={seam_fix} steps={steps}")

    @mcp.tool(annotations=ToolAnnotations(title="Remove background", readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=True, openWorldHint=True))
    async def remove_background(
        image: Annotated[str, Field(description="Image to cut out: data URL, base64, http(s) URL, or this server's link or file path exactly as given (with the date folder).")],
        ctx: Context,
        output_format: Annotated[Literal["png", "webp"], Field(description="png (default) or webp; both keep the alpha channel.")] = "png",
        wait_seconds: WaitSeconds = None,
    ) -> CallToolResult:
        """Remove the background of an image and return the subject with a transparent background (PNG by default)."""
        if service.matter is None:
            return _error("background removal is not available yet (models are still loading) or it is disabled "
                          "(transparency.method is 'native')")
        _remember_base_url(ctx)
        job = _submit("remove_background", lambda progress: service.remove_background(
            image=image, output_format=output_format, progress=progress), ctx)
        return await _collect(job, ctx, wait_seconds)

    if cfg.watermark.enabled:
        @mcp.tool(description=guide["watermark"],
                  annotations=ToolAnnotations(title="Remove watermarks", readOnlyHint=False, destructiveHint=False,
                                              idempotentHint=False, openWorldHint=True))
        async def remove_watermark(
            image: Annotated[str, Field(description="Image to clean: data URL, base64, http(s) URL, or this server's link or file path exactly as given (with the date folder).")],
            ctx: Context,
            seed: EditSeed = None,
            output_format: Annotated[Fmt | None, Field(description="File format of the saved image (png default).")] = None,
            wait_seconds: WaitSeconds = None,
        ) -> CallToolResult:
            return await _start("remove_watermark", lambda progress: service.remove_watermark(
                image=image, seed=seed, output_format=output_format, progress=progress), ctx, wait_seconds)

    if cfg.upscale.enabled:
        @mcp.tool(description=UPSCALE_DESCRIPTION,
                  annotations=ToolAnnotations(title="Upscale image", readOnlyHint=False, destructiveHint=False,
                                              idempotentHint=True, openWorldHint=True))
        async def upscale_image(
            image: Annotated[str, Field(description="Image to enlarge: data URL, base64, http(s) URL, or this server's link or file path exactly as given (with the date folder).")],
            ctx: Context,
            scale: Annotated[Literal[2, 4], Field(description="Enlargement factor: 4 (default) or 2.")] = 4,
            output_format: Annotated[Fmt | None, Field(description="File format of the saved image (png default, jpeg for 360 panoramas; jpeg is much smaller for very large results).")] = None,
            panorama: Annotated[bool | None, Field(description="Leave unset: images tagged as 360 panoramas (Photo Sphere metadata, as generate_panorama makes them) are detected. true: treat a 2:1 image without that tag as a 360 panorama. false: plain upscale.")] = None,
            tileable: Annotated[bool | None, Field(description="Leave unset: seamless tiles made by this server are detected and stay seamless (both directions wrap). true: treat the image as a seamless tile (e.g. a texture from elsewhere). false: plain upscale.")] = None,
            wait_seconds: WaitSeconds = None,
        ) -> CallToolResult:
            return await _start("upscale", lambda progress: service.upscale(
                image=image, scale=scale, output_format=output_format, as_panorama=panorama, as_tileable=tileable,
                progress=progress), ctx, wait_seconds, f"scale={scale}"
                + (f" panorama={panorama}" if panorama is not None else "")
                + (f" tileable={tileable}" if tileable is not None else ""))

    @mcp.tool(annotations=ToolAnnotations(title="Get job result", readOnlyHint=True, openWorldHint=False))
    async def get_job(
        job_id: Annotated[str, Field(description="The job_id returned by a tool call that did not finish in time.")],
        ctx: Context,
        wait_seconds: JobWait = None,
    ) -> CallToolResult:
        """Wait for a running job and return its image(s), or its progress if it is still running."""
        job = service.jobs.get(job_id)
        if job is None:
            return _error(f"unknown job_id {job_id!r} (finished jobs are kept for 24 hours)")
        return await _collect(job, ctx, wait_seconds)

    @mcp.tool(annotations=ToolAnnotations(title="Cancel job", readOnlyHint=False, destructiveHint=True,
                                          idempotentHint=True, openWorldHint=False))
    async def cancel_job(job_id: Annotated[str, Field(description="The job to cancel.")]) -> dict:
        """Cancel a job. If the engine already started it, it finishes in the background but the result is discarded."""
        job = service.jobs.cancel(job_id)
        if job is None:
            return {"error": f"unknown job_id {job_id!r}"}
        await asyncio.sleep(0)
        return job.summary()

    @mcp.tool(annotations=ToolAnnotations(title="List images", readOnlyHint=True, openWorldHint=False))
    async def list_images(
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=200, description="Maximum number of entries in the combined results-and-uploads list (and in the inputs folder list).")] = 30,
    ) -> dict:
        """List recent generated and uploaded images (and files in the inputs folder) that can be used as inputs by their 'file' path or 'url'. Users upload images on this server's /upload page; uploads appear under uploads/. Results and uploads share one list of `limit` entries, newest first."""
        _remember_base_url(ctx)
        return service.list_images(limit)

    @mcp.tool(annotations=ToolAnnotations(title="View image", readOnlyHint=True, openWorldHint=False))
    async def view_image(
        image: Annotated[str, Field(description="An image on this server: the file path or link from a result or list_images (e.g. '2026-09-23/image-021530-ab12cd34.png'), or its bare file name.")],
        ctx: Context,
    ) -> CallToolResult:
        """Look at an image stored on this server at its original size and resolution, e.g. to check a result or an upload before editing it. Returns the original file (PNG or JPEG as saved; other formats as PNG) plus its size and link. Use list_images to find file names. Images attached in the chat are not on this server: use the chat's own handle for those."""
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

    @mcp.tool(annotations=ToolAnnotations(title="Server status", readOnlyHint=True, openWorldHint=False))
    async def server_status() -> dict:
        """Report model, device placement, download/loading progress, running jobs and default settings."""
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
