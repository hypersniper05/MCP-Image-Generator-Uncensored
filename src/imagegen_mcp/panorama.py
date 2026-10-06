"""360-degree equirectangular panorama helpers.

Qwen-Image-2.1 can produce 2:1 equirectangular images from the prompt alone,
but the left and right edges are not guaranteed to meet. The seam is repaired
by rolling the image half a turn (moving the seam to the centre), repainting a
narrow vertical band with a masked image-to-image pass, and rolling back.
The result carries Google Photo Sphere (GPano) XMP metadata so viewers such as
Google Photos, Facebook, Kuula or Pannellum open it as a 360 photo.
"""

from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image, ImageFilter

# Wording follows the official Qwen-Image-2.1 examples ("360-degree panorama" edit case).
PROJECTION = ("Use a true equirectangular projection covering the entire 360-degree horizontal and 180-degree "
              "vertical field of view, including the sky overhead and the ground below, with a level horizon across "
              "the vertical center. Keep the left and right edges seamless so they join into a single continuous "
              "scene. Output resolution: {w} x {h} pixels, with an exact 2:1 aspect ratio.")


def panorama_prompt(prompt: str, width: int, height: int) -> str:
    return (f"Generate a complete 360-degree equirectangular panorama. {prompt.strip()}\n\n"
            + PROJECTION.format(w=width, h=height))


def panorama_from_image_prompt(prompt: str, width: int, height: int) -> str:
    return (f"Generate a complete 360-degree equirectangular panorama from the input perspective image <image1>. "
            f"{prompt.strip()}\n\n" + PROJECTION.format(w=width, h=height))


def seam_score(img: Image.Image) -> float:
    """~1.0 means the wrap-around is as smooth as the rest of the image; >2 is a visible seam."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    wrap = np.abs(a[:, 0] - a[:, -1]).mean()
    inner = np.abs(np.diff(a, axis=1)).mean()
    return round(float(wrap / max(inner, 1e-6)), 2)


def roll_half(img: Image.Image, back: bool = False) -> Image.Image:
    a = np.asarray(img)
    w = a.shape[1]
    shift = w // 2
    return Image.fromarray(np.roll(a, -shift if back else shift, axis=1), img.mode)


def seam_mask(size: tuple[int, int], band_fraction: float) -> tuple[Image.Image, Image.Image]:
    """Return (hard mask for the model, feathered mask for compositing). White = repaint."""
    w, h = size
    band = max(64, int(w * band_fraction) // 32 * 32)
    x = np.arange(w, dtype=np.float32)
    d = np.abs(x - (w - 1) / 2.0)
    hard = (d <= band / 2).astype(np.float32)
    feather = max(8.0, band / 4)
    soft = np.clip((band / 2 - d) / feather, 0.0, 1.0)
    hard_img = Image.fromarray((np.tile(hard, (h, 1)) * 255).astype(np.uint8), "L")
    soft_img = Image.fromarray((np.tile(soft, (h, 1)) * 255).astype(np.uint8), "L")
    return hard_img, soft_img


def roll_both(img: Image.Image, back: bool = False) -> Image.Image:
    """Shift by half the width and half the height, so the image's wrap-around edges meet in the middle."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    sign = -1 if back else 1
    return Image.fromarray(np.roll(a, (sign * (h // 2), sign * (w // 2)), axis=(0, 1)), img.mode)


def cross_mask(size: tuple[int, int], band_fraction: float) -> tuple[Image.Image, Image.Image]:
    """Masks for a vertical and a horizontal band through the center (hard for the model, feathered for
    compositing). After roll_both, those bands hold the seams where a tiled image repeats."""
    w, h = size
    hard_v, soft_v = seam_mask((w, h), band_fraction)
    hard_h, soft_h = seam_mask((h, w), band_fraction)
    hard = np.maximum(np.asarray(hard_v), np.asarray(hard_h).T)
    soft = np.maximum(np.asarray(soft_v), np.asarray(soft_h).T)
    return Image.fromarray(hard, "L"), Image.fromarray(soft, "L")


def tile_seam_score(img: Image.Image) -> float:
    """Wrap-around seam strength in both directions, relative to ordinary neighbour differences (1 = seamless)."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    inner = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2 + 1e-6
    wrap = (np.abs(a[:, 0] - a[:, -1]).mean() + np.abs(a[0, :] - a[-1, :]).mean()) / 2
    return round(float(wrap / inner), 2)


def wrap_report(img: Image.Image) -> dict:
    """Per-axis wrap-around check for a tile. ratio: the step across the wrap edge relative to the average step
    between neighbouring lines (about 1 = no seam). pct: share of the image's own line-to-line steps that are
    smaller than the wrap step (a seam sits near 100). tile_seam_score averages both axes and can hide a seam
    that is only on one axis."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    out = {}
    for name, axis in (("x", 1), ("y", 0)):
        steps = np.abs(np.diff(a, axis=axis)).mean(axis=1 - axis)
        wrap = float(np.abs(np.take(a, 0, axis) - np.take(a, -1, axis)).mean())
        out[name] = {"ratio": round(wrap / (float(steps.mean()) + 1e-6), 2),
                     "pct": round(float((steps < wrap).mean() * 100), 1)}
    return out


def blend(base: Image.Image, patch: Image.Image, soft_mask: Image.Image) -> Image.Image:
    patch = patch.convert(base.mode).resize(base.size, Image.Resampling.LANCZOS)
    return Image.composite(patch, base, soft_mask)


def crossfade_wrap(img: Image.Image, band: int = 32) -> Image.Image:
    """Cheap no-model fallback that forces column 0 and column W-1 to agree."""
    a = np.asarray(img).astype(np.float32)
    band = min(band, a.shape[1] // 8)
    target = (a[:, :1] + a[:, -1:]) / 2.0
    ramp = np.linspace(1.0, 0.0, band, dtype=np.float32)[None, :, None]
    a[:, :band] = a[:, :band] * (1 - ramp) + target * ramp
    a[:, -band:] = a[:, -band:] * (1 - ramp[:, ::-1]) + target * ramp[:, ::-1]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), img.mode)


def soften_poles(img: Image.Image, lat_start_deg: float = 78.0) -> Image.Image:
    """Blur the top and bottom rows horizontally; they collapse to a point on the sphere."""
    w, h = img.size
    rows = int(h * (90 - lat_start_deg) / 180)
    if rows < 2:
        return img
    out = img.copy()
    for top in (True, False):
        box = (0, 0, w, rows) if top else (0, h - rows, w, h)
        strip = out.crop(box)
        blurred = strip.filter(ImageFilter.BoxBlur(radius=max(2, w // 256)))
        grad = np.linspace(1.0, 0.0, rows, dtype=np.float32) if top else np.linspace(0.0, 1.0, rows, dtype=np.float32)
        m = Image.fromarray((np.tile(grad[:, None], (1, w)) * 255).astype(np.uint8), "L")
        out.paste(Image.composite(blurred, strip, m), box[:2])
    return out


def gpano_xmp(width: int, height: int, *, heading: float = 180.0, hfov: float = 90.0,
              software: str = "imagegen-mcp") -> bytes:
    xmp = f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">
   <GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>
   <GPano:ProjectionType>equirectangular</GPano:ProjectionType>
   <GPano:StitchingSoftware>{software}</GPano:StitchingSoftware>
   <GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>
   <GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>
   <GPano:CroppedAreaImageWidthPixels>{width}</GPano:CroppedAreaImageWidthPixels>
   <GPano:CroppedAreaImageHeightPixels>{height}</GPano:CroppedAreaImageHeightPixels>
   <GPano:FullPanoWidthPixels>{width}</GPano:FullPanoWidthPixels>
   <GPano:FullPanoHeightPixels>{height}</GPano:FullPanoHeightPixels>
   <GPano:InitialViewHeadingDegrees>{int(heading)}</GPano:InitialViewHeadingDegrees>
   <GPano:InitialViewPitchDegrees>0</GPano:InitialViewPitchDegrees>
   <GPano:InitialViewRollDegrees>0</GPano:InitialViewRollDegrees>
   <GPano:InitialHorizontalFOVDegrees>{float(hfov):.1f}</GPano:InitialHorizontalFOVDegrees>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    return xmp.encode("utf-8")


GPANO_NS = "http://ns.google.com/photos/1.0/panorama/"
_PIXEL_FIELDS = ("CroppedAreaLeftPixels", "CroppedAreaTopPixels", "CroppedAreaImageWidthPixels",
                 "CroppedAreaImageHeightPixels", "FullPanoWidthPixels", "FullPanoHeightPixels")


def _xmp_text(img: Image.Image) -> str | None:
    """The image's XMP packet (JPEG and WebP keep it as "xmp", PNG as an iTXt chunk)."""
    for key in ("xmp", "XML:com.adobe.xmp"):
        value = img.info.get(key)
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _gpano_prefix(xmp: str) -> str | None:
    """The prefix the packet binds to the Photo Sphere namespace (usually GPano), or None."""
    m = re.search(r"xmlns:([\w.-]+)\s*=\s*([\"'])" + re.escape(GPANO_NS) + r"\2", xmp)
    if m:
        return m.group(1)
    return "GPano" if "GPano:" in xmp else None


def _field_pattern(prefix: str, name: str) -> re.Pattern:
    # Writers use elements (<GPano:Name>v</GPano:Name>) or attributes (GPano:Name="v" or 'v'). The lookbehind
    # skips closing tags, so a substitution never writes a value after </GPano:Name>.
    return re.compile(rf"((?<![/\w.-]){re.escape(prefix)}:{name}\s*(?:>\s*|=\s*([\"'])))([^<\"']*?)(\s*<|\2)")


def _gpano_field(xmp: str, prefix: str, name: str) -> str | None:
    m = _field_pattern(prefix, name).search(xmp)
    return m.group(3).strip() if m else None


def sphere_xmp(img: Image.Image) -> str | None:
    """The XMP packet of an image tagged as a full 360x180 equirectangular panorama, else None.

    A partial panorama (cropped area smaller than the full sphere) returns None: its edges do not wrap around.
    """
    xmp = _xmp_text(img)
    prefix = _gpano_prefix(xmp) if xmp else None
    if prefix is None or (_gpano_field(xmp, prefix, "ProjectionType") or "").lower() != "equirectangular":
        return None

    def number(name: str) -> float | None:
        try:
            value = float(_gpano_field(xmp, prefix, name) or "")
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    for crop, full in (("CroppedAreaImageWidthPixels", "FullPanoWidthPixels"),
                       ("CroppedAreaImageHeightPixels", "FullPanoHeightPixels")):
        c, f = number(crop), number(full)
        if c and f and c < f:
            return None
    return xmp


def resize_sphere_xmp(xmp: str, width: int, height: int) -> bytes:
    """The same packet for a full-sphere image resized to width x height: only the pixel fields change, so the
    initial view, pose and everything else in the packet are kept."""
    prefix = _gpano_prefix(xmp) or "GPano"
    values = {"CroppedAreaLeftPixels": 0, "CroppedAreaTopPixels": 0, "CroppedAreaImageWidthPixels": width,
              "CroppedAreaImageHeightPixels": height, "FullPanoWidthPixels": width, "FullPanoHeightPixels": height}
    for name in _PIXEL_FIELDS:
        xmp = _field_pattern(prefix, name).sub(lambda m, v=values[name]: f"{m.group(1)}{v}{m.group(4)}", xmp)
    return xmp.encode("utf-8")


def is_two_to_one(width: int, height: int) -> bool:
    return height > 0 and abs(width / height - 2.0) <= 0.02


def wrap_pad(img: Image.Image, pad: int, pad_y: int = 0) -> Image.Image:
    """Add `pad` columns (and `pad_y` rows) from the opposite edge on each side, so filters see across the
    wrap-around of a 360 panorama (columns) or a seamless tile (both)."""
    w, h = img.size
    px, py = max(0, min(pad, w)), max(0, min(pad_y, h))
    a = np.asarray(img)
    widths = ((py, py), (px, px)) + (((0, 0),) if a.ndim == 3 else ())
    return Image.fromarray(np.pad(a, widths, mode="wrap"), img.mode)


# Seamless tiles carry this small XMP packet, so later edits and upscales of the file keep it seamless.
TILE_NS = "urn:imagegen-mcp:tile:1.0:"


def tile_xmp() -> bytes:
    return (f'<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description rdf:about="" xmlns:imagegen="{TILE_NS}" imagegen:Tileable="True"/>'
            f'</rdf:RDF></x:xmpmeta>\n<?xpacket end="w"?>').encode("utf-8")


def is_tile(img: Image.Image) -> bool:
    """True for an image tagged as a seamless tile by this server (tile_xmp)."""
    xmp = _xmp_text(img)
    return bool(xmp and TILE_NS in xmp and re.search(r"Tileable\s*(?:=\s*[\"']|>\s*)True\b", xmp))


VIEWER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>360 viewer</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.7/build/pannellum.css">
<script src="https://cdn.jsdelivr.net/npm/pannellum@2.5.7/build/pannellum.js"></script>
<style>html,body,#pano{margin:0;width:100%;height:100%;background:#111}</style></head>
<body><div id="pano"></div><script>
pannellum.viewer('pano',{type:'equirectangular',panorama:__IMAGE_URL__,autoLoad:true,showZoomCtrl:true,
  hfov:100,compass:false,autoRotate:-2});
</script></body></html>"""
