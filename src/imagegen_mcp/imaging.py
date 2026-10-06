"""Image input decoding, size resolution and output encoding."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import ipaddress
import math
import re
import socket
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlparse

import httpx
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Qwen-Image-2.1 works on a 16x downsampled latent with 2x2 patches -> multiples of 32.
MULTIPLE = 32
MIN_SIDE = 256
MAX_SIDE = 4096

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3),
    "16:9": (16, 9), "9:16": (9, 16), "21:9": (21, 9), "9:21": (9, 21), "2:1": (2, 1), "1:2": (1, 2),
    "5:4": (5, 4), "4:5": (4, 5),
}

# Approximate megapixels per quality tier.
SIZE_TIERS: dict[str, float] = {"small": 0.26, "medium": 1.05, "large": 2.1, "xl": 4.19}


class ImageInputError(ValueError):
    pass


def snap(v: float) -> int:
    return max(MIN_SIDE, min(MAX_SIDE, int(round(v / MULTIPLE)) * MULTIPLE))


def dims_for(aspect: tuple[float, float], pixels: float) -> tuple[int, int]:
    a = aspect[0] / aspect[1]
    return snap(math.sqrt(pixels * a)), snap(math.sqrt(pixels / a))


def parse_aspect(value: str) -> tuple[float, float]:
    v = value.strip().lower().replace("/", ":").replace("x", ":")
    if v in ASPECT_RATIOS:
        return ASPECT_RATIOS[v]
    m = re.match(r"^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$", v)
    if not m or float(m.group(1)) <= 0 or float(m.group(2)) <= 0:
        raise ValueError(f"invalid aspect_ratio {value!r}; use e.g. 16:9, 3:2, 1:1")
    a = float(m.group(1)) / float(m.group(2))
    if not 0.2 <= a <= 5:
        raise ValueError("aspect_ratio must be between 1:5 and 5:1")
    return float(m.group(1)), float(m.group(2))


def resolve_size(*, width: int | None, height: int | None, aspect_ratio: str | None, size: str | None,
                 default: tuple[int, int], max_pixels: int,
                 reference: tuple[int, int] | None = None) -> tuple[int, int, list[str]]:
    """Work out the output size. Returns (width, height, notes)."""
    notes: list[str] = []
    default_px = default[0] * default[1]
    pixels = SIZE_TIERS[size] * 1_000_000 if size else float(default_px)

    if width and height:
        w, h = snap(width), snap(height)
    elif width or height:
        if aspect_ratio:
            ar = parse_aspect(aspect_ratio)
        elif reference:
            ar = reference
        else:
            ar = (1, 1)
        if width:
            w, h = snap(width), snap(width * ar[1] / ar[0])
        else:
            w, h = snap(height * ar[0] / ar[1]), snap(height)
    elif aspect_ratio:
        w, h = dims_for(parse_aspect(aspect_ratio), pixels)
    elif reference:
        w, h = dims_for(reference, pixels)
    elif size:
        w, h = dims_for(default, pixels)
    else:
        w, h = default

    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        nw, nh = int(w * scale) // MULTIPLE * MULTIPLE, int(h * scale) // MULTIPLE * MULTIPLE
        notes.append(f"reduced {w}x{h} to {nw}x{nh} (limit {max_pixels / 1e6:.1f} MP on this server)")
        w, h = max(MIN_SIDE, nw), max(MIN_SIDE, nh)
    if (width and snap(width) != width) or (height and snap(height) != height):
        notes.append("dimensions were rounded to multiples of 32")
    return w, h, notes


def _is_private_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified:
            return True
    return False


def shorten(value: str, keep: int = 60) -> str:
    """Quote user input in messages without echoing megabytes of base64 back to the model."""
    value = str(value)
    return repr(value) if len(value) <= keep else repr(value[:keep]) + f"...({len(value)} chars)"


def decode_base64_image(data: str) -> bytes:
    s = re.sub(r"\s+", "", data.strip())
    s = s.replace("-", "+").replace("_", "/")  # accept URL-safe base64 too
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageInputError(f"invalid base64 image data ({exc.__class__.__name__})") from exc


def decode_data_url(ref: str) -> bytes:
    header, sep, payload = ref.partition(",")
    if not sep:
        raise ImageInputError(f"malformed data URL {shorten(ref)}")
    if ";base64" in header.lower():
        return decode_base64_image(payload)
    return unquote_to_bytes(payload)


_B64_RE = re.compile(r"^[A-Za-z0-9+/=_\-\s]+$")
_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"RIFF", b"GIF8", b"BM", b"II*\x00", b"MM\x00*")


def _looks_like_base64_image(s: str) -> bool:
    if len(s) < 16 or not _B64_RE.match(s[:4096]):
        return False
    head = re.sub(r"\s+", "", s[:64])
    head = head[: len(head) // 4 * 4]
    try:
        raw = base64.b64decode(head.replace("-", "+").replace("_", "/"), validate=False)
    except (binascii.Error, ValueError):
        return False
    return raw.startswith(_MAGIC)


class ImageLoader:
    """Turns the strings clients send into PIL images.

    Accepted forms: data URL, raw base64, http(s) URL, a file name or URL of an
    image this server produced earlier (``/outputs/...``), or a path inside the
    optional read-only ``/inputs`` folder.
    """

    def __init__(self, outputs_dir: Path, inputs_dir: Path = Path("/inputs"), max_bytes: int = 64 * 1024 * 1024,
                 original_for=None):
        self.outputs_dir = outputs_dir.resolve()
        self.inputs_dir = inputs_dir
        self.max_bytes = max_bytes
        # callable(bytes) -> Path | None: the full-resolution file behind a preview this server sent out
        self.original_for = original_for

    @staticmethod
    def _inside(root: Path, rel: str) -> Path | None:
        """``root/rel`` if it is an existing file inside ``root`` (no path traversal), else None."""
        try:
            base = root.resolve()
            candidate = (base / rel.lstrip("/")).resolve()
            if candidate.is_relative_to(base) and candidate.is_file():
                return candidate
        except (OSError, ValueError):
            pass
        return None

    def _local_file(self, name: str) -> Path | None:
        """A short file name, or an /outputs/... or /view/... path, of an image this server has (outputs, uploads,
        inputs)."""
        if len(name) > 1024 or name.count("/") > 32:  # never resolve absurd paths (slow, never ours)
            return None
        for route in ("/outputs/", "/view/"):  # a 360 viewer link points at the same file
            if name.startswith(route):
                name = name[len(route):]
                break
        found = self._inside(self.outputs_dir, name) or self._inside(self.inputs_dir, name)
        if found is None and "/" not in name and "\\" not in name and re.fullmatch(r"[\w.-]+\.\w{2,5}", name):
            # A bare result name without its date folder (clients often drop it): newest match wins.
            matches = sorted(self.outputs_dir.glob(f"*/{name}"), key=lambda p: p.stat().st_mtime, reverse=True)
            found = next((p for p in matches if self._inside(self.outputs_dir, str(p.relative_to(self.outputs_dir)))), None)
        return found

    def local_path(self, ref: str) -> Path | None:
        """The file behind a reference to an image this server has: a link to /outputs/..., a path or a bare name."""
        ref = (ref or "").strip()
        if ref.startswith(("http://", "https://")):
            path = urlparse(ref).path
            return self._local_file(path) if path.startswith(("/outputs/", "/view/")) else None
        if len(ref) >= 512 or "," in ref or ":" in ref:
            return None
        return self._local_file(ref)

    async def load(self, ref: str) -> Image.Image:
        if not isinstance(ref, str) or not ref.strip():
            raise ImageInputError("empty image reference")
        ref = ref.strip()
        if ref.startswith(("http://", "https://")):
            path = urlparse(ref).path
            # URLs of this server's own files (or their 360 viewer pages) are read locally, whatever host name the
            # client used.
            if not (path.startswith(("/outputs/", "/view/")) and self._local_file(path)):
                data = await self._fetch(ref)
                return await asyncio.to_thread(self._open, data)
        # Decoding megabytes of base64 and images is CPU work: keep it off the event loop.
        return await asyncio.to_thread(self._load_sync, ref)

    def _load_sync(self, ref: str) -> Image.Image:
        data: bytes
        own = False  # this server's own files (results, uploads, ./inputs) are not limited by max_request_mb
        if ref.startswith("data:"):
            data = decode_data_url(ref)
        elif ref.startswith(("http://", "https://")):
            data, own = self._local_file(urlparse(ref).path).read_bytes(), True
        elif len(ref) < 512 and "," not in ref and ":" not in ref and (local := self._local_file(ref)):
            data, own = local.read_bytes(), True
        elif _looks_like_base64_image(ref):
            data = decode_base64_image(ref)
        else:
            hint = ""
            if len(ref) < 512 and re.search(r"\.(png|jpe?g|webp|gif|bmp|heic|avif)$", ref, re.IGNORECASE):
                hint = (" This server has no file by that name: if it is the name of an image attached in the chat, "
                        "pass the chat's image handle (e.g. img_1) or the image data instead, not the file name.")
            raise ImageInputError(
                f"could not interpret image reference {shorten(ref)}.{hint} Send a data URL, base64, an http(s) URL, "
                "or the URL/file name of an image produced or uploaded to this server (see list_images).")
        if self.original_for is not None and not ref.startswith(("http://", "https://")):
            original = self.original_for(data)
            if original is not None:
                img = self._open(original.read_bytes(), own=True)
                img.info["imagegen_note"] = (f"a reduced preview was sent as input; used its full-resolution "
                                             f"original {original.name} instead")
                return img
        return self._open(data, own=own)

    def _open(self, data: bytes, own: bool = False) -> Image.Image:
        if len(data) > self.max_bytes and not own:
            raise ImageInputError(f"image is larger than {self.max_bytes // (1024 * 1024)} MB")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as exc:  # noqa: BLE001
            raise ImageInputError(f"not a readable image: {exc}") from exc
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.getbands() or img.mode == "P" else "RGB")
        return img

    async def _fetch(self, url: str) -> bytes:
        # Redirects are followed by hand so every hop gets the same host check (a public URL must not
        # be able to redirect to loopback or link-local services).
        async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
            for _ in range(6):
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    raise ImageInputError(f"refusing to fetch a {parsed.scheme or 'relative'} URL")
                if await asyncio.to_thread(_is_private_host, parsed.hostname or ""):
                    raise ImageInputError("refusing to fetch images from loopback or link-local addresses")
                async with client.stream("GET", url, headers={"User-Agent": "imagegen-mcp/1.0"}) as r:
                    if r.is_redirect and r.headers.get("location"):
                        url = str(r.url.join(r.headers["location"]))
                        continue
                    if r.status_code != 200:
                        raise ImageInputError(f"fetching {shorten(url, 120)} failed with HTTP {r.status_code}")
                    chunks, total = [], 0
                    async for c in r.aiter_bytes():
                        total += len(c)
                        if total > self.max_bytes:
                            raise ImageInputError(f"image at {shorten(url, 120)} is larger than "
                                                  f"{self.max_bytes // (1024 * 1024)} MB")
                        chunks.append(c)
                    return b"".join(chunks)
        raise ImageInputError(f"too many redirects fetching {shorten(url, 120)}")


def to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode()


def fit_reference(img: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale a reference image to at most ``max_pixels`` (keeps aspect)."""
    px = img.width * img.height
    if px <= max_pixels:
        return img
    s = math.sqrt(max_pixels / px)
    return img.resize((max(32, int(img.width * s)), max(32, int(img.height * s))), Image.Resampling.LANCZOS)


def rgb_icc(img: Image.Image) -> bytes | None:
    """The image's ICC profile if it describes RGB data, else None (e.g. a CMYK or grey profile that no longer
    matches pixels the loader converted to RGB)."""
    icc = img.info.get("icc_profile")
    if not icc:
        return None
    try:
        from PIL import ImageCms

        space = ImageCms.ImageCmsProfile(io.BytesIO(icc)).profile.xcolor_space.strip()
    except Exception:  # noqa: BLE001 - unreadable profile: do not copy it
        return None
    return icc if space == "RGB" else None


def restore_unchanged(original: Image.Image, sent: Image.Image, result: Image.Image, *, threshold: int = 16,
                      grow: int = 7, feather: float = 4.0,
                      max_changed: float = 0.85) -> tuple[Image.Image | None, float]:
    """Put an edit result back onto the original so only the areas the model changed are replaced.

    ``sent`` is the input as the model saw it and ``result`` the model output, both at the working size;
    ``original`` may be larger or smaller. Differences above ``threshold`` (0-255, after a slight blur that
    ignores re-encoding noise) are grown by ``grow`` pixels and feathered. Returns (image at the original
    size, changed fraction), or (None, fraction) when more than ``max_changed`` of the image changed, where a
    patchwork would look worse than the plain result."""
    a = np.asarray(sent.convert("RGB"), dtype=np.int16)
    b = np.asarray(result.convert("RGB").resize(sent.size, Image.Resampling.LANCZOS), dtype=np.int16)
    diff = Image.fromarray(np.abs(a - b).max(axis=2).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    mask = Image.fromarray(((np.asarray(diff) > threshold) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MaxFilter(2 * grow + 1)).filter(ImageFilter.GaussianBlur(feather))
    changed = float((np.asarray(mask) > 127).mean())
    if changed > max_changed:
        return None, changed
    orig = original.convert("RGB")
    if mask.size != orig.size:
        mask = mask.resize(orig.size, Image.Resampling.BILINEAR)
    patch = result.convert("RGB")
    if patch.size != orig.size:
        patch = patch.resize(orig.size, Image.Resampling.LANCZOS)
    return Image.composite(patch, orig, mask), changed


def normalize_mask(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Turn any mask image into a clean white-on-black RGB image the size of the edit target.

    Transparent pixels of an RGBA mask count as black, so a painted layer with alpha works too.
    """
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    gray = img.convert("L").resize(size, Image.Resampling.BILINEAR)
    return gray.point(lambda v: 255 if v >= 128 else 0).convert("RGB")


def has_transparency(img: Image.Image) -> bool:
    if img.mode != "RGBA":
        return False
    lo, _ = img.getchannel("A").getextrema()
    return lo < 250


def flatten(img: Image.Image, color=(255, 255, 255)) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, color)
    bg.paste(img, mask=img.getchannel("A"))
    return bg


def degrid(img: Image.Image, bandwidth: float = 0.045, pad: int = 32,
           wrap: tuple[bool, bool] = (False, False)) -> Image.Image:
    """Remove the 2-pixel grid (a spike at the Nyquist frequency) that the Qwen-Image VAE decoder leaves.

    A narrow Gaussian notch at the three Nyquist points (x, y and diagonal) takes the grid out and leaves
    real detail, which is broadband, practically untouched. The alpha channel is kept as is.
    wrap = (x, y): axes on which the image repeats (a tile, a 360 panorama). Those axes get no padding, so the
    FFT filter is circular there and the left/right (top/bottom) edges stay continuous.
    """
    if min(img.size) < 2 * pad:
        return img
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = a.shape[:2]
    px, py = (0 if wrap[0] else pad), (0 if wrap[1] else pad)
    p = np.pad(a, ((py, py), (px, px), (0, 0)), mode="reflect")  # even pad keeps the grid's phase
    fy = np.abs(np.fft.fftfreq(p.shape[0]))[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]

    def bump(d):
        return np.exp(-(d * d) / (2 * bandwidth * bandwidth))

    ny, nx = bump(fy - 0.5), bump(fx - 0.5)
    notch = (1.0 - (nx * bump(fy) + bump(fx) * ny + nx * ny)).astype(np.float32)
    spectrum = np.fft.rfft2(p, axes=(0, 1)) * notch[:, :, None]
    out = np.fft.irfft2(spectrum, s=p.shape[:2], axes=(0, 1))[py:py + h, px:px + w]
    result = Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")
    if alpha is not None:
        result.putalpha(alpha)
    return result


def encode(img: Image.Image, fmt: str, *, quality: int = 95, xmp: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    fmt = fmt.lower()
    icc = img.info.get("icc_profile")  # PNG keeps it from img.info on its own; JPEG and WebP need it passed
    if fmt == "jpeg":
        kw = {"quality": quality, "optimize": True}
        if xmp:
            kw["xmp"] = xmp
        if icc:
            kw["icc_profile"] = icc
        flatten(img).save(buf, "JPEG", **kw)
    elif fmt == "webp":
        kw = {"quality": quality, "method": 4}
        if xmp:
            kw["xmp"] = xmp
        if icc:
            kw["icc_profile"] = icc
        img.save(buf, "WEBP", **kw)
    else:
        from PIL.PngImagePlugin import PngInfo

        info = None
        if xmp:
            info = PngInfo()
            info.add_itxt("XML:com.adobe.xmp", xmp.decode("utf-8"), zip=False)
        img.save(buf, "PNG", pnginfo=info, optimize=False, compress_level=6)
    return buf.getvalue()


MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
EXT = {"png": "png", "jpeg": "jpg", "webp": "webp"}


def chat_preview(img: Image.Image, max_side: int) -> tuple[bytes, str]:
    """Preview limited to JPEG/PNG (some clients, e.g. llama.cpp, cannot decode WebP)."""
    im = img.copy()
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if has_transparency(im):
        return encode(im, "png"), "image/png"
    return encode(im, "jpeg", quality=85), "image/jpeg"


def preview(img: Image.Image, max_side: int, keep_alpha: bool) -> tuple[bytes, str]:
    im = img.copy()
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if keep_alpha and im.mode == "RGBA":
        return encode(im, "webp", quality=85), "image/webp"
    return encode(im, "jpeg", quality=85), "image/jpeg"
