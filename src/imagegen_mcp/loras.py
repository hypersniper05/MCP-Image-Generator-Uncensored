"""LoRA helpers: match a LoRA's tensor names to the layout of the diffusion model file.

Qwen-Image-2.1 checkpoints store the image MLP input in one of two ways: fused (``img_mlp.gate_up``, the
official layout, which LoRA trainers such as ai-toolkit use) or split (``img_mlp.gate_layer`` +
``img_mlp.proj``, used by some GGUF conversions, including the ones this server downloads).
stable-diffusion.cpp applies a LoRA tensor only when its name matches the model, so a fused-layout LoRA on
a split model silently loses its MLP weights. ``prepare_lora`` writes a converted copy in that case: the
fused ``lora_B`` rows are cut in half (first half -> gate_layer, second half -> proj, the same order the
engine uses when it runs a fused model) and ``lora_A`` is shared by both halves.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path
from typing import BinaryIO

log = logging.getLogger("imagegen.loras")

FUSED, SPLIT = "fused", "split"
CONVERTED_DIR = "converted"

# GGUF value types -> struct format of a fixed-size value (8 = string, 9 = array are handled separately)
_GGUF_FIXED = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4,
                "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def _read(fh: BinaryIO, fmt: str):
    size = struct.calcsize("<" + fmt)
    data = fh.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack("<" + fmt, data)[0]


def _skip_string(fh: BinaryIO) -> None:
    fh.seek(_read(fh, "Q"), os.SEEK_CUR)


def _skip_value(fh: BinaryIO, vtype: int) -> None:
    if vtype in _GGUF_FIXED:
        fh.seek(struct.calcsize("<" + _GGUF_FIXED[vtype]), os.SEEK_CUR)
    elif vtype == 8:
        _skip_string(fh)
    elif vtype == 9:
        etype, count = _read(fh, "I"), _read(fh, "Q")
        if etype in _GGUF_FIXED:
            fh.seek(struct.calcsize("<" + _GGUF_FIXED[etype]) * count, os.SEEK_CUR)
        else:
            for _ in range(count):
                _skip_value(fh, etype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def gguf_tensor_names(path: Path) -> list[str]:
    """Tensor names from a GGUF file's header (reads only the header, not the weights)."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(fh, "I")
        if version < 2:
            raise ValueError(f"GGUF version {version} is not supported")
        n_tensors, n_kv = _read(fh, "Q"), _read(fh, "Q")
        for _ in range(n_kv):
            _skip_string(fh)
            _skip_value(fh, _read(fh, "I"))
        names = []
        for _ in range(n_tensors):
            names.append(fh.read(_read(fh, "Q")).decode("utf-8"))
            fh.seek(8 * _read(fh, "I") + 4 + 8, os.SEEK_CUR)  # dims, type, offset
        return names


def mlp_layout(model_path: Path) -> str | None:
    """'fused' or 'split' for a Qwen-Image-2.1 GGUF, None when it cannot be told (other formats, other models)."""
    if model_path.suffix.lower() != ".gguf":
        return None
    try:
        names = gguf_tensor_names(model_path)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        log.warning("could not read the tensor names of %s: %s", model_path, exc)
        return None
    if any(n.endswith("transformer_blocks.0.img_mlp.gate_up.weight") for n in names):
        return FUSED
    if any(n.endswith("transformer_blocks.0.img_mlp.proj.weight") for n in names):
        return SPLIT
    return None


def _safetensors_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = _read(fh, "Q")
        return json.loads(fh.read(n)), 8 + n


def _is_shared(key: str) -> bool:
    """Tensors on the input side of the layer (or scalars) apply unchanged to both halves."""
    return any(key.endswith(s) for s in (".lora_A.weight", ".lora_down.weight", ".alpha"))


def split_fused_mlp(src: Path, dst: Path) -> int:
    """Write ``dst``: ``src`` with every img_mlp.gate_up tensor split into gate_layer and proj. Returns how
    many fused tensors were converted (0 = nothing to do, ``dst`` not written)."""
    header, data_start = _safetensors_header(src)
    meta = header.pop("__metadata__", None)
    fused = [k for k in header if ".img_mlp.gate_up." in k]
    if not fused:
        return 0
    out: list[tuple[str, str, list[int], bytes]] = []
    with open(src, "rb") as fh:
        def blob(v: dict) -> bytes:
            a, b = v["data_offsets"]
            fh.seek(data_start + a)
            return fh.read(b - a)

        for k, v in header.items():
            b = blob(v)
            if ".img_mlp.gate_up." not in k:
                out.append((k, v["dtype"], v["shape"], b))
            elif _is_shared(k):
                for part in ("gate_layer", "proj"):
                    out.append((k.replace(".img_mlp.gate_up.", f".img_mlp.{part}."), v["dtype"], v["shape"], b))
            else:  # output side (lora_B / lora_up / dora_scale): rows [gate | proj]
                shape = list(v["shape"])
                if not shape or shape[0] % 2 or len(b) != _numel(shape) * _DTYPE_BYTES.get(v["dtype"], 0):
                    raise ValueError(f"cannot split {k} with shape {shape} ({v['dtype']})")
                half = len(b) // 2
                shape[0] //= 2
                out.append((k.replace(".img_mlp.gate_up.", ".img_mlp.gate_layer."), v["dtype"], shape, b[:half]))
                out.append((k.replace(".img_mlp.gate_up.", ".img_mlp.proj."), v["dtype"], shape, b[half:]))
    new_header: dict = {"__metadata__": meta} if meta else {}
    offset = 0
    for k, dtype, shape, b in out:
        new_header[k] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
    hj = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    hj += b" " * (-len(hj) % 8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)))
        fh.write(hj)
        for _, _, _, b in out:
            fh.write(b)
    os.replace(tmp, dst)
    return len(fused)


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def prepare_lora(lora: Path, lora_root: Path, layout: str | None) -> str:
    """The path to send the engine (relative to ``lora_root``) for ``lora`` on a model with ``layout``.

    For a split-layout model a converted copy is written once to ``lora_root/converted/`` (and rewritten when
    the original changes); every other case uses the file as it is."""
    rel = lora.relative_to(lora_root).as_posix()
    if layout != SPLIT:
        return rel
    header, _ = _safetensors_header(lora)
    if not any(".img_mlp.gate_up." in k for k in header):
        return rel
    dst = lora_root / CONVERTED_DIR / rel
    st = lora.stat()
    if not dst.exists() or dst.stat().st_mtime < st.st_mtime:
        n = split_fused_mlp(lora, dst)
        log.info("converted %d fused MLP tensors of %s to the model's split layout -> %s", n, rel,
                 dst.relative_to(lora_root).as_posix())
    return dst.relative_to(lora_root).as_posix()
