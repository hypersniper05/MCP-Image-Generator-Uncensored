"""Qwen-Image-2.1's official noise schedule, sent to stable-diffusion.cpp as custom sigmas.

stable-diffusion.cpp's default flux schedule shifts much harder than the reference pipeline (mu 1.15 instead
of 0.69 at 1024x1024, 3.2 instead of 1.3 at 2048x2048) and skips the terminal stretch, so the last step
jumps to a clean image from sigma 0.05-0.1 instead of 0.02. Both cost fine detail.
"""
from __future__ import annotations

import math

# Scheduler config of Qwen/Qwen-Image-2.1 (diffusers FlowMatchEulerDiscreteScheduler, exponential time shift).
BASE_SEQ_LEN, MAX_SEQ_LEN = 256, 8192
BASE_SHIFT, MAX_SHIFT = 0.5, 0.9
SHIFT_TERMINAL = 0.02
LATENT_SCALE = 16  # one latent token per 16x16 pixels


def official_mu(width: int, height: int) -> float:
    tokens = (width // LATENT_SCALE) * (height // LATENT_SCALE)
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    return BASE_SHIFT + slope * (tokens - BASE_SEQ_LEN)


def official_sigmas(width: int, height: int, steps: int) -> list[float]:
    """steps + 1 sigmas from 1.0 down to 0.0, as the reference pipeline computes them."""
    steps = max(1, int(steps))
    shift = math.exp(official_mu(width, height))
    sigmas = [1.0 - i / steps for i in range(steps)]  # linspace(1, 1/steps, steps)
    sigmas = [shift / (shift + (1.0 / s - 1.0)) for s in sigmas]
    if steps > 1:
        scale = (1.0 - sigmas[-1]) / (1.0 - SHIFT_TERMINAL)
        sigmas = [1.0 - (1.0 - s) / scale for s in sigmas]
    return [round(s, 6) for s in sigmas] + [0.0]
