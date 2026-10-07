"""Workflow input checks and the exact IMAGE <-> uint8 conversion used at both transport points.

ComfyUI IMAGE = torch float32 [B, H, W, C] in 0..1, RGB, on the CPU. The runtime exchanges raw RGB uint8
[N, H, W, 3]. Conversion in: round(x * 255) (exact inverse of u8 / 255 for every byte value); out: u8 / 255.
Nothing is clipped silently: values outside 0..1 or non-finite values are refused.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .protocol import MAX_PROMPT_CHARS, MAX_SIDE, MIN_SIDE, ProtocolError, check_prompt


class InputError(ValueError):
    """A workflow input that the node refuses (shown to the user as is)."""


def validate_images(images, *, max_frames: int, what: str = "images") -> tuple[int, int, int]:
    import torch

    if not isinstance(images, torch.Tensor):
        raise InputError(f"{what}: expected an IMAGE tensor")
    if images.dim() != 4:
        raise InputError(f"{what}: expected an IMAGE batch [B, H, W, C], got {tuple(images.shape)}")
    b, h, w, c = (int(x) for x in images.shape)
    if c != 3:
        raise InputError(f"{what}: {c} channels - convert to RGB first (RGBA / grayscale are not accepted)")
    if not images.is_floating_point():
        raise InputError(f"{what}: expected float values in 0..1, got {images.dtype}")
    if b < 1:
        raise InputError(f"{what}: the batch is empty")
    if b > max_frames:
        raise InputError(f"{what}: {b} frames; this runtime accepts at most {max_frames} frames per job")
    if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
        raise InputError(f"{what}: {w}x{h}; width and height must be within {MIN_SIDE}..{MAX_SIDE}")
    finite = bool(torch.isfinite(images).all())
    if not finite or float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise InputError(f"{what}: IMAGE values must be finite and within 0..1")
    return b, h, w


def validate_prompt(prompt) -> str:
    try:
        return check_prompt(prompt)
    except ProtocolError as exc:
        raise InputError(f"prompt: {exc} (1..{MAX_PROMPT_CHARS} characters, not only whitespace)") from None


def images_to_u8(images) -> np.ndarray:
    """IMAGE float32 0..1 -> C-contiguous uint8 [N, H, W, 3] (round half to even on exact .5, never hit for u8/255)."""
    import torch

    with torch.no_grad():
        u8 = images.detach().to("cpu", torch.float32).mul(255.0).round().to(torch.uint8)
    return np.ascontiguousarray(u8.numpy())


def u8_to_images(frames: np.ndarray):
    import torch

    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected uint8 [N, H, W, 3], got {frames.dtype} {frames.shape}")
    return torch.from_numpy(np.ascontiguousarray(frames)).to(torch.float32).div_(255.0)


def sha256_bytes(buf) -> str:
    return hashlib.sha256(memoryview(buf)).hexdigest()


def orientation_for(choice: str, h: int, w: int) -> str:
    """auto follows the upstream server rule: requested height > width -> portrait, else landscape."""
    if choice == "auto":
        return "portrait" if h > w else "landscape"
    return choice
