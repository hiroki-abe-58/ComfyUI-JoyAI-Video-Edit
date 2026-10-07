"""Streamed DiT loader: the same checkpoint, the same parameters and the same FP8 conversion as upstream's
`xvideo.models.models.load_dit` in low-VRAM mode, without building the whole bf16 model on the CPU first.

Upstream (ca17e1d) builds `Transformer3DModel` on the CPU (fp32 for modules created without factory kwargs, bf16
otherwise), loads the mmap'd checkpoint into it with `load_state_dict(strict=True)`, casts the mixed model to bf16,
then moves the non-block children and each double block to the GPU and installs the FP8 Linears per block.
Here the model is built on the meta device and every tensor is materialised once, at its final dtype, directly where
upstream places it; blocks go one at a time, and upstream's own `_maybe_install_fp8_stream` runs on each block right
after it lands, exactly as in `_move_dit_to_device_lowvram`.
"""

from __future__ import annotations

import os
import time

import torch

_PREFIXES = ("model.", "module.", "transformer.")


class CoverageError(RuntimeError):
    """Checkpoint keys and model tensors do not match one to one."""


def _arch_params(arch_config) -> dict:
    return dict(arch_config.get("params", {})) if isinstance(arch_config, dict) else {}


def read_checkpoint(path: str) -> dict:
    """Upstream's checkpoint read and key normalisation (mmap, `model` unwrap, prefix strip), with collisions refused."""
    mmap_ok = "SPACE_ID" not in os.environ
    state_dict = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap_ok)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    for prefix in _PREFIXES:
        if any(k.startswith(prefix) for k in state_dict):
            renamed = {}
            for k, v in state_dict.items():
                nk = k[len(prefix):] if k.startswith(prefix) else k
                if nk in renamed:
                    raise CoverageError(f"key collision after removing {prefix!r}: {nk}")
                renamed[nk] = v
            state_dict = renamed
    return state_dict


def build_meta_model(cfg, dtype: torch.dtype):
    from xvideo.models.dit import Transformer3DModel

    with torch.device("meta"):
        model = Transformer3DModel(dtype=dtype, device=torch.device("meta"), **_arch_params(cfg.dit_arch_config))
    return model


def _final_dtype(meta_tensor: torch.Tensor, cast_to: torch.dtype | None) -> torch.dtype:
    return cast_to if (cast_to is not None and meta_tensor.is_floating_point()) else meta_tensor.dtype


def _materialise(module, prefix, state_dict, device, cast_to, consumed, stats):
    """Load one module's parameters + persistent buffers from the checkpoint at their final dtype, on `device`."""
    expected = module.state_dict(keep_vars=True)
    values = {}
    for name, meta in expected.items():
        key = prefix + name
        if key not in state_dict:
            raise CoverageError(f"missing in checkpoint: {key}")
        if key in consumed:
            raise CoverageError(f"checkpoint key used twice: {key}")
        src = state_dict[key]
        if key == "img_in.weight" and tuple(src.shape) != tuple(meta.shape):  # upstream's img_in inflate
            src = src.reshape_as(src.new_zeros(meta.shape))
        if tuple(src.shape) != tuple(meta.shape):
            raise CoverageError(f"shape mismatch {key}: checkpoint {tuple(src.shape)} vs model {tuple(meta.shape)}")
        # cast on the CPU as upstream's model.to(dtype) does, then one copy to the target device (never alias the mmap)
        values[name] = src.to(dtype=_final_dtype(meta, cast_to)).to(device, copy=True)
        consumed.add(key)
        stats["tensors"] += 1
        stats["bytes"] += src.numel() * src.element_size()
    module.load_state_dict(values, strict=True, assign=True)


def _remaining_meta(model) -> list[str]:
    names = [n for n, p in model.named_parameters() if p.is_meta]
    names += [n for n, b in model.named_buffers() if b.is_meta]
    return names


def load_dit_streamed(cfg, device: torch.device, log=print, on_block=None) -> torch.nn.Module:
    """Drop-in for upstream `load_dit(cfg, device)` in low-VRAM mode (device must be the execution device)."""
    from xvideo.models.dit.dit import _fp8_stream_wanted, _maybe_install_fp8_stream
    from xvideo.models.pipeline import PRECISION_TO_TYPE

    device = torch.device(device)
    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    if cfg.dit_ckpt is None:
        raise ValueError("load_dit_streamed needs cfg.dit_ckpt")
    t0 = time.time()
    state_dict = read_checkpoint(cfg.dit_ckpt)
    model = build_meta_model(cfg, dtype)
    # upstream casts the whole CPU-built model to `dtype` when its parameters have mixed dtypes
    mixed = len({p.dtype for p in model.parameters()}) > 1
    cast_to = dtype if mixed else None
    if model._parameters or model._buffers:
        raise CoverageError("unexpected root-level parameters / buffers in Transformer3DModel")
    consumed: set[str] = set()
    stats = {"tensors": 0, "bytes": 0}
    for name, child in model.named_children():
        if name != "double_blocks":
            _materialise(child, name + ".", state_dict, device, cast_to, consumed, stats)
    fp8_streams = [s for s in ("img", "txt") if _fp8_stream_wanted(s)]
    for i, block in enumerate(model.double_blocks):
        before = stats["tensors"]
        _materialise(block, f"double_blocks.{i}.", state_dict, device, cast_to, consumed, stats)
        for stream in fp8_streams:
            _maybe_install_fp8_stream(block, stream)
        if on_block is not None:
            on_block(i, stats["tensors"] - before)
    unused = sorted(set(state_dict) - consumed)
    if unused:
        raise CoverageError(f"{len(unused)} checkpoint keys not used, e.g. {unused[:5]}")
    left = _remaining_meta(model)
    if left:
        raise CoverageError(f"{len(left)} model tensors still on meta, e.g. {left[:5]}")
    del state_dict
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log(f"[lowram] DiT streamed: {stats['tensors']} tensors, {stats['bytes'] / 2**30:.2f} GiB from checkpoint, "
        f"cast_to={cast_to}, fp8={fp8_streams}, {time.time() - t0:.1f}s")
    model.lowram_load_stats = {**stats, "cast_to": str(cast_to), "fp8_streams": fp8_streams, "seconds": round(time.time() - t0, 2)}
    return model.eval()
