"""MiMo-VL (Qwen2.5-VL) text encoder with upstream's CPU-offload placement but no host-RAM copy of the weights.

Upstream (`xvideo.models.models.load_text_encoder`, cpu_offload=True) loads the whole 15.5 GiB model into CPU RAM
with `from_pretrained`, optionally makes a pinned copy, then calls `accelerate.cpu_offload(model, execution_device,
state_dict=...)`, so every forward streams each layer's weights from that host dict to the GPU.
Here the model is built on the meta device with the same class / config / dtype / attention implementation
(`_from_config`, real buffers), and the same `accelerate.cpu_offload` hooks read each weight from the original
safetensors shards on demand. Checkpoint names are mapped with transformers' own `_get_key_renaming_mapping`
(the call from_pretrained makes); coverage is strict both ways.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch


class CoverageError(RuntimeError):
    pass


class LazyCheckpointWeights(Mapping):
    """model parameter name -> tensor read from the original safetensors shards (cast like from_pretrained)."""

    def __init__(self, model, ckpt_dir, dtype: torch.dtype, resident: bool = False, pin: bool = False):
        from safetensors import safe_open

        # resident: keep each tensor in host RAM after its first read (pinned if `pin`, as upstream's TE_PIN copy);
        # the default re-reads from the shards on every access (no host copy at all).
        self.resident, self.pin, self._cache = resident, pin, {}

        self._safe_open = safe_open
        self.dir = Path(ckpt_dir)
        self.dtype = dtype
        index = json.loads((self.dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        original_keys = list(weight_map)
        # the same flags and call as transformers' _load_pretrained_model
        prefix = model.base_model_prefix
        has_prefix_module = any(s.startswith(prefix) for s in original_keys) if len(prefix) > 0 else False
        expects_prefix_module = hasattr(model, prefix) if len(prefix) > 0 else False
        renaming = model._get_key_renaming_mapping(
            original_keys, type(model)._checkpoint_conversion_mapping or None,
            has_prefix_module and not expects_prefix_module, not has_prefix_module and expects_prefix_module)
        self.where = {}
        for ckpt_key, name in renaming.items():
            if name in self.where:
                raise CoverageError(f"two checkpoint keys map to {name}")
            self.where[name] = (weight_map[ckpt_key], ckpt_key)
        expected = model.state_dict()
        missing = sorted(set(expected) - set(self.where))
        unexpected = sorted(set(self.where) - set(expected))
        if missing or unexpected:
            raise CoverageError(f"missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")
        self._handles = {}
        for name, meta in expected.items():
            shard, key = self.where[name]
            shape = tuple(self._handle(shard).get_slice(key).get_shape())
            if shape != tuple(meta.shape):
                raise CoverageError(f"shape mismatch {name}: checkpoint {shape} vs model {tuple(meta.shape)}")
        self.names = list(expected)

    def _handle(self, shard):
        if shard not in self._handles:
            self._handles[shard] = self._safe_open(str(self.dir / shard), "pt", device="cpu")
        return self._handles[shard]

    def __getitem__(self, name):
        if name in self._cache:
            return self._cache[name]
        shard, key = self.where[name]
        t = self._handle(shard).get_tensor(key)
        t = t.to(self.dtype) if t.is_floating_point() else t
        if self.resident:
            if self.pin:
                t = t.pin_memory()
            self._cache[name] = t
        return t

    def cached_bytes(self):
        return sum(t.numel() * t.element_size() for t in self._cache.values())

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


def build_meta_text_encoder(text_encoder_ckpt, torch_dtype):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    config = AutoConfig.from_pretrained(text_encoder_ckpt, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = Qwen2_5_VLForConditionalGeneration._from_config(config, dtype=torch_dtype, attn_implementation="sdpa")
    return model


def load_text_encoder_lowram(text_encoder_ckpt, device=torch.device("cpu"), torch_dtype=torch.bfloat16, cpu_offload=False):
    """Drop-in for upstream `load_text_encoder` when cpu_offload=True (the low-VRAM placement)."""
    if not cpu_offload:
        raise ValueError("load_text_encoder_lowram only replaces the cpu_offload placement")
    from accelerate import cpu_offload as _accelerate_cpu_offload
    from transformers import Qwen2Tokenizer

    model = build_meta_text_encoder(text_encoder_ckpt, torch_dtype)
    model = model.eval().requires_grad_(False)
    import os

    resident = os.environ.get("JOYAI_TE_RESIDENT", "0") == "1"
    pin = False
    if resident:
        from xvideo.lowvram import te_pin_memory_enabled  # upstream's own TE_PIN switch (default on)

        pin = te_pin_memory_enabled()
    weights = LazyCheckpointWeights(model, text_encoder_ckpt, torch_dtype, resident=resident, pin=pin)
    _accelerate_cpu_offload(model, execution_device=torch.device(device), state_dict=weights)
    model.lowram_weights = weights
    tokenizer = Qwen2Tokenizer.from_pretrained(text_encoder_ckpt, local_files_only=True)
    return tokenizer, model
