"""Test setup.

All tests here run on CPU without model weights; the model-free fake engine (runtime/joyai_fake_engine.py) stands in
for JoyAI behind the real worker, protocol, job files and process control. Tests marked ``comfy`` import a real
ComfyUI checkout given by ``COMFYUI_PATH`` and fail (not skip) when it is missing; deselect them with
``-m "not comfy"``. Real-GPU checks live in scripts/ (see docs/TESTING.md).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def runtime_raw(tmp_path: Path, **over) -> dict:
    raw = {
        "kind": "posix",
        "python": sys.executable,
        "engine": "fake",
        "deploy_dir": "/opt/joyai/upstream/deploy",
        "dit_ckpt": "/opt/joyai/ckpt/dit.pth",
        "vae_ckpt": "/opt/joyai/ckpt/vae",
        "text_encoder_ckpt": "/opt/joyai/ckpt/te",
        "cache_dir": "/opt/joyai/cache",
        "jobs_dir": str(tmp_path / "jobs"),
        "require_attention": "any",
        "hello_timeout_s": 60,
        "load_timeout_s": 120,
        "cancel_grace_s": 10,
        "memory_guard": {"enabled": False},
    }
    raw.update(over)
    return raw


@pytest.fixture
def make_rt(tmp_path):
    from joyai_comfy import config

    def make(rid: str = "fake", **over):
        return config.parse_runtime(rid, runtime_raw(tmp_path, **over))

    return make


@pytest.fixture
def manager():
    from joyai_comfy.manager import Manager

    m = Manager()
    yield m
    m.unload("test end")


def write_config(path: Path, runtimes: dict) -> Path:
    path.write_text(json.dumps({"schema_version": 1, "runtimes": runtimes}), encoding="utf-8")
    return path


def frames(n: int, h: int = 72, w: int = 96, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def fake_expected(src, prompt: str, seed: int, out_h: int, out_w: int):
    """The fake engine's documented transform (nearest resize, channel swap, shift)."""
    import hashlib

    import numpy as np

    key = int.from_bytes(hashlib.sha256(f"{prompt}\0{seed}".encode()).digest()[:2], "little")
    ys = (np.arange(out_h) * src.shape[1] // out_h).astype(np.int64)
    xs = (np.arange(out_w) * src.shape[2] // out_w).astype(np.int64)
    out = src[:, ys][:, :, xs][..., ::-1].astype(np.uint16)
    return ((out + key) % 256).astype(np.uint8)


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
