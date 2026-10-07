"""Load the pack into a real ComfyUI checkout (COMFYUI_PATH) and run its nodes against the fake runtime."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import fake_expected, frames, runtime_raw, write_config

pytestmark = pytest.mark.comfy
REPO = Path(__file__).resolve().parents[1]
_STATE: dict = {}


@pytest.fixture(scope="module")
def comfy(tmp_path_factory):
    if _STATE:
        return _STATE
    path = os.environ.get("COMFYUI_PATH")
    if not path or not (Path(path) / "comfy").is_dir():
        pytest.fail("COMFYUI_PATH must point to a ComfyUI checkout for tests marked 'comfy'", pytrace=False)
    if path not in sys.path:
        sys.path.insert(0, path)
    root = tmp_path_factory.mktemp("comfy")
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import folder_paths
    import nodes

    for name in ("temp", "input", "output", "user"):
        (root / name).mkdir()
    folder_paths.set_temp_directory(str(root / "temp"))
    folder_paths.set_user_directory(str(root / "user"))
    cfg = write_config(root / "user" / "joyai.runtimes.json", {"fake-runtime": runtime_raw(root)})
    os.environ["COMFYUI_JOYAI_CONFIG"] = str(cfg)
    alias = root / "custom_nodes" / "JoyAI alias ü (test)"
    alias.parent.mkdir()
    shutil.copytree(REPO, alias, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "_comfyui", "tests"))
    assert asyncio.run(nodes.load_custom_node(str(alias)))
    _STATE.update(nodes=nodes, root=root)
    return _STATE


def test_nodes_registered_as_v3(comfy):
    m = comfy["nodes"].NODE_CLASS_MAPPINGS
    for nid in ("JoyAIRuntimeLoader", "JoyAIVideoEdit", "JoyAIUnload"):
        assert nid in m
    info = m["JoyAIRuntimeLoader"].INPUT_TYPES()
    assert info["required"]["runtime"][1]["options"] == ["fake-runtime"]
    edit = m["JoyAIVideoEdit"].INPUT_TYPES()
    assert set(edit["required"]) == {"runtime", "images", "prompt", "seed"}  # v0.1: no unverified knobs
    assert not edit.get("optional")
    assert m["JoyAIUnload"].OUTPUT_NODE


def test_loader_edit_unload_through_the_node_classes(comfy):
    import torch

    m = comfy["nodes"].NODE_CLASS_MAPPINGS
    out = m["JoyAIRuntimeLoader"].execute(runtime="fake-runtime", idle_unload_minutes=0)
    handle, status = out.result
    assert json.loads(status)["state"] == "READY"
    src = frames(9, 64, 96)
    images = torch.from_numpy(src).float() / 255.0
    res = m["JoyAIVideoEdit"].execute(runtime=handle, images=images, prompt="turn the ball blue", seed=42)
    img, report = res.result
    assert img.shape == (9, 480, 840, 3) and img.dtype == torch.float32
    got = (img * 255).round().to(torch.uint8).numpy()
    assert np.array_equal(got, fake_expected(src, "turn the ball blue", 42, 480, 840))  # L0..L3 with the fake engine
    rep = json.loads(report)
    assert rep["frames_in"] == rep["frames_out"] == 9 and "<path>" not in json.dumps(rep["engine"])
    with pytest.raises(ValueError, match="RGB"):
        m["JoyAIVideoEdit"].execute(runtime=handle, images=torch.zeros(2, 64, 64, 4), prompt="x", seed=1)
    with pytest.raises(ValueError, match="prompt"):
        m["JoyAIVideoEdit"].execute(runtime=handle, images=images, prompt="  ", seed=1)
    gen = m["JoyAIRuntimeLoader"].fingerprint_inputs(runtime="fake-runtime", idle_unload_minutes=0)
    un = json.loads(m["JoyAIUnload"].execute(runtime=handle).result[0])
    assert un["unloaded"] is True
    assert m["JoyAIRuntimeLoader"].fingerprint_inputs(runtime="fake-runtime", idle_unload_minutes=0) != gen
    with pytest.raises(RuntimeError, match="not loaded"):
        m["JoyAIVideoEdit"].execute(runtime=handle, images=images, prompt="x", seed=1)


def test_unknown_runtime_message(comfy):
    m = comfy["nodes"].NODE_CLASS_MAPPINGS
    with pytest.raises(RuntimeError, match="not defined"):
        m["JoyAIRuntimeLoader"].execute(runtime="nope", idle_unload_minutes=0)


def test_portrait_input_is_refused_by_the_node(comfy):
    import torch

    m = comfy["nodes"].NODE_CLASS_MAPPINGS
    handle = m["JoyAIRuntimeLoader"].execute(runtime="fake-runtime", idle_unload_minutes=0).result[0]
    try:
        with pytest.raises(ValueError, match="landscape input only"):
            m["JoyAIVideoEdit"].execute(runtime=handle, images=torch.zeros(2, 96, 64, 3), prompt="x", seed=1)
        out = m["JoyAIVideoEdit"].execute(runtime=handle, images=torch.zeros(2, 64, 64, 3), prompt="x", seed=1).result
        assert json.loads(out[1])["profile"] == {"num_inference_steps": 2, "max_temporal_ids": 8, "orientation": "landscape"}
    finally:
        m["JoyAIUnload"].execute()


def test_unverified_inputs_are_ignored_not_honoured(comfy):
    """An old / hand-written workflow that still carries the removed inputs cannot reach them."""
    m = comfy["nodes"].NODE_CLASS_MAPPINGS
    import inspect

    params = set(inspect.signature(m["JoyAIVideoEdit"].execute).parameters)
    assert params == {"runtime", "images", "prompt", "seed"}
