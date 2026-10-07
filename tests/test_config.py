from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import runtime_raw, write_config

from joyai_comfy import config


def wsl_raw(**over):
    raw = {
        "kind": "wsl",
        "distro": "JoyAI-Ubuntu",
        "user": "root",
        "python": "/opt/joyai/env/bin/python",
        "deploy_dir": "/opt/joyai/upstream/deploy",
        "dit_ckpt": "/opt/joyai/deps/checkpoints/dit.pth",
        "vae_ckpt": "/opt/joyai/deps/checkpoints/vae",
        "text_encoder_ckpt": "/opt/joyai/deps/checkpoints/te",
        "cache_dir": "/opt/joyai/cache_rtx5090",
    }
    raw.update(over)
    return raw


def test_valid_wsl_runtime_defaults():
    rt = config.parse_runtime("joyai-5090", wsl_raw())
    assert rt.kind == "wsl" and rt.distro == "JoyAI-Ubuntu" and rt.user == "root"
    assert (rt.width, rt.height) == (840, 480) and rt.max_frames == 600
    assert rt.memory_guard["max_commit_pct"] == 90.0 and rt.memory_guard["hard_stop_commit_pct"] == 95.0
    assert rt.memory_guard["expected_worker_gib"] == 65.0 and rt.memory_guard["warn_commit_pct"] == 85.0
    assert rt.require_attention == "sage" and rt.engine == "joyai" and not rt.keep_job_dirs


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"surprise": 1}, "unknown keys"),
        ({"kind": "docker"}, "kind must be"),
        ({"distro": "bad name; rm -rf /"}, "distro"),
        ({"distro": "-x"}, "distro"),
        ({"user": "Root$"}, "user"),
        ({"python": "env/bin/python"}, "absolute Linux path"),
        ({"python": "/opt/../bin/sh"}, "'..'"),
        ({"deploy_dir": "/opt/x\n"}, "absolute Linux path"),
        ({"dit_ckpt": "C:\\x.pth"}, "absolute Linux path"),
        ({"env": {"LD_PRELOAD": "/x.so"}}, "env may only set"),
        ({"env": {"CUDA_VISIBLE_DEVICES": "0; echo"}}, "env may only set"),
        ({"engine": "other"}, "engine"),
        ({"memory_max": "lots"}, "memory_max"),
        ({"max_frames": 601}, "max_frames"),
        ({"max_frames": True}, "max_frames"),
        ({"keep_job_dirs": "yes"}, "keep_job_dirs"),
        ({"require_attention": "fa4"}, "require_attention"),
        ({"memory_guard": {"max_commit_pct": 95}}, "may not be above 90"),
        ({"memory_guard": {"hard_stop_commit_pct": 97}}, "may not be above 95"),
        ({"memory_guard": {"expected_worker_gib": 57.7}}, "may not be below 65"),
        ({"memory_guard": {"warn_commit_pct": 88}}, "may not be above 85"),
        ({"memory_guard": {"bogus": 1}}, "memory_guard may only set"),
        ({"jobs_dir": "relative/jobs"}, "jobs_dir"),
    ],
)
def test_bad_values_are_refused(over, msg):
    with pytest.raises(config.ConfigError, match=msg):
        config.parse_runtime("rt", wsl_raw(**over))


@pytest.mark.parametrize("rid", ["", "../x", "a b", "x" * 65, "-lead"])
def test_bad_runtime_ids(rid):
    with pytest.raises(config.ConfigError):
        config.parse_runtime(rid, wsl_raw())


def test_distro_only_for_wsl(tmp_path):
    with pytest.raises(config.ConfigError, match="only valid for kind 'wsl'"):
        config.parse_runtime("p", runtime_raw(tmp_path, distro="JoyAI-Ubuntu"))


def test_guard_can_be_made_stricter_not_looser():
    rt = config.parse_runtime("rt", wsl_raw(memory_guard={"max_commit_pct": 85, "margin_gib": 4, "expected_worker_gib": 70}))
    assert rt.memory_guard["max_commit_pct"] == 85 and rt.memory_guard["margin_gib"] == 4 and rt.memory_guard["expected_worker_gib"] == 70
    assert rt.memory_guard["hard_stop_commit_pct"] == 95.0  # untouched keys keep their defaults


def test_size_is_normalised_to_long_short():
    rt = config.parse_runtime("rt", wsl_raw(width=480, height=840))
    assert (rt.width, rt.height) == (840, 480)


def test_load_runtimes_file(tmp_path):
    p = write_config(tmp_path / "joyai.runtimes.json", {"a": wsl_raw(), "b": wsl_raw(distro="Other")})
    rts = config.load_runtimes(p)
    assert sorted(rts) == ["a", "b"]
    assert config.load_runtimes(tmp_path / "missing.json") == {}


@pytest.mark.parametrize("content", ["not json", json.dumps({"schema_version": 2, "runtimes": {}}), json.dumps({"schema_version": 1, "runtimes": {}, "x": 1})])
def test_broken_files_raise(tmp_path, content):
    p = tmp_path / "joyai.runtimes.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load_runtimes(p)


def test_oversized_file_refused(tmp_path):
    p = tmp_path / "joyai.runtimes.json"
    p.write_text(" " * (config.MAX_CONFIG_BYTES + 1), encoding="utf-8")
    with pytest.raises(config.ConfigError, match="larger than"):
        config.load_runtimes(p)


def test_config_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv(config.CONFIG_ENV, str(tmp_path / "c.json"))
    assert config.config_path() == Path(tmp_path / "c.json")


def test_host_to_runtime_path_mapping():
    rt = config.parse_runtime("rt", wsl_raw())
    assert config.host_to_runtime_path(rt, r"E:\AI\jobs\job-1\job.json") == "/mnt/e/AI/jobs/job-1/job.json"
    assert config.host_to_runtime_path(rt, "C:/x") == "/mnt/c/x"
    for bad in (r"\\server\share\x", "relative\\x", r"E:\a\..\b"):
        with pytest.raises(config.ConfigError):
            config.host_to_runtime_path(rt, bad)
    rt2 = config.parse_runtime("rt", wsl_raw(wsl_mount_root="/win"))
    assert config.host_to_runtime_path(rt2, r"D:\j") == "/win/d/j"
