"""Machine-wide singleton: one JoyAI model per distro across processes; stale cleanup never touches live owners."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import runtime_raw

from joyai_comfy import config, singleton
from joyai_comfy import manager as M

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def lock_dir(tmp_path, monkeypatch):
    d = tmp_path / "locks"
    monkeypatch.setenv("JOYAI_LOCK_DIR", str(d))
    return d


def test_lock_is_exclusive_and_reports_its_owner():
    a, b = singleton.MachineLock("wsl-Test"), singleton.MachineLock("wsl-Test")
    assert a.acquire({"runtime_id": "r"}) and a.held
    assert b.acquire() is False and not b.held  # second holder refused, also inside one process
    owner = b.owner()
    assert owner["pid"] == os.getpid() and owner["runtime_id"] == "r"
    a.release()
    assert b.acquire() and b.held
    b.release()
    assert singleton.MachineLock("wsl-Test").acquire()


@pytest.mark.parametrize("name", ["", "../x", "a b", "x" * 101])
def test_bad_lock_names(name):
    with pytest.raises(ValueError):
        singleton.MachineLock(name)


def test_owner_alive():
    assert singleton.owner_alive(singleton.process_identity()) is True
    assert singleton.owner_alive({"pid": os.getpid(), "create_time": 1.0}) is False  # same pid, other start time
    assert singleton.owner_alive(None) is None


def test_second_manager_is_refused_while_the_first_holds_the_runtime(tmp_path):
    rt = config.parse_runtime("fake", runtime_raw(tmp_path))
    m1, m2 = M.Manager(), M.Manager()
    try:
        m1.load(rt)
        with pytest.raises(singleton.RuntimeInUse, match="in use by another ComfyUI process"):
            m2.load(rt)
        assert m2._worker is None
        m1.unload()
        assert m1.state == M.STOPPED and not m1._mlock.held  # released with the worker
        assert m2.load(rt)["state"] == M.READY
    finally:
        m1.unload()
        m2.unload()


RACER = r"""
import json, os, sys, time
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "tests")]
from conftest import runtime_raw
from pathlib import Path
from joyai_comfy import config
from joyai_comfy.manager import Manager
from joyai_comfy.singleton import RuntimeInUse
rt = config.parse_runtime("fake", runtime_raw(Path(sys.argv[2])))
go = Path(sys.argv[3])
while not go.exists():
    time.sleep(0.001)
m = Manager()
try:
    st = m.load(rt)
    print(json.dumps({"result": "loaded", "pid": st["worker"]["pid"]}), flush=True)
    time.sleep(4)  # keep it loaded while the others try
except RuntimeInUse as exc:
    print(json.dumps({"result": "in_use"}), flush=True)
finally:
    m.unload()
"""


def test_simultaneous_start_race_has_exactly_one_winner(tmp_path, lock_dir):
    script = tmp_path / "racer.py"
    script.write_text(RACER, encoding="utf-8")
    go = tmp_path / "GO"
    env = dict(os.environ, JOYAI_LOCK_DIR=str(lock_dir))
    procs = [
        subprocess.Popen([sys.executable, str(script), str(REPO), str(tmp_path / f"r{i}"), str(go)], stdout=subprocess.PIPE, text=True, env=env)
        for i in range(4)
    ]
    time.sleep(2.0)  # all racers waiting on the start flag
    go.write_text("go")
    results = [json.loads(p.communicate(timeout=180)[0].strip().splitlines()[-1])["result"] for p in procs]
    assert sorted(results) == ["in_use", "in_use", "in_use", "loaded"], results


def test_stale_cleanup_never_touches_a_live_owner(tmp_path, monkeypatch):
    rt = config.parse_runtime("fake", runtime_raw(tmp_path))
    root = tmp_path / "jobs"
    root.mkdir()
    calls = []
    monkeypatch.setattr(M, "run_kill_helper", lambda rt, p, grace_s=5.0: calls.append(p.name) or {"status": "stopped"})
    for name, owner in (
        ("worker-20991231-000000-aaaaaaaaaaaa", singleton.process_identity()),  # alive (this process)
        ("worker-20991231-000001-bbbbbbbbbbbb", None),  # unknown
        ("worker-20991231-000002-cccccccccccc", {"pid": os.getpid(), "create_time": 1.0}),
    ):  # gone
        d = root / name
        d.mkdir()
        (d / "runner.pid").write_text(json.dumps({"pid": 4242, "pgid": 4242, "starttime": 1}))
        (d / "worker.json").write_text(json.dumps({"owner": owner}))
    m = M.Manager()
    m._clean_stale(rt)
    assert calls == ["worker-20991231-000002-cccccccccccc"]


def test_memory_guard_is_mandatory_for_wsl(tmp_path):
    raw = {
        "kind": "wsl",
        "distro": "D",
        "python": "/p",
        "deploy_dir": "/d",
        "dit_ckpt": "/a",
        "vae_ckpt": "/b",
        "text_encoder_ckpt": "/c",
        "cache_dir": "/e",
        "memory_guard": {"enabled": False},
    }
    with pytest.raises(config.ConfigError, match="mandatory for WSL"):
        config.parse_runtime("r", raw)
    assert config.parse_runtime("t", runtime_raw(tmp_path)).memory_guard["enabled"] is False  # posix test runtimes only


def test_release_messages_are_not_repeated_every_poll(tmp_path, monkeypatch):
    from joyai_comfy import memory

    rt = config.parse_runtime("fake", runtime_raw(tmp_path, memory_guard={"enabled": True, "min_available_gib": 0.0}))
    state = {"commit": 22.0}
    monkeypatch.setattr(
        memory,
        "snapshot",
        lambda: {
            "measurable": True,
            "commit_enforced": True,
            "commit_gib": state["commit"],
            "commit_limit_gib": 100.0,
            "commit_pct": state["commit"],
            "available_gib": 40.0,
            "t": time.monotonic(),
        },
    )
    m = M.Manager()
    try:
        m.load(rt)
        state["commit"] = 25.5  # released (<= 26) but not admissible (25.5 + 65 > 90)
        m.unload()
        notes = []
        import threading

        threading.Timer(7.0, lambda: state.update(commit=24.0)).start()
        m.load(rt, notify=notes.append)
        waits = [n for n in notes if n.startswith("Waiting")]
        assert len(waits) == 1, notes  # ~3 polls in 7 s, one message
    finally:
        m.unload()
