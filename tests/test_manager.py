"""The real manager + worker + protocol + job files, with the model-free fake engine in a real child process."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from conftest import fake_expected, frames

from joyai_comfy import manager as M
from joyai_comfy import memory


def edit(m, rt, f, prompt="make it snow", seed=42, **kw):
    return m.edit(rt, f, prompt=prompt, seed=seed, steps=2, temporal_ids=8, orientation=kw.pop("orientation", "landscape"), **kw)


def test_load_edit_reuse_and_exact_transport(manager, make_rt):
    rt = make_rt()
    st = manager.load(rt)
    assert st["state"] == M.READY and st["worker"]["attention_backend"] == "fake"
    pid = st["worker"]["pid"]
    f = frames(12)
    seen = []
    res, out = edit(manager, rt, f, on_progress=lambda d, t: seen.append((d, t)))
    assert out.shape == (12, 480, 840, 3) and out.dtype == np.uint8
    assert np.array_equal(out, fake_expected(f, "make it snow", 42, 480, 840))  # L1 + L3: bytes in == bytes the engine saw
    assert seen[-1] == (12, 12)
    assert res["worker"]["job_number_in_worker"] == 1 and res["worker"]["model_loads_in_worker"] == 1
    # A -> B -> A on the same worker
    res_b, out_b = edit(manager, rt, f, prompt="make it rain")
    res_a2, out_a2 = edit(manager, rt, f)
    assert res_b["worker"]["pid"] == res_a2["worker"]["pid"] == res["worker"]["pid"]
    assert res_a2["worker"]["job_number_in_worker"] == 3
    assert np.array_equal(out_a2, out) and not np.array_equal(out_b, out)
    assert manager.status_dict()["worker"]["pid"] == pid
    assert manager.state == M.READY


def test_portrait_orientation(manager, make_rt):
    rt = make_rt()
    manager.load(rt)
    _, out = edit(manager, rt, frames(3, 96, 72), orientation="portrait")
    assert out.shape == (3, 840, 480, 3)


def test_reference_image_is_transported(manager, make_rt):
    rt = make_rt()
    manager.load(rt)
    res, out = edit(manager, rt, frames(2), reference=frames(1, 80, 64, seed=3))
    assert out.shape[0] == 2


def test_edit_without_load_is_refused(manager, make_rt):
    with pytest.raises(M.WorkerError, match="not loaded"):
        edit(manager, make_rt(), frames(1))


def test_identity_change_restarts(manager, make_rt):
    rt = make_rt()
    p1 = manager.load(rt)["worker"]["pid"]
    rt2 = make_rt(cache_dir="/opt/joyai/other-cache")
    p2 = manager.load(rt2)["worker"]["pid"]
    assert p1 != p2 and manager.generation == 1


def test_load_twice_reuses(manager, make_rt):
    rt = make_rt()
    a = manager.load(rt)["worker"]["worker_id"]
    b = manager.load(rt)["worker"]["worker_id"]
    assert a == b and manager.generation == 0


def test_cancel_mid_job_then_next_job_works(manager, make_rt):
    rt = make_rt()
    manager.load(rt)
    progress = []
    flag = {"stop": False}

    def on_progress(done, total):
        progress.append(done)
        if done >= 16:
            flag["stop"] = True

    with pytest.raises(M.JobCancelled):
        edit(manager, rt, frames(64), prompt="[fake:slow] x", interrupted=lambda: flag["stop"], on_progress=on_progress)
    assert manager.state == M.READY and 16 <= progress[-1] < 64
    _, out = edit(manager, rt, frames(4))
    assert out.shape[0] == 4
    ev = [e for e in manager.events if e["event"] == "job_cancelled"][-1]
    assert ev["clean"] and 16 <= ev["frames_done"] < 64 and ev["worker_seconds"] > 0


def test_cancel_not_confirmed_stops_the_worker(manager, make_rt):
    rt = make_rt(cancel_grace_s=2)
    manager.load(rt)
    t0 = time.time()
    with pytest.raises(M.JobCancelled):
        edit(manager, rt, frames(12), prompt="[fake:hang] x", interrupted=lambda: time.time() - t0 > 1.0)
    assert manager.state == M.ERROR and manager._worker is None and manager.generation == 1
    assert manager.load(rt)["state"] == M.READY  # recovers


def test_worker_crash_mid_job(manager, make_rt):
    rt = make_rt()
    manager.load(rt)
    with pytest.raises(M.WorkerError, match="exited during the job"):
        edit(manager, rt, frames(12), prompt="[fake:crash] x")
    assert manager._worker is None and manager.state == M.ERROR
    with pytest.raises(M.WorkerError, match="not loaded"):
        edit(manager, rt, frames(2))
    manager.load(rt)
    assert edit(manager, rt, frames(2))[1].shape[0] == 2


def test_fatal_error_replaces_worker(manager, make_rt):
    rt = make_rt()
    manager.load(rt)
    with pytest.raises(M.WorkerError, match="FatalEngineError"):
        edit(manager, rt, frames(12), prompt="[fake:fatal] x")
    assert manager._worker is None


def test_job_timeout(manager, make_rt):
    rt = make_rt()
    manager.load(rt)
    with pytest.raises(M.JobTimeout):
        edit(manager, rt, frames(64), prompt="[fake:slow] x", timeout_s=0.5)
    assert manager.state == M.READY
    assert edit(manager, rt, frames(2))[1].shape[0] == 2


def test_cancel_during_load(manager, make_rt, monkeypatch):
    monkeypatch.setenv("JOYAI_FAKE_LOAD_SECONDS", "30")
    rt = make_rt()
    t0 = time.time()
    with pytest.raises(M.JobCancelled):
        manager.load(rt, interrupted=lambda: time.time() - t0 > 1.5)
    assert time.time() - t0 < 25 and manager._worker is None and manager.state == M.STOPPED


def test_load_timeout(manager, make_rt, monkeypatch):
    monkeypatch.setenv("JOYAI_FAKE_LOAD_SECONDS", "30")
    rt = make_rt(load_timeout_s=10)
    with pytest.raises(M.JobTimeout):
        manager.load(rt)
    assert manager._worker is None and manager.state == M.ERROR


def test_unload_and_status_never_starts(manager, make_rt):
    assert manager.status_dict()["worker"] is None
    assert manager.unload()["unloaded"] is False
    rt = make_rt()
    manager.load(rt)
    rep = manager.unload()
    assert rep["unloaded"] and manager.state == M.STOPPED and manager.generation == 1
    assert manager.status_dict()["worker"] is None


def test_concurrent_edits_are_serialised(manager, make_rt):
    rt = make_rt()
    manager.load(rt)
    results = []

    def run(p):
        results.append(edit(manager, rt, frames(16), prompt=f"[fake:slow] {p}")[0]["worker"]["job_number_in_worker"])

    ts = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert sorted(results) == [1, 2, 3]


def snap_of(state):
    return lambda: {
        "measurable": True,
        "commit_enforced": True,
        "commit_gib": state["commit"],
        "commit_limit_gib": 100.0,
        "commit_pct": state["commit"],
        "available_gib": 40.0,
        "t": time.monotonic(),
    }


def test_admission_refuses_start(manager, make_rt, monkeypatch):
    rt = make_rt(memory_guard={"enabled": True})
    monkeypatch.setattr(memory, "snapshot", snap_of({"commit": 26.0}))  # 26 + 65 = 91 > 90 % of 100
    with pytest.raises(M.AdmissionRefused, match="needs 65.0 GiB of commit headroom"):
        manager.load(rt)
    assert manager._worker is None


def test_admission_refuses_unmeasurable(manager, make_rt, monkeypatch):
    rt = make_rt(memory_guard={"enabled": True})
    monkeypatch.setattr(memory, "snapshot", lambda: {"measurable": False, "t": time.monotonic()})
    with pytest.raises(M.AdmissionRefused, match="cannot measure"):
        manager.load(rt)


def test_memory_pressure_stops_a_job_at_95(manager, make_rt, monkeypatch):
    rt = make_rt(memory_guard={"enabled": True, "min_available_gib": 0.0})
    state = {"commit": 20.0}
    monkeypatch.setattr(memory, "snapshot", snap_of(state))
    manager.load(rt)

    def on_progress(done, total):
        if done >= 8:
            state["commit"] = 95.0  # hard stop as soon as 95 % is reached

    with pytest.raises(M.AdmissionRefused, match="memory guard stopped"):
        edit(manager, rt, frames(64), prompt="[fake:slow] x", on_progress=on_progress)
    assert manager.state == M.READY


def test_warning_is_logged_once(manager, make_rt, monkeypatch):
    rt = make_rt(memory_guard={"enabled": True, "min_available_gib": 0.0})
    state = {"commit": 20.0}
    monkeypatch.setattr(memory, "snapshot", snap_of(state))
    manager.load(rt)

    def on_progress(done, total):
        state["commit"] = 86.0

    edit(manager, rt, frames(40), prompt="[fake:slow] x", on_progress=on_progress)
    assert sum(1 for e in manager.events if e["event"] == "memory_warning") == 1


def test_pressure_watch_rules():
    g = {"enabled": True, "warn_commit_pct": 85.0, "hard_stop_commit_pct": 95.0}
    s = lambda pct, t: {"measurable": True, "commit_enforced": True, "commit_pct": pct, "t": t}  # noqa: E731
    w = memory.PressureWatch(g)
    assert not w.update(s(80.0, 0.0)) and not w.warned
    assert not w.update(s(85.0, 1.0)) and w.warned
    assert not w.update(s(94.9, 2.0))
    assert w.update(s(95.0, 3.0)) and "hard stop" in w.reason
    w3 = memory.PressureWatch(g)
    assert not w3.update({"measurable": False, "t": 0.0}) and w3.update({"measurable": False, "t": 5.0})


def test_admission_formula_v0():
    g = dict(__import__("joyai_comfy.config", fromlist=["x"]).GUARD_DEFAULTS)
    snap = {"measurable": True, "commit_enforced": True, "commit_gib": 22.79, "commit_limit_gib": 101.391, "available_gib": 46.0}
    ok, why = memory.check_admission(snap, g, g["expected_worker_gib"], start=True)
    assert ok, why  # 22.79 + 65 = 87.79 <= 91.25 (90 % of 101.391)
    ok, why = memory.check_admission({**snap, "commit_gib": 26.3}, g, 65.0, start=True)
    assert not ok and "91.3 GiB" in why  # 26.3 + 65 = 91.3 > 91.25
    ok, why = memory.check_admission({**snap, "commit_gib": 91.3}, g, 0.1, start=False)
    assert not ok and "no new large-memory stage" in why
    ok, why = memory.check_admission({**snap, "available_gib": 20.0}, g, 65.0, start=True)
    assert not ok and "physical" in why
    ok, why = memory.check_admission({**snap, "commit_gib": 22.0}, g, 65.0, start=True)
    assert ok and "warning" in why  # 87 GiB projected = 85.8 % of the limit
    ok, why = memory.check_admission({**snap, "commit_gib": 15.0}, g, 65.0, start=True)
    assert ok and "warning" not in why  # 80 GiB = 78.9 %


def test_decode_wsl_messages():
    msg = "There is no distribution with the supplied name.\r\n".encode("utf-16-le")
    assert "no distribution" in M._decode_wsl(msg)
    assert M._decode_wsl("plain utf-8 ü".encode()) == "plain utf-8 ü"


def test_missing_distro_is_named(manager, make_rt, monkeypatch):
    from joyai_comfy import config

    raw = {
        "kind": "wsl",
        "distro": "NoSuchDistro",
        "python": "/p",
        "deploy_dir": "/d",
        "dit_ckpt": "/a",
        "vae_ckpt": "/b",
        "text_encoder_ckpt": "/c",
        "cache_dir": "/e",
        "memory_guard": {"enabled": False},
    }
    rt = config.parse_runtime("r", raw)
    monkeypatch.setattr(M, "_wsl_list", lambda rt, *flags: {"JoyAI-Ubuntu", "Other"})
    with pytest.raises(M.WorkerError, match="'NoSuchDistro' is not installed"):
        manager.load(rt)
    assert manager._worker is None


def test_unload_enters_releasing_and_load_waits(manager, make_rt, monkeypatch):
    rt = make_rt(memory_guard={"enabled": True, "min_available_gib": 0.0})
    state = {"commit": 20.0}
    monkeypatch.setattr(memory, "snapshot", snap_of(state))
    manager.load(rt)
    state["commit"] = 50.0  # the VM still holds memory after the worker stopped
    manager.unload()
    assert manager.state == M.RELEASING and manager.status_dict()["release"]["baseline_gib"] == 20.0
    notes = []
    threading.Timer(3.0, lambda: state.update(commit=21.0)).start()
    t0 = time.time()
    st = manager.load(rt, notify=notes.append)
    assert st["state"] == M.READY and time.time() - t0 >= 2.5
    assert notes and notes[0].startswith("Waiting for WSL memory release") and "released" in notes[-1]
    assert any(e["event"] == "memory_released" for e in manager.events)


def test_release_timeout_refuses_load(manager, make_rt, monkeypatch):
    rt = make_rt(memory_guard={"enabled": True, "min_available_gib": 0.0, "release_timeout_s": 3.0})
    state = {"commit": 20.0}
    monkeypatch.setattr(memory, "snapshot", snap_of(state))
    manager.load(rt)
    state["commit"] = 50.0
    manager.unload()
    with pytest.raises(M.ReleaseTimeout, match="not released within 3 s"):
        manager.load(rt)
    assert manager._worker is None and manager.state == M.STOPPED
    assert any(e["event"] == "release_timeout" for e in manager.events)


def test_release_wait_is_interruptible(manager, make_rt, monkeypatch):
    rt = make_rt(memory_guard={"enabled": True, "min_available_gib": 0.0})
    state = {"commit": 20.0}
    monkeypatch.setattr(memory, "snapshot", snap_of(state))
    manager.load(rt)
    state["commit"] = 50.0
    manager.unload()
    t0 = time.time()
    with pytest.raises(M.JobCancelled):
        manager.load(rt, interrupted=lambda: time.time() - t0 > 1.0)
    assert time.time() - t0 < 5


def test_quick_release_goes_straight_to_stopped(manager, make_rt, monkeypatch):
    rt = make_rt(memory_guard={"enabled": True, "min_available_gib": 0.0})
    monkeypatch.setattr(memory, "snapshot", snap_of({"commit": 20.0}))
    manager.load(rt)
    manager.unload()
    assert manager.state == M.STOPPED and manager.release["released"] is True
