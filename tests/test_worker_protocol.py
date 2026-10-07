"""The worker's request handling, driven directly over its pipes (no manager)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import frames

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "runtime" / "joyai_worker.py"


class W:
    def __init__(self, tmp: Path, **cfg_over):
        self.root = tmp / "jobs"
        self.root.mkdir()
        self.session = self.root / "worker-20261007-000000-abcdefabcdef"
        self.session.mkdir()
        cfg = {
            "schema_version": 1,
            "worker_id": "worker-20261007-000000-abcdefabcdef",
            "runtime_id": "t",
            "identity": "id",
            "engine": "fake",
            "deploy_dir": "/opt/d",
            "dit_ckpt": "/opt/a",
            "vae_ckpt": "/opt/b",
            "text_encoder_ckpt": "/opt/c",
            "cache_dir": "/opt/e",
            "python": sys.executable,
            "width": 840,
            "height": 480,
            "require_attention": "any",
            "env": {},
            "jobs_root": str(self.root),
        }
        cfg.update(cfg_over)
        (self.session / "worker.json").write_text(json.dumps(cfg), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k.upper() in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME")}
        env["MY_API_TOKEN"] = "secret-value"
        self.p = subprocess.Popen(
            [sys.executable, "-I", "-B", str(WORKER), str(self.session / "worker.json")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(tmp / "w.log", "wb"),
            env=env,
        )
        self.n = 0

    def send(self, obj) -> None:
        self.p.stdin.write((json.dumps(obj) + "\n").encode())
        self.p.stdin.flush()

    def raw(self, b: bytes) -> None:
        self.p.stdin.write(b)
        self.p.stdin.flush()

    def recv(self, timeout=30.0) -> dict:
        line = self.p.stdout.readline()
        assert line, "worker closed stdout"
        return json.loads(line)

    def until(self, pred, timeout=60.0) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            m = self.recv()
            if pred(m):
                return m
        raise AssertionError("timeout")

    def rid(self) -> str:
        self.n += 1
        return f"req-test-{self.n:08d}"

    def job(self, n=4, **over) -> Path:
        jid = f"job-20261007-000000-{self.n:012d}"
        self.n += 1
        d = self.root / jid
        d.mkdir()
        f = frames(n)
        (d / "input.u8").write_bytes(f.tobytes())
        job = {
            "schema_version": 1,
            "job_id": jid,
            "frames": n,
            "height": 72,
            "width": 96,
            "input_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
            "prompt": "p",
            "seed": 1,
            "num_inference_steps": 2,
            "max_temporal_ids": 8,
            "orientation": "landscape",
            "reference": None,
        }
        job.update(over)
        (d / "job.json").write_text(json.dumps(job), encoding="utf-8")
        return d / "job.json"

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        self.p.wait(15)


@pytest.fixture
def w(tmp_path):
    x = W(tmp_path)
    hello = x.recv()
    assert hello["type"] == "hello" and isinstance(hello["pid"], int)  # a venv python.exe may be a launcher (other pid)
    assert hello["dropped_env_keys"] >= 1  # MY_API_TOKEN
    ready = x.recv()
    assert ready["type"] == "ready" and ready["identity"] == "id"
    yield x
    x.close()


def test_status_and_env_scrub(w):
    r = w.rid()
    w.send({"v": 1, "type": "status", "request_id": r})
    st = w.until(lambda m: m.get("request_id") == r)
    assert st["type"] == "status" and st["model_loads"] == 1 and "MY_API_TOKEN" not in st["env_keys"]
    assert "HF_HUB_OFFLINE" in st["env_keys"] and st["busy_with"] is None


def test_edit_happy_path(w):
    r = w.rid()
    w.send({"v": 1, "type": "edit", "request_id": r, "job": str(w.job())})
    assert w.until(lambda m: m.get("request_id") == r)["type"] == "accepted"
    res = w.until(lambda m: m.get("request_id") == r and m["type"] in ("result", "error"))
    assert res["type"] == "result"


@pytest.mark.parametrize(
    "envelope",
    [
        {"v": 2, "type": "status", "request_id": "req-aaaaaaaa"},
        {"v": 1, "type": "status", "request_id": "bad id!"},
        {"v": 1, "type": "status"},
        {"v": 1, "type": "launch", "request_id": "req-bbbbbbbb"},
        {"v": 1, "type": "edit", "request_id": "req-cccccccc"},
        [1, 2, 3],
    ],
)
def test_bad_envelopes(w, envelope):
    w.send(envelope)
    m = w.recv()
    assert m["type"] == "error" and not m["fatal"]


def test_not_json_and_long_lines(w):
    w.raw(b"\xff\xfe not json\n")
    assert w.recv()["error_type"] == "ProtocolError"
    w.raw(b"x" * (70 * 1024) + b"\n")
    assert w.recv()["error_type"] == "ProtocolError"


def test_duplicate_request_id(w):
    w.send({"v": 1, "type": "status", "request_id": "req-dup-0001"})
    w.recv()
    w.send({"v": 1, "type": "status", "request_id": "req-dup-0001"})
    assert w.recv()["error_type"] == "DuplicateRequest"


@pytest.mark.parametrize("make_path", [lambda w: "/etc/passwd", lambda w: str(w.root.parent / "job.json"), lambda w: str(w.root / ".." / "x" / "job.json")])
def test_job_path_outside_root(w, make_path):
    r = w.rid()
    w.send({"v": 1, "type": "edit", "request_id": r, "job": make_path(w)})
    m = w.until(lambda m: m.get("request_id") == r)
    assert m["type"] == "error" and "outside" in m["error"]


def test_input_hash_mismatch(w):
    r = w.rid()
    w.send({"v": 1, "type": "edit", "request_id": r, "job": str(w.job(input_sha256="0" * 64))})
    m = w.until(lambda m: m.get("request_id") == r and m["type"] != "accepted")
    assert m["type"] == "error" and "sha256" in m["error"]


def test_second_edit_while_busy_refused(w):
    r1, r2 = w.rid(), w.rid()
    w.send({"v": 1, "type": "edit", "request_id": r1, "job": str(w.job(n=40, prompt="[fake:slow] p"))})
    w.send({"v": 1, "type": "edit", "request_id": r2, "job": str(w.job())})
    m = w.until(lambda m: m.get("request_id") == r2)
    assert m["type"] == "error" and m["error_type"] == "Busy"
    assert w.until(lambda m: m.get("request_id") == r1 and m["type"] in ("result", "error"))["type"] == "result"


def test_cancel_unknown_target(w):
    r = w.rid()
    w.send({"v": 1, "type": "cancel", "request_id": r, "target": "req-nothing-1"})
    assert w.until(lambda m: m.get("request_id") == r)["error_type"] == "NotRunning"


def test_cancel_file_stops_job(w):
    r = w.rid()
    jp = w.job(n=60, prompt="[fake:slow] p")
    w.send({"v": 1, "type": "edit", "request_id": r, "job": str(jp)})
    w.until(lambda m: m.get("request_id") == r and m["type"] == "progress")
    (jp.parent / "CANCEL").write_text("x")
    m = w.until(lambda m: m.get("request_id") == r and m["type"] in ("cancelled", "result", "error"))
    assert m["type"] == "cancelled"
    assert not (jp.parent / "output.u8").exists() and not (jp.parent / "result.json").exists()


def test_stdin_eof_while_busy_exits(tmp_path):
    x = W(tmp_path)
    x.recv(), x.recv()
    r = x.rid()
    x.send({"v": 1, "type": "edit", "request_id": r, "job": str(x.job(n=200, prompt="[fake:slow] p"))})
    x.until(lambda m: m.get("request_id") == r and m["type"] == "progress")
    x.p.stdin.close()
    assert x.until(lambda m: m["type"] == "bye")["reason"] == "stdin closed while busy"
    assert x.p.wait(15) == 0


def test_shutdown(tmp_path):
    x = W(tmp_path)
    x.recv(), x.recv()
    x.send({"v": 1, "type": "shutdown", "request_id": "req-shutdown-1"})
    assert x.until(lambda m: m["type"] == "bye")["reason"] == "shutdown requested"
    assert x.p.wait(15) == 0
    assert isinstance(json.loads((x.session / "runner.pid").read_text())["pid"], int)


def test_bad_config_exit_code(tmp_path):
    x = W(tmp_path, engine="evil")
    assert x.p.wait(15) == 2


def test_engine_refuses_missing_checkpoints(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("joyai_engine_t", REPO / "runtime" / "joyai_engine.py")
    eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eng)  # module import is stdlib + numpy only; torch / upstream load inside Engine()
    (tmp_path / "dit.pth").write_bytes(b"x")
    (tmp_path / "vae").mkdir()
    te = tmp_path / "MiMo"
    te.mkdir()
    (te / "config.json").write_text("{}")
    cfg = {"dit_ckpt": str(tmp_path / "dit.pth"), "vae_ckpt": str(tmp_path / "vae"), "text_encoder_ckpt": str(te)}
    with pytest.raises(eng.EngineError, match=r"text_encoder_ckpt index \(MiMo\)"):
        eng.check_checkpoints(cfg)
    (te / "model.safetensors.index.json").write_text("{}")
    eng.check_checkpoints(cfg)
    with pytest.raises(eng.EngineError, match=r"dit_ckpt \(gone.pth\).*nothing is downloaded"):
        eng.check_checkpoints({**cfg, "dit_ckpt": str(tmp_path / "gone.pth")})
