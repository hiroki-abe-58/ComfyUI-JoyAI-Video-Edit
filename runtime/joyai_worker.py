"""Persistent JoyAI worker: load the model once, then serve edit jobs over stdin/stdout (runtime side, Linux / WSL2).

Usage: python -I -B joyai_worker.py <session_dir>/worker.json

ComfyUI (joyai_comfy/manager.py) starts one worker, keeps the pipe open and sends one JSON request per line; this
process answers on a private copy of the original stdout. Everything else (upstream prints, warnings, tracebacks)
goes to stderr = the session log, so it can never be mistaken for a protocol message. See joyai_protocol.py.

Lifetime: the worker exits when stdin reaches EOF (ComfyUI closed the pipe or died - idle or busy) or on
"shutdown". On every exit it kills the rest of its own process group (compile workers etc.). Jobs run one at a
time; a second "edit" while busy, or a reused request id, is answered with an error. Every job gets a new upstream
session (fresh generator from the seed, KV cache reset, VAE caches cleared); only the loaded weights, the resident
text-encoder cache and compiled kernels / CUDA graphs are reused.

Exit codes: 0 normal, 2 invalid configuration, 3 start-up failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joyai_protocol as P  # noqa: E402

SECRET_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|CREDENTIAL|WANDB|OPENAI|HF_TOKEN)", re.I)
ENV_KEYS = {"CUDA_VISIBLE_DEVICES"}
_CONFIG_KEYS = {
    "schema_version",
    "worker_id",
    "runtime_id",
    "identity",
    "engine",
    "deploy_dir",
    "dit_ckpt",
    "vae_ckpt",
    "text_encoder_ckpt",
    "cache_dir",
    "python",
    "width",
    "height",
    "require_attention",
    "env",
    "jobs_root",
}


class ConfigError(ValueError):
    pass


class JobError(ValueError):
    pass


def load_config(path: Path) -> dict:
    if path.stat().st_size > 64 * 1024:
        raise ConfigError("worker.json too large")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1 or set(cfg) - _CONFIG_KEYS:
        raise ConfigError("unsupported worker.json")
    if not isinstance(cfg.get("worker_id"), str) or not P.ID_RE.fullmatch(cfg["worker_id"]):
        raise ConfigError("invalid worker_id")
    if cfg.get("engine") not in ("joyai", "fake"):
        raise ConfigError("unknown engine")
    for key in ("deploy_dir", "dit_ckpt", "vae_ckpt", "text_encoder_ckpt", "cache_dir", "jobs_root"):
        if not isinstance(cfg.get(key), str) or not os.path.isabs(cfg[key]) or "\0" in cfg[key]:
            raise ConfigError(f"{key} must be an absolute path")
    for key in ("width", "height"):
        if not isinstance(cfg.get(key), int) or not P.MIN_SIDE <= cfg[key] <= P.MAX_SIDE:
            raise ConfigError(f"{key} out of range")
    env = cfg.get("env", {})
    if not isinstance(env, dict) or set(env) - ENV_KEYS or not all(isinstance(v, str) for v in env.values()):
        raise ConfigError("env not allowed")
    if cfg.get("require_attention") not in ("sage", "sdpa", "any"):
        raise ConfigError("require_attention must be sage, sdpa or any")
    return cfg


def apply_env(env: dict) -> list[str]:
    """Drop anything that looks like a credential, apply the allowlisted settings, force offline mode."""
    dropped = [k for k in os.environ if SECRET_RE.search(k)]
    for key in dropped:
        del os.environ[key]
    os.environ.update(env)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # upstream prompt enhancement (PE) must never run: empty model / endpoint / key
    os.environ["PE_MODEL"] = ""
    os.environ["OPENAI_BASE_URL"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    return dropped


def _proc_stat(pid: int) -> list[str] | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            s = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    return s[s.rfind(")") + 2 :].split()


def become_session_leader(pid_file: Path) -> dict:
    """Own process group (so the whole tree can be stopped); record pid, pgid and start time."""
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    me = _proc_stat(os.getpid())
    record = {"pid": os.getpid(), "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None, "starttime": int(me[19]) if me else None}
    tmp = pid_file.with_name(pid_file.name + ".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, pid_file)
    return record


def stop_own_group() -> None:
    """SIGKILL every other member of this worker's process group (only if it leads its own session)."""
    me = os.getpid()
    if not hasattr(os, "getsid") or os.getpgid(0) != me or os.getsid(0) != me or not os.path.isdir("/proc"):
        return
    for entry in os.listdir("/proc"):
        if entry.isdigit() and int(entry) != me:
            st = _proc_stat(int(entry))
            if st and int(st[2]) == me:
                try:
                    os.kill(int(entry), 9)
                except OSError:
                    pass


def _rss() -> dict:
    out = {}
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile"):
                    out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


class Proto:
    """Line-oriented JSON writer on a private fd (thread safe)."""

    def __init__(self, fd: int, worker_id: str):
        self._f = os.fdopen(fd, "w", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self.worker_id = worker_id

    def send(self, msg_type: str, **fields) -> None:
        line = json.dumps(
            {"v": P.PROTOCOL_VERSION, "type": msg_type, "worker_id": self.worker_id, "t": round(time.time(), 3), **fields}, ensure_ascii=False, default=str
        )
        with self._lock:
            try:
                self._f.write(line + "\n")
                self._f.flush()
            except (BrokenPipeError, OSError):
                pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Worker:
    def __init__(self, cfg: dict, proto: Proto, engine_factory, record: dict):
        self.cfg = cfg
        self.proto = proto
        self.stdin_fd = 0
        self.engine_factory = engine_factory
        self.engine = None
        self.record = record
        self.jobs_root = os.path.realpath(cfg["jobs_root"])
        self.requests: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.busy_with: str | None = None
        self.cancel_target: str | None = None
        self.job_dir: Path | None = None
        self.seen_ids: set[str] = set()
        self.started = time.time()
        self.load_seconds = None
        self.jobs_done = self.jobs_failed = self.jobs_cancelled = 0
        self.history: list[dict] = []  # last jobs: request id, job id, frames, seconds (no prompts)

    # -- reader thread ------------------------------------------------------------------------------------
    def _chunk(self) -> bytes:
        """Next bytes from the manager; b"" when the pipe is closed (the manager is gone).

        Linux: a blocking os.read on the raw fd (no Python-level lock held; not sys.stdin.buffer). Windows (test
        runtimes only): never leave a synchronous ReadFile pending on the pipe - while one is pending, other threads'
        calls on that handle (e.g. GetFileType while a DLL such as numpy's loads) block until data arrives, which
        hangs imports (lesson from ComfyUI-LeapTalk). Poll with PeekNamedPipe and read only bytes already there."""
        try:
            if os.name != "nt":
                return os.read(self.stdin_fd, 65536)
            import _winapi
            import msvcrt

            handle = msvcrt.get_osfhandle(self.stdin_fd)
            while True:
                try:
                    avail, _left = _winapi.PeekNamedPipe(handle, 0)
                except OSError:
                    return b""
                if avail:
                    return os.read(self.stdin_fd, min(avail, 65536))
                time.sleep(0.02)
        except OSError:
            return b""

    def reader(self) -> None:
        buf = b""
        while True:
            chunk = self._chunk()
            if not chunk:
                # the parent closed the pipe or died: never keep running on our own, idle or busy
                if self.busy_with is not None or self.engine is None:
                    self.proto.send(
                        "bye", request_id=None, reason="stdin closed while busy" if self.engine else "stdin closed during start-up", jobs_done=self.jobs_done
                    )
                    sys.stderr.flush()
                    stop_own_group()
                    os._exit(0)
                self.requests.put({"type": "_eof"})
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._handle_line(line)
            if len(buf) > P.MAX_LINE:
                buf = b""
                self.proto.send("error", request_id=None, error_type="ProtocolError", error="request line too long", fatal=False)

    def _handle_line(self, line: bytes) -> None:
        if not line.strip():
            return
        if len(line) > P.MAX_LINE:
            self.proto.send("error", request_id=None, error_type="ProtocolError", error="request line too long", fatal=False)
            return
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.proto.send("error", request_id=None, error_type="ProtocolError", error="not UTF-8 JSON", fatal=False)
            return
        rid = msg.get("request_id") if isinstance(msg, dict) else None
        if not isinstance(msg, dict) or msg.get("v") != P.PROTOCOL_VERSION or not isinstance(rid, str) or not P.ID_RE.fullmatch(rid):
            self.proto.send(
                "error",
                request_id=rid if isinstance(rid, str) and P.ID_RE.fullmatch(rid) else None,
                error_type="ProtocolError",
                error="bad request envelope",
                fatal=False,
            )
            return
        kind = msg.get("type")
        with self.lock:
            if rid in self.seen_ids:
                self.proto.send("error", request_id=rid, error_type="DuplicateRequest", error="request_id already used", fatal=False)
                return
            self.seen_ids.add(rid)
            if kind == "status":
                self.proto.send("status", request_id=rid, **self._status())
                return
            if kind == "cancel":
                target = msg.get("target")
                if target is not None and target == self.busy_with:
                    self.cancel_target = target
                    self.proto.send("accepted", request_id=rid, target=target)
                else:
                    self.proto.send("error", request_id=rid, error_type="NotRunning", error="no such running request", fatal=False)
                return
            if kind == "edit":
                if self.engine is None or self.busy_with is not None or not self.requests.empty():
                    self.proto.send("error", request_id=rid, error_type="Busy", error="worker is busy or not loaded; jobs run one at a time", fatal=False)
                    return
                if not isinstance(msg.get("job"), str):
                    self.proto.send("error", request_id=rid, error_type="ProtocolError", error="edit needs a job path", fatal=False)
                    return
                self.busy_with = rid  # reserve before queueing so a second edit is refused
                self.cancel_target = None
                self.requests.put(msg)
                return
            if kind == "shutdown":
                self.requests.put(msg)
                return
        self.proto.send("error", request_id=rid, error_type="ProtocolError", error=f"unknown request type {kind!r}", fatal=False)

    def _status(self) -> dict:
        st = {
            **{k: self.record.get(k) for k in ("pid", "pgid", "starttime")},
            "uptime_s": round(time.time() - self.started, 1),
            "busy_with": self.busy_with,
            "engine_loaded": self.engine is not None,
            "model_loads": 1 if self.engine is not None else 0,
            "load_seconds": self.load_seconds,
            "jobs_done": self.jobs_done,
            "jobs_failed": self.jobs_failed,
            "jobs_cancelled": self.jobs_cancelled,
            "history": self.history[-8:],
            "user": _user(),
            "env_keys": sorted(os.environ),
            **_rss(),
        }
        if self.engine is not None:
            try:
                st["engine"] = self.engine.status()
            except Exception as exc:  # noqa: BLE001
                st["engine_status_error"] = type(exc).__name__
        return st

    def should_cancel(self) -> bool:
        if self.cancel_target is not None and self.cancel_target == self.busy_with:
            return True
        return self.job_dir is not None and (self.job_dir / P.CANCEL_FILE).exists()

    # -- main loop ----------------------------------------------------------------------------------------
    def load(self) -> None:
        t0 = time.time()
        self.engine = self.engine_factory(self.cfg)
        self.load_seconds = round(time.time() - t0, 3)
        self.proto.send(
            "ready",
            request_id=None,
            identity=self.cfg.get("identity"),
            runtime_id=self.cfg.get("runtime_id"),
            load_seconds=self.load_seconds,
            **self.engine.ready_info(),
        )

    def serve(self) -> str:
        while True:
            msg = self.requests.get()
            kind = msg.get("type")
            if kind == "_eof":
                return "stdin closed"
            if kind == "shutdown":
                return "shutdown requested"
            if kind == "edit":
                self._edit(msg)

    def _resolve_job(self, raw: str) -> Path:
        path = os.path.realpath(raw)
        parent = os.path.dirname(path)
        if os.path.dirname(parent) != self.jobs_root or os.path.basename(path) != P.JOB_FILE or not P.ID_RE.fullmatch(os.path.basename(parent)):
            raise JobError("job path outside the jobs root")
        if os.path.islink(raw) or os.path.islink(os.path.dirname(raw)):
            raise JobError("job path is a link")
        return Path(path)

    def _read_frames(self, path: Path, n: int, h: int, w: int, sha: str):
        import numpy as np

        if os.path.islink(path) or not path.is_file():
            raise JobError(f"{path.name} is missing")
        if path.stat().st_size != P.u8_nbytes(n, h, w):
            raise JobError(f"{path.name} has {path.stat().st_size} bytes, expected {P.u8_nbytes(n, h, w)}")
        data = np.fromfile(path, dtype=np.uint8)
        if hashlib.sha256(memoryview(data)).hexdigest() != sha:
            raise JobError(f"{path.name} does not match its sha256")
        return data.reshape(n, h, w, 3)

    def _edit(self, msg: dict) -> None:
        rid = msg["request_id"]
        t0 = time.time()
        job_id = None
        try:
            job_path = self._resolve_job(msg["job"])
            if job_path.stat().st_size > P.MAX_JSON_FILE:
                raise JobError("job.json too large")
            job = P.validate_job(json.loads(job_path.read_text(encoding="utf-8")))
            job_id = job["job_id"]
            job_dir = job_path.parent
            if job_dir.name != job_id:
                raise JobError("job_id does not match its folder")
            if (job_dir / P.OUTPUT_FILE).exists() or (job_dir / P.RESULT_FILE).exists():
                raise JobError("job already has outputs")
            self.job_dir = job_dir
            frames = self._read_frames(job_dir / P.INPUT_FILE, job["frames"], job["height"], job["width"], job["input_sha256"])
            ref = None
            if job["reference"] is not None:
                r = job["reference"]
                ref = self._read_frames(job_dir / P.REF_FILE, 1, r["height"], r["width"], r["sha256"])[0]
            read_s = time.time() - t0
            self.proto.send("accepted", request_id=rid, job_id=job_id)  # from here on the job is never re-sent
            out, info = self.engine.run(
                job,
                frames,
                ref,
                should_cancel=self.should_cancel,
                on_progress=lambda done, chunks: self.proto.send("progress", request_id=rid, frames_done=done, frames_total=job["frames"], chunks_done=chunks),
            )
            if self.should_cancel():
                raise P.JobCancelled()
            t_w = time.time()
            self._write_output(job_dir, out)
            n, h, w = (int(x) for x in out.shape[:3])
            self.jobs_done += 1
            result = {
                "schema_version": 1,
                "job_id": job_id,
                "frames": n,
                "height": h,
                "width": w,
                "output_bytes": P.u8_nbytes(n, h, w),
                "output_sha256": _sha256_file(job_dir / P.OUTPUT_FILE),
                "worker": {
                    "worker_id": self.proto.worker_id,
                    "pid": os.getpid(),
                    "request_id": rid,
                    "job_number_in_worker": self.jobs_done,
                    "model_loads_in_worker": 1,
                    "worker_load_seconds": self.load_seconds,
                    "input_read_seconds": round(read_s, 3),
                    "output_write_seconds": round(time.time() - t_w, 3),
                    "job_seconds": round(time.time() - t0, 3),
                    "user": _user(),
                    **_rss(),
                },
                "engine": info,
            }
            tmp = job_dir / (P.RESULT_FILE + ".tmp")
            tmp.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
            os.replace(tmp, job_dir / P.RESULT_FILE)
            self.history.append({"request_id": rid, "job_id": job_id, "frames": n, "seconds": round(time.time() - t0, 2)})
            self.proto.send("result", request_id=rid, job_id=job_id)
        except P.JobCancelled:
            self.jobs_cancelled += 1
            if self.job_dir is not None:
                for name in (P.OUTPUT_FILE, P.OUTPUT_FILE + ".tmp", P.RESULT_FILE, P.RESULT_FILE + ".tmp"):
                    try:
                        (self.job_dir / name).unlink()
                    except OSError:
                        pass
            self.proto.send("cancelled", request_id=rid, job_id=job_id, **(self.engine.memory_info() if self.engine else {}))
        except Exception as exc:
            self.jobs_failed += 1
            traceback.print_exc(file=sys.stderr)
            fatal = bool(getattr(exc, "fatal", False)) or "CUDA" in str(exc) or "cuda" in type(exc).__module__.lower()
            self.proto.send("error", request_id=rid, job_id=job_id, error_type=type(exc).__name__, error=str(exc)[:2000], fatal=fatal)
            if fatal:
                raise
        finally:
            with self.lock:
                self.busy_with = None
                self.cancel_target = None
                self.job_dir = None

    def _write_output(self, job_dir: Path, out) -> None:
        import numpy as np

        if out.dtype != np.uint8 or out.ndim != 4 or out.shape[-1] != 3:
            raise JobError(f"engine returned {out.dtype} {out.shape}")
        out = np.ascontiguousarray(out)
        tmp = job_dir / (P.OUTPUT_FILE + ".tmp")
        with open(tmp, "wb") as f:
            step = 16
            for i in range(0, out.shape[0], step):
                if self.should_cancel():
                    raise P.JobCancelled()
                f.write(memoryview(out[i : i + step]).cast("B"))
        os.replace(tmp, job_dir / P.OUTPUT_FILE)


def _user() -> str | None:
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_name
    except Exception:  # noqa: BLE001
        return None


def _engine_factory(cfg: dict):
    if cfg["engine"] == "fake":
        import joyai_fake_engine

        return joyai_fake_engine.FakeEngine(cfg)
    import joyai_engine

    return joyai_engine.Engine(cfg)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: joyai_worker.py <worker.json>", file=sys.stderr)
        return 2
    cfg_path = Path(argv[1]).resolve()
    try:
        cfg = load_config(cfg_path)
    except (ConfigError, OSError, json.JSONDecodeError) as exc:
        print(f"invalid worker configuration: {exc}", file=sys.stderr)
        return 2
    dropped = apply_env(cfg.get("env", {}))
    record = become_session_leader(cfg_path.parent / "runner.pid")
    # Protocol on private copies of stdin / stdout; fd 0 -> /dev/null, fd 1 and sys.stdout -> stderr (the session log),
    # so nothing else in this process (upstream prints, DLL runtimes, child processes) touches the pipes.
    proto_fd = os.dup(1)
    in_fd = os.dup(0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.stdin = open(os.devnull, encoding="utf-8")  # noqa: SIM115
    proto = Proto(proto_fd, cfg["worker_id"])
    worker = Worker(cfg, proto, _engine_factory, record)
    worker.stdin_fd = in_fd
    proto.send("hello", request_id=None, user=_user(), dropped_env_keys=len(dropped), **record)
    threading.Thread(target=worker.reader, name="joyai-stdin", daemon=True).start()
    try:
        worker.load()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        proto.send("error", request_id=None, error_type=type(exc).__name__, error=str(exc)[:2000], fatal=True, phase="startup")
        return 3
    try:
        reason = worker.serve()
    except Exception as exc:  # fatal job error (e.g. CUDA): leave, the manager starts a new worker next time
        reason = f"fatal error: {type(exc).__name__}"
    proto.send("bye", request_id=None, reason=reason, jobs_done=worker.jobs_done)
    return 0


if __name__ == "__main__":
    code = main(sys.argv)
    sys.stderr.flush()
    stop_own_group()  # compile workers and other helpers never outlive the worker
    os._exit(code)
