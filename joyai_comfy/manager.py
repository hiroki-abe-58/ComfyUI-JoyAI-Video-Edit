"""One JoyAI worker per ComfyUI process: start, load, jobs, cancel, idle unload, stop.

The worker (runtime/joyai_worker.py) runs inside the administrator's WSL2 distro and keeps the model loaded between
queue jobs. ComfyUI talks to it over the stdin/stdout pipes of ``wsl.exe`` (JSON lines; no network port) and
exchanges frames as raw RGB uint8 files in a fresh job folder. The ``wsl.exe`` launcher is created suspended inside
a Job Object with KILL_ON_JOB_CLOSE, so it dies with ComfyUI; the worker then sees EOF on stdin and stops its own
process group. A kill helper (runtime/joyai_kill.py) removes anything left behind, checked by pid, start time and
command line.

At most one worker runs, jobs are serialised by one lock, and a job is never re-sent after the worker accepted it.
State: STOPPED -> STARTING -> LOADING -> READY -> (RUNNING -> READY)* -> STOPPING -> STOPPED; ERROR on failures.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path, PureWindowsPath

import numpy as np

from . import jobs, memory
from . import protocol as P
from .config import Runtime, host_to_runtime_path
from .process import ProcessTree
from .redact import redact
from .singleton import MachineLock, RuntimeInUse, owner_alive, process_identity

PACKAGE_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PACKAGE_DIR / "runtime"
WORKER_SCRIPT = "joyai_worker.py"
KILL_SCRIPT = "joyai_kill.py"
# Files copied into the worker's session folder: the worker runs this frozen copy, so updating the node pack while
# a worker is alive never changes the code it executes (the identity changes -> restart).
CODE_FILES = (
    "joyai_worker.py",
    "joyai_engine.py",
    "joyai_fake_engine.py",
    "joyai_protocol.py",
    "joyai_kill.py",
    "joyai_lowram/__init__.py",
    "joyai_lowram/dit_loader.py",
    "joyai_lowram/te_loader.py",
)
STATUS_TIMEOUT_S = 10.0
SESSION_DIRS_KEPT = 5

STOPPED, STARTING, LOADING, READY, RUNNING, CANCELLING, ERROR, STOPPING, RELEASING = (
    "STOPPED",
    "STARTING",
    "LOADING",
    "READY",
    "RUNNING",
    "CANCELLING",
    "ERROR",
    "STOPPING",
    "RELEASING",  # the worker is gone, the WSL VM has not returned its memory yet (typically 20-80 s)
)


class WorkerError(RuntimeError):
    """The worker failed or stopped; the message is safe to show (paths redacted)."""


class JobCancelled(RuntimeError):
    pass


class JobTimeout(RuntimeError):
    pass


class AdmissionRefused(RuntimeError):
    pass


class ReleaseTimeout(AdmissionRefused):
    pass


def code_digest() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode())
        h.update((RUNTIME_DIR / name).read_bytes())
    return h.hexdigest()


def identity(rt: Runtime) -> str:
    """Everything that decides which model, settings and code a worker has loaded."""
    data = {
        "protocol": P.PROTOCOL_VERSION,
        "runtime_id": rt.id,
        "kind": rt.kind,
        "distro": rt.distro,
        "user": rt.user,
        "python": rt.python,
        "deploy_dir": rt.deploy_dir,
        "dit_ckpt": rt.dit_ckpt,
        "vae_ckpt": rt.vae_ckpt,
        "text_encoder_ckpt": rt.text_encoder_ckpt,
        "cache_dir": rt.cache_dir,
        "engine": rt.engine,
        "memory_max": rt.memory_max,
        "size": [rt.width, rt.height],
        "require_attention": rt.require_attention,
        "env": dict(sorted(rt.env.items())),
        "code": code_digest(),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _wsl_exe() -> str:
    root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or r"C:\Windows"
    return str(PureWindowsPath(root) / "System32" / "wsl.exe")


def launcher_argv(rt: Runtime, script_path: str, *args: str, scope: bool = False) -> list[str]:
    """argv that runs one of this package's scripts with the runtime's Python. Fixed list, never a shell."""
    py = [rt.python, "-I", "-B", script_path, *args]
    if rt.kind == "wsl":
        head = [_wsl_exe(), "-d", rt.distro]
        if rt.user:
            head += ["-u", rt.user]
        head += ["--cd", "/", "--exec"]
        if scope and rt.memory_max:
            head += ["/usr/bin/systemd-run", "--scope", "--quiet", "-p", f"MemoryMax={rt.memory_max}", "--"]
        return head + py
    return py


def child_env(rt: Runtime) -> dict[str, str]:
    """Minimal environment for the launcher. Nothing secret, nothing forwarded into Linux (WSLENV empty)."""
    src = {k.upper(): v for k, v in os.environ.items()} if os.name == "nt" else dict(os.environ)
    if rt.kind == "wsl":
        keep = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC", "PROGRAMDATA", "LOCALAPPDATA", "USERPROFILE")
        env = {k: src[k] for k in keep if k in src}
        sysroot = env.get("SYSTEMROOT", r"C:\Windows")
        env["PATH"] = f"{sysroot}\\System32;{sysroot}"
        env["WSLENV"] = ""
        return env
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME", "JOYAI_FAKE_LOAD_SECONDS")
    if os.name == "nt":  # only the test suite runs 'posix' runtimes on a Windows host; Python needs these there
        keep += ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP")
    env = {k: src[k] for k in keep if k in src}
    env.setdefault("PATH", "/usr/bin:/bin")
    return env


def _decode_wsl(raw: bytes) -> str:
    """wsl.exe writes its own messages as UTF-16; Linux output is UTF-8."""
    return raw.decode("utf-16-le", "replace") if raw.count(b"\x00") > len(raw) // 4 else raw.decode("utf-8", "replace")


def _wsl_list(rt: Runtime, *flags: str) -> set[str] | None:
    """Distro names from `wsl.exe -l -q [flags]` (read-only; never starts a distro). None if unknown."""
    try:
        out = subprocess.run(
            [_wsl_exe(), "-l", *flags, "-q"],
            capture_output=True,
            timeout=30,
            env=child_env(rt),  # noqa: S603
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {line.strip().strip("\ufeff") for line in _decode_wsl(out.stdout).splitlines() if line.strip()}


def distro_running(rt: Runtime) -> bool:
    """True if the configured distro is running (never starts it). Non-WSL runtimes: True."""
    if rt.kind != "wsl":
        return True
    names = _wsl_list(rt, "--running")
    return True if names is None else rt.distro in names  # unknown: let the caller try (the helper refuses anything foreign)


def distro_installed(rt: Runtime) -> bool:
    if rt.kind != "wsl":
        return True
    names = _wsl_list(rt)
    return True if names is None else rt.distro in names


def run_kill_helper(rt: Runtime, session_dir: Path, grace_s: float = 5.0) -> dict:
    argv = launcher_argv(rt, host_to_runtime_path(rt, RUNTIME_DIR / KILL_SCRIPT), host_to_runtime_path(rt, session_dir), str(grace_s))
    try:
        done = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=child_env(rt),
            timeout=90,  # noqa: S603
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        lines = done.stdout.decode("utf-8", "replace").strip().splitlines()
        return json.loads(lines[-1]) if lines else {"status": "no_output", "returncode": done.returncode}
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return {"status": "kill_helper_failed", "error": type(exc).__name__}


class WorkerProcess:
    def __init__(self, rt: Runtime, ident: str):
        self.rt = rt
        self.identity = ident
        self.worker_id, self.session_dir = jobs.new_dir(rt, "worker")
        self.messages: queue.Queue = queue.Queue()
        self.tree: ProcessTree | None = None
        self.hello: dict | None = None
        self.ready: dict | None = None
        self.bye: dict | None = None
        self.started_at = time.time()
        self.jobs = 0
        self.stop_report: dict | None = None
        self.preload_commit: float | None = None  # commit charge right before this worker's model load (release baseline)

    def start(self) -> None:
        code = self.session_dir / "code"
        for name in CODE_FILES:
            dst = code / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RUNTIME_DIR / name, dst)
        rt = self.rt
        cfg = {
            "schema_version": 1,
            "worker_id": self.worker_id,
            "runtime_id": rt.id,
            "identity": self.identity,
            "engine": rt.engine,
            "deploy_dir": rt.deploy_dir,
            "dit_ckpt": rt.dit_ckpt,
            "vae_ckpt": rt.vae_ckpt,
            "text_encoder_ckpt": rt.text_encoder_ckpt,
            "cache_dir": rt.cache_dir,
            "python": rt.python,
            "width": rt.width,
            "height": rt.height,
            "require_attention": rt.require_attention,
            "env": dict(rt.env),
            "jobs_root": host_to_runtime_path(rt, jobs.jobs_root(rt)),
            "owner": process_identity(),  # the ComfyUI process that owns this worker (stale cleanup never touches live owners)
        }
        jobs.write_json(self.session_dir / "worker.json", cfg)
        argv = launcher_argv(rt, host_to_runtime_path(rt, code / WORKER_SCRIPT), host_to_runtime_path(rt, self.session_dir / "worker.json"), scope=True)
        self.tree = ProcessTree(argv, cwd=self.session_dir, env=child_env(rt), stdout=None, stderr=self.session_dir / "worker.log")
        threading.Thread(target=self._read, name=f"{self.worker_id}-reader", daemon=True).start()

    def _read(self) -> None:
        assert self.tree is not None and self.tree.proc.stdout is not None
        for raw in self.tree.proc.stdout:
            if len(raw) > P.MAX_REPLY:
                continue
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(msg, dict) or msg.get("worker_id") != self.worker_id:
                continue  # not from this worker: never mix up replies
            if msg.get("type") == "bye":
                self.bye = msg
            self.messages.put(msg)
        self.messages.put({"type": "_closed"})

    def alive(self) -> bool:
        return self.tree is not None and self.tree.poll() is None

    def send(self, msg_type: str, **fields) -> str:
        rid = fields.pop("request_id", None) or f"req-{uuid.uuid4().hex}"
        line = json.dumps({"v": P.PROTOCOL_VERSION, "type": msg_type, "request_id": rid, **fields}) + "\n"
        if len(line) > P.MAX_LINE:
            raise WorkerError("request too large")
        if self.tree is None or self.tree.proc.stdin is None or self.tree.proc.stdin.closed:
            raise WorkerError("worker is not running")
        try:
            self.tree.proc.stdin.write(line.encode("utf-8"))
            self.tree.proc.stdin.flush()
        except OSError as exc:
            raise WorkerError("worker pipe is closed") from exc
        return rid

    def log_tail(self, n: int = 1500) -> str:
        try:
            return redact(_decode_wsl((self.session_dir / "worker.log").read_bytes()[-n:]).strip())
        except OSError:
            return ""

    def stop(self, reason: str, grace_s: float = 30.0) -> dict:
        """Cooperative shutdown, then the kill helper, then the Job Object."""
        report: dict = {"reason": reason}
        if self.alive():
            try:
                self.send("shutdown")
            except WorkerError:
                pass
        if self.tree is not None:
            self.tree.close_stdin()
            try:
                self.tree.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
        report["exited_cleanly"] = self.tree is not None and self.tree.poll() is not None
        helper_possible = (self.rt.kind == "wsl" and distro_running(self.rt)) or (self.rt.kind == "posix" and os.name != "nt")
        if self.hello is not None and helper_possible:
            report["kill_helper"] = run_kill_helper(self.rt, self.session_dir)
        if self.tree is not None:
            report["tree"] = self.tree.terminate(grace_s=5.0)
            self.tree.close()
        self.stop_report = report
        return report


class Manager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker: WorkerProcess | None = None
        self.state = STOPPED
        self.generation = 0  # bumped whenever a worker goes away: the Loader node re-runs
        self.events: deque = deque(maxlen=60)  # lifecycle log (no prompts, no paths)
        self.last_error: str | None = None
        self._idle_timer: threading.Timer | None = None
        self._idle_minutes = 0
        self._cleaned_roots: set[str] = set()
        self.job_counter = 0
        self._release_done = threading.Event()
        self._release_done.set()
        self.release: dict = {}  # last release: started, baseline, seconds, released, commit
        self._mlock: MachineLock | None = None  # machine-wide: one JoyAI model per distro across ComfyUI processes

    # -- helpers ----------------------------------------------------------------------------------------
    def _note(self, event: str, **fields) -> None:
        ev = {"t": round(time.time(), 1), "event": event, "state": self.state, **fields}
        self.events.append(ev)
        logging.info("[JoyAI] %s", json.dumps(ev, default=str))  # lifecycle only: no prompts, no paths

    def _set(self, state: str) -> None:
        self.state = state

    def _discard(self, reason: str, *, error: bool = False) -> dict | None:
        w, self._worker = self._worker, None
        self._cancel_idle()
        if w is None:
            self._set(ERROR if error else STOPPED)
            self._release_machine_lock_if_idle()
            return None
        self._set(STOPPING)
        report = w.stop(reason)
        self.generation += 1
        if error:
            self.last_error = reason
        self._begin_release(w, error)  # session folder kept for its log; pruned at the next start
        self._note(
            "worker_stopped",
            worker_id=w.worker_id,
            reason=redact(reason),
            jobs=w.jobs,
            exited_cleanly=report.get("exited_cleanly"),
            helper=(report.get("kill_helper") or {}).get("status"),
        )
        return report

    def _begin_release(self, w: WorkerProcess, error: bool) -> None:
        """RELEASING until the commit charge is back near its pre-load level (polled every 2 s, at most
        release_timeout_s); then STOPPED (or ERROR). A new model load waits for this instead of starting on residual
        commit."""
        guard = w.rt.memory_guard
        final = ERROR if error else STOPPED
        baseline = getattr(w, "preload_commit", None)
        self.release = {
            "started": time.time(),
            "baseline_gib": baseline,
            "tolerance_gib": guard["release_tolerance_gib"],
            "timeout_s": guard["release_timeout_s"],
            "released": None,
        }
        if not guard["enabled"] or baseline is None or memory.released(memory.snapshot(), baseline, guard):
            self.release.update(released=True, seconds=0.0)
            self._set(final)
            self._release_machine_lock_if_idle()
            self._release_done.set()
            return
        self._set(RELEASING)
        self._release_done.clear()
        rel = self.release

        def watch() -> None:
            deadline = time.monotonic() + guard["release_timeout_s"]
            snap: dict = {}
            while time.monotonic() < deadline:
                snap = memory.snapshot()
                if memory.released(snap, baseline, guard):
                    rel.update(released=True, seconds=round(time.time() - rel["started"], 1), commit_gib=snap.get("commit_gib"))
                    break
                self._release_done.wait(2.0)  # sleeps 2 s (the event is only set by this thread)
            else:
                rel.update(released=False, seconds=round(time.time() - rel["started"], 1), commit_gib=snap.get("commit_gib"))
            if self.state == RELEASING:
                self._set(final)
            self._release_machine_lock_if_idle()
            self._note("memory_released" if rel["released"] else "release_timeout", **{k: v for k, v in rel.items() if k != "started"})
            self._release_done.set()

        threading.Thread(target=watch, name="joyai-release", daemon=True).start()

    def _wait_release(self, rt: Runtime, interrupted: Callable[[], bool], notify: Callable[[str], None] | None) -> None:
        """Before a model load: if a worker stopped recently, wait (interruptible, inside the release window of
        release_timeout_s after the stop) until its memory is released AND the load would pass admission. The VM can
        still hold a few GiB after the release criterion is met, so both are checked."""
        rel = self.release
        if not rel or not rel.get("started"):
            return
        guard = rt.memory_guard
        deadline = rel["started"] + rel.get("timeout_s", guard["release_timeout_s"])
        told = False
        last = {"key": None, "t": 0.0}

        def tell(text: str, key: str) -> None:
            # one line when the reason changes, then at most every 30 s (no 2 s spam in the log / on the node)
            nonlocal told
            if not told:
                self._note("waiting_for_release")
                told = True
            now = time.monotonic()
            if notify and (key != last["key"] or now - last["t"] >= 30.0):
                notify(text)
                last.update(key=key, t=now)

        if not self._release_done.is_set():
            tell(
                f"Waiting for WSL memory release (commit back to <= {rel['baseline_gib'] + rel['tolerance_gib']:.1f} GiB expected; up to "
                f"{rel['timeout_s']:.0f} s after the previous worker stopped)",
                "release",
            )
            while not self._release_done.wait(0.5):
                if interrupted():
                    raise JobCancelled("cancelled while waiting for the WSL memory release")
            if not rel.get("released"):
                raise ReleaseTimeout(
                    f"WSL memory was not released within {rel['timeout_s']:.0f} s after the previous worker stopped (commit "
                    f"{rel.get('commit_gib')} GiB); a new model load was not started. Retry in a minute, or check what else holds memory."
                )
        # released; within the window, also wait until a load would be admitted (the VM may still be shrinking)
        while time.time() < deadline:
            ok, why = memory.check_admission(memory.snapshot(), guard, guard["expected_worker_gib"], start=True)
            if ok:
                if told and notify:
                    notify(f"WSL memory released after {round(time.time() - rel['started'], 1)} s")
                return
            tell(f"Waiting for WSL memory release: {why}", "admission")
            end = time.monotonic() + 2.0
            while time.monotonic() < end:
                if interrupted():
                    raise JobCancelled("cancelled while waiting for the WSL memory release")
                time.sleep(0.25)

    @staticmethod
    def _lock_name(rt: Runtime) -> str:
        return f"wsl-{rt.distro}" if rt.kind == "wsl" else f"posix-{rt.id}"

    def _acquire_machine_lock(self, rt: Runtime) -> None:
        name = self._lock_name(rt)
        if self._mlock is not None and self._mlock.name != name and self._worker is None:
            self._mlock.release()
            self._mlock = None
        if self._mlock is None:
            self._mlock = MachineLock(name)
        if self._mlock.acquire({"runtime_id": rt.id}):
            return
        owner = self._mlock.owner() or {}
        self._note("machine_lock_busy", lock=name, owner_pid=owner.get("pid"))
        raise RuntimeInUse(
            f"the JoyAI runtime for WSL distro {rt.distro!r} is in use by another ComfyUI process"
            + (f" (pid {owner.get('pid')})" if owner.get("pid") else "")
            + "; only one JoyAI model may be loaded per distro on this machine - unload it there (JoyAI Unload) and retry"
        )

    def _release_machine_lock_if_idle(self) -> None:
        if self._mlock is not None and self._worker is None and self.state in (STOPPED, ERROR):
            self._mlock.release()

    def _cancel_idle(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _arm_idle(self) -> None:
        self._cancel_idle()
        if self._idle_minutes > 0 and self._worker is not None:
            t = threading.Timer(self._idle_minutes * 60.0, self._idle_fire)
            t.daemon = True
            t.start()
            self._idle_timer = t

    def _idle_fire(self) -> None:
        if not self._lock.acquire(blocking=False):
            return  # a job or a load is running: the timer is re-armed when it ends
        try:
            if self._worker is not None and self.state == READY:
                self._discard(f"idle for {self._idle_minutes} min")
        finally:
            self._lock.release()

    def _clean_stale(self, rt: Runtime) -> None:
        """Once per jobs root: for worker sessions and job folders whose owning ComfyUI process is confirmed gone
        (owner record: pid + start time), stop any leftover worker and delete the folders, keeping the newest few
        session logs. Folders of a live or unknown owner - another ComfyUI process - are never touched."""
        root = jobs.jobs_root(rt)
        if str(root) in self._cleaned_roots:
            return
        self._cleaned_roots.add(str(root))

        def owner_of(p: Path):
            for name, key in (("worker.json", "owner"), (jobs.OWNER_FILE, None)):
                try:
                    data = json.loads((p / name).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                return data.get(key) if key else data
            return None

        dead_sessions, dead_jobs = [], []
        for p in sorted(root.iterdir(), key=lambda x: x.name):
            if not p.is_dir() or not (p.name.startswith("worker-") or p.name.startswith("job-")):
                continue
            if owner_alive(owner_of(p)) is not False:
                continue  # owner alive or unknown: never touch another process's folders or worker
            (dead_sessions if p.name.startswith("worker-") else dead_jobs).append(p)
        running = None
        for p in dead_sessions:
            if (p / "runner.pid").is_file():
                if running is None:
                    running = distro_running(rt)
                if running:
                    rep = run_kill_helper(rt, p)
                    if rep.get("status") not in ("not_running", "no_runner_record"):
                        self._note("stale_worker", session=p.name, helper=rep.get("status"))
        if rt.keep_job_dirs:
            return
        for p in dead_sessions[:-SESSION_DIRS_KEPT] + dead_jobs:
            jobs.remove(rt, p)

    # -- start / load -----------------------------------------------------------------------------------
    def _wait(self, w: WorkerProcess, kinds: tuple[str, ...], timeout_s: float, interrupted: Callable[[], bool], phase: str) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                msg = w.messages.get(timeout=0.2)
            except queue.Empty:
                if interrupted():
                    raise JobCancelled(f"cancelled while the worker was {phase}") from None
                if not w.alive():
                    raise WorkerError(f"worker exited while {phase} (exit {w.tree.poll()}): {w.log_tail()}") from None
                continue
            kind = msg.get("type")
            if kind in kinds:
                return msg
            if kind == "error" and msg.get("fatal"):
                raise WorkerError(f"worker failed while {phase}: {msg.get('error_type')}: {redact(msg.get('error'))}")
            if kind == "_closed":
                raise WorkerError(f"worker exited while {phase}: {w.log_tail()}")
        raise JobTimeout(f"worker did not finish {phase} within {timeout_s:.0f} s")

    def load(self, rt: Runtime, *, idle_minutes: int = 0, interrupted: Callable[[], bool] = lambda: False, notify: Callable[[str], None] | None = None) -> dict:
        with self._lock:
            self._idle_minutes = int(idle_minutes)
            ident = identity(rt)
            w = self._worker
            if w is not None:
                if not w.alive():
                    why = (w.bye or {}).get("reason", "process ended")
                    self._discard(f"worker had exited ({why})", error=True)
                elif w.identity != ident:
                    self._discard("runtime, code or settings changed")
                elif self.state == READY:
                    self._arm_idle()
                    return self.status_dict(live=False)
            if not distro_installed(rt):
                self.last_error = f"WSL distro {rt.distro!r} is not installed (see `wsl.exe -l -v`)"
                raise WorkerError(f"{self.last_error}; check the runtime {rt.id!r} in the JoyAI runtime config")
            self._wait_release(rt, interrupted, notify)  # never start a model load on residual commit of the previous worker
            self._acquire_machine_lock(rt)  # never two JoyAI models of one distro on this machine (any ComfyUI process)
            self._clean_stale(rt)
            snap = memory.snapshot()
            ok, why = memory.check_admission(snap, rt.memory_guard, rt.memory_guard["expected_worker_gib"], start=True)
            self._note("admission_start", ok=ok, commit_gib=snap.get("commit_gib"), limit_gib=snap.get("commit_limit_gib"), detail=why)
            if not ok:
                self.last_error = why
                self._release_machine_lock_if_idle()
                raise AdmissionRefused(why)
            w = WorkerProcess(rt, ident)
            w.preload_commit = snap.get("commit_gib")
            self._worker = w
            self._set(STARTING)
            self._note("worker_starting", worker_id=w.worker_id, runtime_id=rt.id)
            t0 = time.time()
            try:
                w.start()
                w.hello = self._wait(w, ("hello",), rt.hello_timeout_s, interrupted, "starting")
                self._set(LOADING)
                self._note("worker_hello", pid=w.hello.get("pid"))
                w.ready = self._wait(w, ("ready",), rt.load_timeout_s, interrupted, "loading the model")
                if w.ready.get("identity") != ident:
                    raise WorkerError("worker reported a different identity than requested")
            except BaseException as exc:
                self._discard(f"start-up failed: {type(exc).__name__}: {exc}", error=not isinstance(exc, JobCancelled))
                raise
            self._set(READY)
            self.last_error = None
            self._note(
                "worker_ready",
                pid=w.hello.get("pid"),
                load_s=round(time.time() - t0, 1),
                attention=w.ready.get("attention_backend"),
                fp8=w.ready.get("fp8_blocks"),
            )
            self._arm_idle()
            return self.status_dict(live=False)

    # -- jobs -------------------------------------------------------------------------------------------
    def edit(
        self,
        rt: Runtime,
        frames: np.ndarray,
        *,
        prompt: str,
        seed: int,
        steps: int,
        temporal_ids: int,
        orientation: str,
        interrupted: Callable[[], bool] = lambda: False,
        on_progress: Callable[[int, int], None] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[dict, np.ndarray]:
        with self._lock:
            w = self._worker
            if w is None or self.state != READY or w.identity != identity(rt):
                raise WorkerError("JoyAI runtime is not loaded (run the JoyAI Runtime node first)")
            if not w.alive():
                why = (w.bye or {}).get("reason", "process ended")
                self._discard(f"worker had exited ({why})", error=True)
                raise WorkerError(f"worker stopped unexpectedly ({why}); run the JoyAI Runtime node again")
            self._cancel_idle()
            n, h, wd = (int(x) for x in frames.shape[:3])
            out_h, out_w = (rt.width, rt.height) if orientation == "portrait" else (rt.height, rt.width)
            est = memory.job_estimate_gib(n, h, wd, out_h, out_w)
            snap = memory.snapshot()
            ok, why = memory.check_admission(snap, rt.memory_guard, rt.memory_guard["job_factor"] * est, start=False)
            self._note("admission_job", ok=ok, estimate_gib=round(est, 2), commit_gib=snap.get("commit_gib"), limit_gib=snap.get("commit_limit_gib"))
            if not ok:
                self._arm_idle()
                raise AdmissionRefused(why)
            job_id, job_dir = jobs.new_dir(rt, "job")
            t0 = time.time()
            try:
                job = {
                    "schema_version": 1,
                    "job_id": job_id,
                    "frames": n,
                    "height": h,
                    "width": wd,
                    "input_sha256": jobs.write_u8(jobs.child(job_dir, P.INPUT_FILE), frames),
                    "prompt": prompt,
                    "seed": int(seed),
                    "num_inference_steps": int(steps),
                    "max_temporal_ids": int(temporal_ids),
                    "orientation": orientation,
                    "reference": None,
                }
                P.validate_job(job)
                jobs.write_json(jobs.child(job_dir, P.JOB_FILE), job)
                write_s = time.time() - t0
                self.job_counter += 1
                timeout = timeout_s if timeout_s is not None else rt.job_timeout_base_s + rt.job_timeout_per_frame_s * n
                rid = w.send("edit", job=host_to_runtime_path(rt, jobs.child(job_dir, P.JOB_FILE)))
                self._set(RUNNING)
                self._note("job_sent", job_id=job_id, frames=n, request_id=rid)
                self._follow(w, rid, n, on_progress, interrupted, time.monotonic() + timeout, job_dir)
                t_read = time.time()
                result, out = jobs.read_result(job_dir, job)
                result.setdefault("manager", {}).update(
                    job_number_in_manager=self.job_counter,
                    wall_seconds=round(time.time() - t0, 3),
                    input_write_seconds=round(write_s, 3),
                    output_read_seconds=round(time.time() - t_read, 3),
                    worker_pid=(w.hello or {}).get("pid"),
                    worker_id=w.worker_id,
                    job_dir_kept=rt.keep_job_dirs,
                )
                w.jobs += 1
                self._note("job_done", job_id=job_id, frames=n, wall_s=round(time.time() - t0, 2))
                return result, out
            finally:
                if self._worker is w and self.state in (RUNNING, CANCELLING):
                    self._set(READY)
                if not rt.keep_job_dirs:
                    jobs.remove(rt, job_dir)
                if self._worker is not None:
                    self._arm_idle()

    def _follow(self, w: WorkerProcess, rid: str, n: int, on_progress, interrupted, deadline: float, job_dir: Path) -> None:
        watch = memory.PressureWatch(w.rt.memory_guard)
        warned = False
        next_mem = 0.0
        try:
            while True:
                try:
                    msg = w.messages.get(timeout=0.2)
                except queue.Empty:
                    msg = None
                if msg is not None:
                    kind = msg.get("type")
                    if kind == "_closed" or (kind == "bye" and msg.get("request_id") is None):
                        why = msg.get("reason") or "process ended"
                        self._discard(f"worker ended during a job ({why})", error=True)
                        raise WorkerError(f"the worker exited during the job ({why}); run the JoyAI Runtime node again")
                    if msg.get("request_id") == rid:
                        if kind == "progress" and on_progress is not None:
                            on_progress(min(int(msg.get("frames_done", 0)), n), n)
                        elif kind == "result":
                            return
                        elif kind == "cancelled":
                            raise JobCancelled("the job was cancelled")
                        elif kind == "error":
                            if msg.get("fatal"):
                                self._discard(f"fatal worker error: {msg.get('error_type')}", error=True)
                            raise WorkerError(f"JoyAI job failed: {msg.get('error_type')}: {redact(msg.get('error'))}")
                if msg is None and not w.alive():
                    self._discard("worker process ended during a job", error=True)
                    raise WorkerError(f"the worker exited during the job: {w.log_tail()}")
                if interrupted():
                    self._cancel(w, rid, job_dir)
                    raise JobCancelled("cancelled by the user")
                if time.monotonic() > deadline:
                    self._cancel(w, rid, job_dir)
                    raise JobTimeout(f"the {n}-frame job exceeded its time limit (job_timeout_base_s + job_timeout_per_frame_s x frames in the runtime config)")
                if time.monotonic() >= next_mem:
                    next_mem = time.monotonic() + 1.0
                    stop = watch.update(memory.snapshot())
                    if watch.warned and not warned:
                        warned = True
                        self._note("memory_warning", commit_pct=round(watch.peak_pct, 1), warn_pct=w.rt.memory_guard["warn_commit_pct"])
                    if stop:
                        self._note("memory_stop", reason=watch.reason)
                        self._cancel(w, rid, job_dir)
                        raise AdmissionRefused(f"memory guard stopped the job: {watch.reason}")
        except (JobCancelled, JobTimeout, WorkerError, AdmissionRefused):
            raise
        except BaseException:
            # e.g. ComfyUI's progress hook raising its interrupt exception from on_progress
            if self._worker is w and w.alive():
                self._cancel(w, rid, job_dir)
            raise

    def _cancel(self, w: WorkerProcess, rid: str, job_dir: Path) -> None:
        """Cooperative cancel; if the worker does not confirm within cancel_grace_s it is stopped."""
        self._set(CANCELLING)
        try:
            (job_dir / P.CANCEL_FILE).write_text("cancel\n", encoding="utf-8")
        except OSError:
            pass
        try:
            w.send("cancel", target=rid)
        except WorkerError:
            self._discard("pipe closed during cancel", error=True)
            return
        deadline = time.monotonic() + w.rt.cancel_grace_s
        while time.monotonic() < deadline:
            try:
                msg = w.messages.get(timeout=0.2)
            except queue.Empty:
                if not w.alive():
                    break
                continue
            if msg.get("request_id") == rid and msg.get("type") in ("cancelled", "result", "error"):
                self._note(
                    "job_cancelled",
                    clean=msg.get("type") == "cancelled",
                    frames_done=msg.get("frames_done"),
                    worker_seconds=msg.get("seconds"),
                    cuda_allocated=msg.get("cuda_allocated_bytes"),
                )
                if msg.get("type") == "error" and msg.get("fatal"):
                    self._discard("fatal error during cancel", error=True)
                else:
                    self._set(READY)
                return
            if msg.get("type") == "_closed":
                break
        self._discard("cancel not confirmed in time", error=True)

    # -- status / unload --------------------------------------------------------------------------------
    def status_dict(self, *, live: bool = True) -> dict:
        with self._lock:
            w = self._worker
            out: dict = {"state": self.state, "generation": self.generation, "last_error": self.last_error, "events": list(self.events)[-12:]}
            if self.release:
                out["release"] = {k: v for k, v in self.release.items() if k != "started"}
            if w is None:
                out["worker"] = None
                return out
            ready = w.ready or {}
            out["worker"] = {
                "worker_id": w.worker_id,
                "runtime_id": w.rt.id,
                "alive": w.alive(),
                "jobs_via_this_manager": w.jobs,
                "started_s_ago": round(time.time() - w.started_at, 1),
                "pid": (w.hello or {}).get("pid"),
                "load_seconds": ready.get("load_seconds"),
                "attention_backend": ready.get("attention_backend"),
                "fp8_blocks": ready.get("fp8_blocks"),
                "text_encoder_resident": ready.get("text_encoder_resident"),
                "cuda_allocated_bytes": ready.get("cuda_allocated_bytes"),
                "versions": ready.get("versions"),
            }
            if live and w.alive() and self.state == READY:  # never while a job runs; never starts a worker
                try:
                    rid = w.send("status")
                    end = time.monotonic() + STATUS_TIMEOUT_S
                    while time.monotonic() < end:
                        try:
                            msg = w.messages.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        if msg.get("request_id") == rid and msg.get("type") == "status":
                            out["worker"]["live"] = {k: v for k, v in msg.items() if k not in ("v", "type", "request_id", "worker_id")}
                            break
                        if msg.get("type") == "_closed":
                            w.messages.put(msg)
                            break
                except WorkerError as exc:
                    out["worker"]["status_error"] = str(exc)
            return out

    def unload(self, reason: str = "unload requested") -> dict:
        with self._lock:
            w = self._worker
            if w is None:
                return {"unloaded": False, "reason": "no worker running", "state": self.state}
            report = self._discard(reason)
            return {"unloaded": True, "worker_id": w.worker_id, "state": self.state, "stop": _safe_report(report)}


def _safe_report(report: dict | None) -> dict | None:
    if report is None:
        return None
    return json.loads(redact(json.dumps(report)))


MANAGER = Manager()


@atexit.register
def _shutdown_on_exit() -> None:  # ComfyUI exits normally (Ctrl+C): do not leave a loaded worker behind
    try:
        MANAGER.unload("ComfyUI is exiting")
    except Exception:  # noqa: BLE001
        pass
