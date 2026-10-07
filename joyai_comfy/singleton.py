"""Machine-wide exclusive lock: at most one JoyAI model per WSL distro on this machine, across ComfyUI processes.

Windows: an exclusive handle on a lock file (CreateFileW with share mode 0). The kernel refuses a second open with
ERROR_SHARING_VIOLATION, so two processes racing for it cannot both win; the handle belongs to the process (not to a
thread, unlike a named mutex - this lock is released from other threads: the release watcher, atexit, the idle timer)
and the OS closes it when the process exits or crashes. POSIX hosts (tests): fcntl.flock(LOCK_EX | LOCK_NB).
The holder writes a small owner record (pid, process start time) next to the lock so a refused process can say who
holds it. The lock file itself carries no data and is never deleted while held.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

IS_WINDOWS = os.name == "nt"


class RuntimeInUse(RuntimeError):
    """Another ComfyUI process holds the JoyAI runtime of this distro."""


def lock_dir() -> Path:
    explicit = os.environ.get("JOYAI_LOCK_DIR")
    if explicit:
        d = Path(explicit)
    elif IS_WINDOWS:
        d = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "ComfyUI-JoyAI-Video-Edit" / "locks"
    else:
        d = Path(os.environ.get("XDG_RUNTIME_DIR") or Path.home() / ".cache") / "comfyui-joyai-video-edit" / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def process_identity(pid: int | None = None) -> dict:
    """{pid, create_time} of a process (create_time None if unknown)."""
    pid = os.getpid() if pid is None else pid
    try:
        import psutil

        return {"pid": pid, "create_time": round(psutil.Process(pid).create_time(), 3)}
    except Exception:  # noqa: BLE001
        return {"pid": pid, "create_time": None}


def owner_alive(owner) -> bool | None:
    """True / False if it can be decided, None if unknown (callers must then treat the owner as alive)."""
    if not isinstance(owner, dict) or not isinstance(owner.get("pid"), int):
        return None
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return None
    try:
        p = psutil.Process(owner["pid"])
    except psutil.NoSuchProcess:
        return False
    except Exception:  # noqa: BLE001
        return None
    ct = owner.get("create_time")
    if ct is None:
        return True
    try:
        return abs(p.create_time() - ct) < 1.0  # same pid, other start time = pid reused = original owner is gone
    except Exception:  # noqa: BLE001
        return None


if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _GENERIC_READ, _GENERIC_WRITE = 0x80000000, 0x40000000
    _OPEN_ALWAYS, _FILE_ATTRIBUTE_NORMAL = 4, 0x80
    _ERROR_SHARING_VIOLATION = 32
    _INVALID = ctypes.c_void_p(-1).value


class MachineLock:
    def __init__(self, name: str):
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", name):
            raise ValueError(f"bad lock name {name!r}")
        self.name = name
        self.path = lock_dir() / f"{name}.lock"
        self.owner_path = lock_dir() / f"{name}.owner.json"
        self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self, info: dict | None = None) -> bool:
        """Non-blocking. True if this process now holds the lock (or already did)."""
        if self._handle is not None:
            return True
        if IS_WINDOWS:
            h = _k32.CreateFileW(str(self.path), _GENERIC_READ | _GENERIC_WRITE, 0, None, _OPEN_ALWAYS, _FILE_ATTRIBUTE_NORMAL, None)
            if not h or h == _INVALID:
                err = ctypes.get_last_error()
                if err == _ERROR_SHARING_VIOLATION:
                    return False
                raise OSError(err, f"cannot open the JoyAI lock file {self.path.name}")
            self._handle = h
        else:
            import fcntl

            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                return False
            self._handle = fd
        rec = {**process_identity(), "since": round(time.time(), 1), **(info or {})}
        tmp = self.owner_path.with_name(self.owner_path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp, self.owner_path)
        return True

    def owner(self) -> dict | None:
        try:
            return json.loads(self.owner_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            rec = self.owner()
            if rec and rec.get("pid") == os.getpid():
                self.owner_path.unlink(missing_ok=True)
        except OSError:
            pass
        if IS_WINDOWS:
            _k32.CloseHandle(self._handle)
        else:
            import fcntl

            try:
                fcntl.flock(self._handle, fcntl.LOCK_UN)
            finally:
                os.close(self._handle)
        self._handle = None
