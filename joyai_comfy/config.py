"""Administrator-registered JoyAI runtimes.

A workflow can only *name* a runtime (``runtime`` combo). Everything that is executed or read - the WSL
distribution and user, the runtime's Python, the upstream deploy folder, the checkpoints - comes from a local JSON
file that the ComfyUI API cannot write to:

1. the path in the ``COMFYUI_JOYAI_CONFIG`` environment variable, else
2. ``<ComfyUI user directory>/joyai.runtimes.json`` (root of the user directory, outside the per-user folders that
   the userdata API serves).

See ``examples/joyai.runtimes.example.json`` and docs/SETUP.md.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

CONFIG_ENV = "COMFYUI_JOYAI_CONFIG"
CONFIG_FILENAME = "joyai.runtimes.json"
SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

RUNTIME_ENV_KEYS = frozenset({"CUDA_VISIBLE_DEVICES"})
ENGINES = ("joyai", "fake")  # "fake" = model-free test engine (CI / protocol tests only)
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MEMMAX_RE = re.compile(r"[1-9][0-9]{0,4}[MG]")
GUARD_DEFAULTS = {
    "enabled": True,
    "max_commit_pct": 90.0,  # no new large-memory stage (model load or job) if commit now / projected exceeds this share
    "expected_worker_gib": 65.0,  # commit headroom required for a model load (measured start delta ~62 GiB, 2026-10-07)
    "margin_gib": 0.0,  # extra operating margin on top of the headroom
    "job_factor": 1.25,  # admission before a job: 1.25 x the per-job estimate
    "min_available_gib": 24.0,  # physical memory available before a worker start (observed min 20.45 while resident)
    "warn_commit_pct": 85.0,  # warning (logged, shown in status) at or above this share
    "hard_stop_commit_pct": 95.0,  # a running job is stopped as soon as commit reaches this share
    "release_timeout_s": 120.0,  # after a worker stops: wait at most this long for the WSL VM to return its memory
    "release_tolerance_gib": 4.0,  # released = commit back within this of the pre-load commit
}
# Bounds the administrator may not cross: the guard can be made stricter, never looser.
GUARD_BOUNDS = {
    "max_commit_pct": ("max", 90.0),
    "hard_stop_commit_pct": ("max", 95.0),
    "warn_commit_pct": ("max", 85.0),
    "expected_worker_gib": ("min", 65.0),
    "release_timeout_s": ("max", 600.0),
}
_KEYS = {
    "kind",
    "distro",
    "user",
    "python",
    "deploy_dir",
    "dit_ckpt",
    "vae_ckpt",
    "text_encoder_ckpt",
    "cache_dir",
    "engine",
    "memory_max",
    "jobs_dir",
    "wsl_mount_root",
    "description",
    "width",
    "height",
    "max_frames",
    "hello_timeout_s",
    "load_timeout_s",
    "job_timeout_base_s",
    "job_timeout_per_frame_s",
    "cancel_grace_s",
    "keep_job_dirs",
    "require_attention",
    "env",
    "memory_guard",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Runtime:
    id: str
    kind: str  # "wsl" (Windows host -> WSL2 distro) or "posix" (same Linux host; tests)
    python: str
    deploy_dir: str
    dit_ckpt: str
    vae_ckpt: str
    text_encoder_ckpt: str
    cache_dir: str
    distro: str | None = None
    user: str | None = None
    engine: str = "joyai"
    memory_max: str | None = None  # optional systemd-run --scope MemoryMax inside the distro
    jobs_dir: str | None = None  # host path; default <ComfyUI temp>/joyai
    wsl_mount_root: str = "/mnt/"
    description: str = ""
    width: int = 840  # session size of the RTX 5090 profile (long side x short side; orientation per job)
    height: int = 480
    max_frames: int = 600
    hello_timeout_s: int = 60
    load_timeout_s: int = 1800
    job_timeout_base_s: int = 120
    job_timeout_per_frame_s: float = 2.0
    cancel_grace_s: int = 60
    keep_job_dirs: bool = False  # test flag: keep job folders (parity taps); never a workflow input
    require_attention: str = "sage"
    env: dict = field(default_factory=dict)
    memory_guard: dict = field(default_factory=lambda: dict(GUARD_DEFAULTS))


def _linux_abs_path(value, what: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or any(c in value for c in "\0\n\r") or len(value) > 1024:
        raise ConfigError(f"{what} must be an absolute Linux path")
    if ".." in PurePosixPath(value).parts:
        raise ConfigError(f"{what} must not contain '..'")
    return value


def _python_path(kind: str, value, rid: str) -> str:
    if kind == "wsl":
        return _linux_abs_path(value, f"runtime {rid}: python")
    # 'posix' = the runtime runs on this host (CI / tests): an absolute path on this machine
    if not isinstance(value, str) or not os.path.isabs(value) or any(c in value for c in "\0\n\r") or ".." in Path(value).parts:
        raise ConfigError(f"runtime {rid}: python must be an absolute path")
    return value


def _int(raw: dict, key: str, default: int, lo: int, hi: int, rid: str) -> int:
    v = raw.get(key, default)
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ConfigError(f"runtime {rid}: {key} must be an integer in {lo}..{hi}")
    return v


def _guard(raw, rid: str) -> dict:
    if raw is None:
        return dict(GUARD_DEFAULTS)
    if not isinstance(raw, dict) or set(raw) - set(GUARD_DEFAULTS):
        raise ConfigError(f"runtime {rid}: memory_guard may only set {sorted(GUARD_DEFAULTS)}")
    out = dict(GUARD_DEFAULTS)
    for k, v in raw.items():
        if k == "enabled":
            if not isinstance(v, bool):
                raise ConfigError(f"runtime {rid}: memory_guard.enabled must be true/false")
        elif not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            raise ConfigError(f"runtime {rid}: memory_guard.{k} must be a non-negative number")
        out[k] = v
    for k, (kind, bound) in GUARD_BOUNDS.items():
        if (kind == "max" and out[k] > bound) or (kind == "min" and out[k] < bound):
            raise ConfigError(f"runtime {rid}: memory_guard.{k} may not be {'above' if kind == 'max' else 'below'} {bound}")
    return out


def parse_runtime(rid: str, raw) -> Runtime:
    if not isinstance(rid, str) or not _ID_RE.fullmatch(rid):
        raise ConfigError(f"invalid runtime id {rid!r}")
    if not isinstance(raw, dict):
        raise ConfigError(f"runtime {rid}: expected an object")
    extra = set(raw) - _KEYS
    if extra:
        raise ConfigError(f"runtime {rid}: unknown keys {sorted(extra)}")
    kind = raw.get("kind")
    if kind not in ("wsl", "posix"):
        raise ConfigError(f"runtime {rid}: kind must be 'wsl' or 'posix'")
    distro, user = raw.get("distro"), raw.get("user")
    if kind == "wsl":
        if not isinstance(distro, str) or not _ID_RE.fullmatch(distro):
            raise ConfigError(f"runtime {rid}: distro must be a WSL distribution name")
    elif distro is not None:
        raise ConfigError(f"runtime {rid}: distro is only valid for kind 'wsl'")
    if user is not None and (kind != "wsl" or not isinstance(user, str) or not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", user)):
        raise ConfigError(f"runtime {rid}: user must be a Linux user name (kind 'wsl' only)")
    engine = raw.get("engine", "joyai")
    if engine not in ENGINES:
        raise ConfigError(f"runtime {rid}: engine must be one of {ENGINES}")
    memory_max = raw.get("memory_max")
    if memory_max is not None and (not isinstance(memory_max, str) or not _MEMMAX_RE.fullmatch(memory_max)):
        raise ConfigError(f"runtime {rid}: memory_max must look like '64G' or '512M'")
    env = raw.get("env", {})
    if not isinstance(env, dict) or set(env) - RUNTIME_ENV_KEYS or not all(isinstance(v, str) and re.fullmatch(r"[0-9,]{0,32}", v) for v in env.values()):
        raise ConfigError(f"runtime {rid}: env may only set {sorted(RUNTIME_ENV_KEYS)} (digits and commas)")
    jobs_dir = raw.get("jobs_dir")
    if jobs_dir is not None and (not isinstance(jobs_dir, str) or not (Path(jobs_dir).is_absolute() or PureWindowsPath(jobs_dir).is_absolute())):
        raise ConfigError(f"runtime {rid}: jobs_dir must be an absolute path on this machine")
    mount_root = _linux_abs_path(raw.get("wsl_mount_root", "/mnt/"), f"runtime {rid}: wsl_mount_root")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ConfigError(f"runtime {rid}: description must be a string")
    keep = raw.get("keep_job_dirs", False)
    if not isinstance(keep, bool):
        raise ConfigError(f"runtime {rid}: keep_job_dirs must be true/false")
    attn = raw.get("require_attention", "sage")
    if attn not in ("sage", "sdpa", "any"):
        raise ConfigError(f"runtime {rid}: require_attention must be 'sage', 'sdpa' or 'any'")
    per_frame = raw.get("job_timeout_per_frame_s", 2.0)
    if not isinstance(per_frame, (int, float)) or isinstance(per_frame, bool) or not 0 < per_frame <= 600:
        raise ConfigError(f"runtime {rid}: job_timeout_per_frame_s must be a number in (0, 600]")
    width = _int(raw, "width", 840, 64, 4096, rid)
    height = _int(raw, "height", 480, 64, 4096, rid)
    guard = _guard(raw.get("memory_guard"), rid)
    if kind == "wsl" and not guard["enabled"]:
        raise ConfigError(f"runtime {rid}: the memory guard is mandatory for WSL runtimes (memory_guard.enabled cannot be false)")
    return Runtime(
        id=rid,
        kind=kind,
        distro=distro,
        user=user,
        python=_python_path(kind, raw.get("python"), rid),
        deploy_dir=_linux_abs_path(raw.get("deploy_dir"), f"runtime {rid}: deploy_dir"),
        dit_ckpt=_linux_abs_path(raw.get("dit_ckpt"), f"runtime {rid}: dit_ckpt"),
        vae_ckpt=_linux_abs_path(raw.get("vae_ckpt"), f"runtime {rid}: vae_ckpt"),
        text_encoder_ckpt=_linux_abs_path(raw.get("text_encoder_ckpt"), f"runtime {rid}: text_encoder_ckpt"),
        cache_dir=_linux_abs_path(raw.get("cache_dir"), f"runtime {rid}: cache_dir"),
        engine=engine,
        memory_max=memory_max,
        jobs_dir=jobs_dir,
        wsl_mount_root=mount_root if mount_root.endswith("/") else mount_root + "/",
        description=description[:300],
        width=max(width, height),
        height=min(width, height),
        max_frames=_int(raw, "max_frames", 600, 1, 600, rid),
        hello_timeout_s=_int(raw, "hello_timeout_s", 60, 5, 3600, rid),
        load_timeout_s=_int(raw, "load_timeout_s", 1800, 10, 4 * 3600, rid),
        job_timeout_base_s=_int(raw, "job_timeout_base_s", 120, 1, 4 * 3600, rid),
        job_timeout_per_frame_s=float(per_frame),
        cancel_grace_s=_int(raw, "cancel_grace_s", 60, 1, 3600, rid),
        keep_job_dirs=keep,
        require_attention=attn,
        env=dict(env),
        memory_guard=guard,
    )


def config_path() -> Path | None:
    explicit = os.environ.get(CONFIG_ENV)
    if explicit:
        return Path(explicit)
    try:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / CONFIG_FILENAME
    except Exception:
        return None


def load_runtimes(path: Path | None = None) -> dict[str, Runtime]:
    """Return the configured runtimes. A missing file means 'none configured'; a broken file raises."""
    path = path if path is not None else config_path()
    if path is None or not path.is_file():
        return {}
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ConfigError(f"{path.name} is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path.name}: not valid UTF-8 JSON ({exc})") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("runtimes"), dict):
        raise ConfigError(f"{path.name}: expected {{'schema_version': {SCHEMA_VERSION}, 'runtimes': {{...}}}}")
    if set(data) - {"schema_version", "runtimes"}:
        raise ConfigError(f"{path.name}: unknown top-level keys {sorted(set(data) - {'schema_version', 'runtimes'})}")
    return {rid: parse_runtime(rid, raw) for rid, raw in data["runtimes"].items()}


def host_to_runtime_path(rt: Runtime, path) -> str:
    """Translate a host path (job folder, package script) into the path the runtime sees."""
    s = str(path)
    if rt.kind == "posix":
        return s
    m = re.fullmatch(r"([A-Za-z]):[\\/](.*)", s)
    if not m or s.startswith("\\\\"):
        raise ConfigError(f"WSL runtimes need local drive paths (X:\\...), got {s!r}")
    rest = m.group(2).replace("\\", "/").strip("/")
    if ".." in rest.split("/"):
        raise ConfigError("path must not contain '..'")
    return f"{rt.wsl_mount_root}{m.group(1).lower()}/{rest}" if rest else f"{rt.wsl_mount_root}{m.group(1).lower()}"
