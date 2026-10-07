"""Job folders: created by the manager only, confined to the jobs root, fixed file names, atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from pathlib import Path

import numpy as np

from . import protocol as P
from .config import Runtime


class JobFilesError(RuntimeError):
    pass


OWNER_FILE = "owner.json"


def default_jobs_root() -> Path:
    """Per-user folder outside ComfyUI: ComfyUI deletes its own temp folder at start-up and exit, which another ComfyUI
    process of the same installation would otherwise do under a running worker."""
    from .singleton import lock_dir

    return lock_dir().parent / "jobs"


def jobs_root(rt: Runtime) -> Path:
    root = Path(rt.jobs_dir) if rt.jobs_dir else default_jobs_root()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _is_link(path: Path) -> bool:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def new_dir(rt: Runtime, kind: str) -> tuple[str, Path]:
    """A fresh, unique folder directly under the jobs root (never reused, never a link)."""
    ident = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:12]}"
    if not P.ID_RE.fullmatch(ident):
        raise JobFilesError("bad job id")
    root = jobs_root(rt)
    path = (root / ident).resolve()
    if path.parent != root:
        raise JobFilesError("job folder escaped the jobs root")
    path.mkdir(parents=False, exist_ok=False)
    if _is_link(path):
        raise JobFilesError("job folder is a link")
    from .singleton import process_identity

    write_json(path / OWNER_FILE, process_identity())  # cleanup of folders left by a dead ComfyUI process checks this
    return ident, path


def child(job_dir: Path, name: str) -> Path:
    if name not in (P.JOB_FILE, P.INPUT_FILE, P.OUTPUT_FILE, P.RESULT_FILE, P.CANCEL_FILE, OWNER_FILE):
        raise JobFilesError(f"unexpected job file name {name!r}")
    return job_dir / name


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def write_u8(path: Path, frames: np.ndarray) -> str:
    """Write C-order uint8 bytes atomically; return their sha256."""
    if frames.dtype != np.uint8 or not frames.flags["C_CONTIGUOUS"]:
        raise JobFilesError("frames must be C-contiguous uint8")
    digest = hashlib.sha256(memoryview(frames).cast("B")).hexdigest()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(memoryview(frames).cast("B"))
    os.replace(tmp, path)
    return digest


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_result(job_dir: Path, job: dict) -> tuple[dict, np.ndarray]:
    """Validate result.json + output.u8 of this job and return (result, frames uint8 [N, H, W, 3])."""
    rpath = child(job_dir, P.RESULT_FILE)
    if not rpath.is_file() or _is_link(rpath) or rpath.stat().st_size > P.MAX_JSON_FILE:
        raise JobFilesError("the runtime finished without a valid result.json")
    try:
        result = P.validate_result(json.loads(rpath.read_text(encoding="utf-8")), job)
    except (UnicodeDecodeError, json.JSONDecodeError, P.ProtocolError) as exc:
        raise JobFilesError(f"invalid result.json: {exc}") from None
    opath = child(job_dir, P.OUTPUT_FILE)
    if not opath.is_file() or _is_link(opath):
        raise JobFilesError("output.u8 is missing")
    if opath.stat().st_size != result["output_bytes"]:
        raise JobFilesError(f"output.u8 has {opath.stat().st_size} bytes, result.json says {result['output_bytes']}")
    if _sha256_file(opath) != result["output_sha256"]:
        raise JobFilesError("output.u8 does not match its recorded sha256")
    frames = np.fromfile(opath, dtype=np.uint8).reshape(result["frames"], result["height"], result["width"], 3)
    return result, frames


def remove(rt: Runtime, path: Path) -> None:
    """Delete a job / session folder created by new_dir (and nothing else)."""
    root = jobs_root(rt)
    p = Path(path).resolve()
    if p.parent != root or not P.ID_RE.fullmatch(p.name) or not p.is_dir() or _is_link(p):
        return
    shutil.rmtree(p, ignore_errors=True)
