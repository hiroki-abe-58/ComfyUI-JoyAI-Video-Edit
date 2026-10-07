"""Wire format shared by the ComfyUI side (joyai_comfy) and the runtime worker (stdlib only, one source of truth).

Control: one JSON object per line (UTF-8, at most MAX_LINE bytes) over the worker's stdin / a private copy of its
stdout. Bulk data: raw RGB uint8 files in a job folder (C-order [N, H, W, 3]), described by job.json / result.json.

Requests (ComfyUI -> worker), all with {"v": 1, "type": ..., "request_id": ID}:
  edit      {"job": "<runtime path of job.json>"}
  cancel    {"target": ID}
  status    {}
  shutdown  {}
Replies (worker -> ComfyUI), all with {"v": 1, "type": ..., "worker_id": ..., "request_id": ID | null}:
  hello     {pid, pgid, starttime}                       right after start, before the model load
  ready     {identity, load_seconds, attention_backend, fp8_blocks, ...}   model loaded
  accepted  {job_id}                                      the job will not be re-sent from here on
  progress  {frames_done, frames_total, chunks_done}
  result    {job_id}                                      result.json + output.u8 are complete
  cancelled {job_id, cuda_allocated_bytes, ...}
  error     {error_type, error, fatal}
  status    {...}
  bye       {reason, jobs_done}
"""

from __future__ import annotations

import re

PROTOCOL_VERSION = 1
MAX_LINE = 64 * 1024
MAX_REPLY = 1024 * 1024
MAX_JSON_FILE = 256 * 1024
ID_RE = re.compile(r"[A-Za-z0-9_-]{8,80}")

REQUESTS = ("edit", "cancel", "status", "shutdown")
REPLIES = ("hello", "ready", "accepted", "progress", "result", "cancelled", "error", "status", "bye")

# Limits of v0 (verified range in parentheses; see docs/LIMITS.md)
MAX_FRAMES = 600  # upstream server starts a new session every 600 frames (kv_reset_frames); refused above
MIN_SIDE = 64
MAX_SIDE = 4096
MAX_PROMPT_CHARS = 4000
STEPS_RANGE = (1, 8)  # (2 verified)
TEMPORAL_IDS_RANGE = (1, 64)  # (8 verified)
SEED_RANGE = (0, 2**31 - 1)
ORIENTATIONS = ("auto", "landscape", "portrait")

JOB_FILE = "job.json"
INPUT_FILE = "input.u8"
REF_FILE = "reference.u8"
OUTPUT_FILE = "output.u8"
RESULT_FILE = "result.json"
CANCEL_FILE = "CANCEL"

JOB_KEYS = {
    "schema_version",
    "job_id",
    "frames",
    "height",
    "width",
    "input_sha256",
    "prompt",
    "seed",
    "num_inference_steps",
    "max_temporal_ids",
    "orientation",
    "reference",
}
REF_KEYS = {"height", "width", "sha256"}


class ProtocolError(ValueError):
    pass


class JobCancelled(Exception):
    """Raised inside the worker when a job is cancelled (one class shared by the worker and the engines)."""


def u8_nbytes(n: int, h: int, w: int) -> int:
    return int(n) * int(h) * int(w) * 3


def _int(d: dict, key: str, lo: int, hi: int) -> int:
    v = d.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ProtocolError(f"{key} must be an integer in {lo}..{hi}")
    return v


def _sha(d: dict, key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v):
        raise ProtocolError(f"{key} must be a lowercase sha256 hex digest")
    return v


def check_prompt(prompt) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProtocolError("prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ProtocolError(f"prompt is longer than {MAX_PROMPT_CHARS} characters")
    if "\0" in prompt:
        raise ProtocolError("prompt must not contain NUL characters")
    return prompt


def validate_job(job) -> dict:
    """Schema check of job.json (both sides). Raises ProtocolError."""
    if not isinstance(job, dict) or job.get("schema_version") != 1:
        raise ProtocolError("unsupported job.json")
    extra = set(job) - JOB_KEYS
    if extra:
        raise ProtocolError(f"unknown job keys {sorted(extra)}")
    if not isinstance(job.get("job_id"), str) or not ID_RE.fullmatch(job["job_id"]):
        raise ProtocolError("invalid job_id")
    _int(job, "frames", 1, MAX_FRAMES)
    _int(job, "height", MIN_SIDE, MAX_SIDE)
    _int(job, "width", MIN_SIDE, MAX_SIDE)
    _sha(job, "input_sha256")
    check_prompt(job.get("prompt"))
    _int(job, "seed", *SEED_RANGE)
    _int(job, "num_inference_steps", *STEPS_RANGE)
    _int(job, "max_temporal_ids", *TEMPORAL_IDS_RANGE)
    if job.get("orientation") not in ORIENTATIONS:
        raise ProtocolError(f"orientation must be one of {ORIENTATIONS}")
    ref = job.get("reference")
    if ref is not None:
        if not isinstance(ref, dict) or set(ref) != REF_KEYS:
            raise ProtocolError("reference must be {height, width, sha256}")
        _int(ref, "height", MIN_SIDE, MAX_SIDE)
        _int(ref, "width", MIN_SIDE, MAX_SIDE)
        _sha(ref, "sha256")
    return job


def validate_result(result, job: dict) -> dict:
    """Check result.json against its job (ComfyUI side). Raises ProtocolError."""
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ProtocolError("unsupported result.json")
    if result.get("job_id") != job["job_id"]:
        raise ProtocolError("result.json does not belong to this job")
    if result.get("frames") != job["frames"]:
        raise ProtocolError(f"result has {result.get('frames')} frames, the job sent {job['frames']}")
    _int(result, "height", MIN_SIDE, MAX_SIDE)
    _int(result, "width", MIN_SIDE, MAX_SIDE)
    _sha(result, "output_sha256")
    if result.get("output_bytes") != u8_nbytes(result["frames"], result["height"], result["width"]):
        raise ProtocolError("result output_bytes does not match frames x height x width x 3")
    return result
