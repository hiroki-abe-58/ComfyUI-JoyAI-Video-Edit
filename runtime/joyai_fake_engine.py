"""Model-free stand-in for the JoyAI engine (tests and CI only; selected by `"engine": "fake"` in the admin config).

It follows the real engine's contract - output size = the session size for the job's orientation, one output frame
per input frame, progress per 8-frame chunk, cooperative cancel between frames - with a cheap deterministic image
function of (input frame, prompt, seed). Test hooks in the prompt: "[fake:slow]" (0.05 s per frame), "[fake:crash]"
(the process exits mid-job), "[fake:fatal]" (a fatal error reply), "[fake:hang]" (ignores cancel for 30 s); a slow
model load is simulated with the env JOYAI_FAKE_LOAD_SECONDS (passed only to 'posix' test runtimes).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np


class FatalEngineError(RuntimeError):
    fatal = True


class FakeEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        delay = float(os.environ.get("JOYAI_FAKE_LOAD_SECONDS", "0") or 0)
        time.sleep(min(delay, 600.0))
        self.jobs = 0

    def ready_info(self) -> dict:
        return {
            "attention_backend": "fake",
            "fp8_blocks": [0, 0],
            "text_encoder_resident": False,
            "cuda_allocated_bytes": 0,
            "versions": {"engine": "fake", "numpy": np.__version__},
        }

    def status(self) -> dict:
        return {"engine": "fake", "jobs": self.jobs}

    def memory_info(self) -> dict:
        return {"cuda_allocated_bytes": 0}

    def run(self, job: dict, frames: np.ndarray, ref, *, should_cancel, on_progress):
        from joyai_protocol import JobCancelled

        prompt = job["prompt"]
        orientation = job["orientation"]
        if orientation == "auto":
            orientation = "portrait" if job["height"] > job["width"] else "landscape"
        long_, short = self.cfg["width"], self.cfg["height"]
        oh, ow = (long_, short) if orientation == "portrait" else (short, long_)
        n = frames.shape[0]
        key = int.from_bytes(hashlib.sha256(f"{prompt}\0{job['seed']}".encode()).digest()[:2], "little")
        ys = (np.arange(oh) * frames.shape[1] // oh).astype(np.int64)
        xs = (np.arange(ow) * frames.shape[2] // ow).astype(np.int64)
        out = np.empty((n, oh, ow, 3), dtype=np.uint8)
        for i in range(n):
            if "[fake:hang]" in prompt and i == 4:
                time.sleep(30)
            if should_cancel():
                raise JobCancelled()
            if "[fake:crash]" in prompt and i == 4:
                os._exit(17)
            if "[fake:fatal]" in prompt and i == 4:
                raise FatalEngineError("simulated CUDA failure")
            if "[fake:slow]" in prompt:
                time.sleep(0.05)
            src = frames[i][ys][:, xs]
            out[i] = (src[..., ::-1].astype(np.uint16) + key) % 256  # channel swap + prompt/seed dependent shift
            if (i + 1) % 8 == 0 or i + 1 == n:
                on_progress(i + 1, (i + 8) // 8)
        self.jobs += 1
        return out, {"engine": "fake", "session_size": [oh, ow], "orientation": orientation}
