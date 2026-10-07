"""Real JoyAI engine inside the worker: upstream JoyAI-Video-Edit's in-process streaming API, driven like the upstream
server's file mode, with the PATCHED_REFERENCE weight placement.

What runs (nothing of upstream is copied; it is imported from the runtime's own clone at ``deploy_dir``):
- environment = the upstream RTX 5090 profile (DEPLOYMENT.md / run_server.sh): FP8 img + txt, FP8 fast accumulation,
  SageAttention, low-VRAM placement, CUDA graphs; prompt enhancement disabled; offline;
- weight placement = joyai_lowram (load_dit_streamed / load_text_encoder_lowram with the resident pinned text-encoder
  cache), installed before the serving module is imported - exactly as the reference launcher did;
- ``JoyOmniRuntime.load`` with the server's arguments (all devices cuda:0, seed 42, warmup at the session size);
- per job: a new ``create_v2v_session`` with the server's file-mode settings (lossless output, freeze_kv_on_static
  off, the server's size snapping and orientation rule, 2-chunk in-flight limit), frames pushed in order with the
  server's frame_meta, ``flush_pending``, results collected with ``wait_async_result`` until every frame is back,
  then ``request_stop`` + ``close``. The returned frames are the RGB uint8 arrays the session produces
  (``_pack_decoded_pixels`` in lossless mode) - no JPEG / H.264 anywhere.
The global CUDA RNG is not reseeded per job (upstream semantics: the session generator is seeded from the job's seed;
VAE latent sampling uses the global RNG).
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

PROFILE_ENV = {
    "JOYOMNI_FP8_IMG": "1",
    "JOYOMNI_FP8_TXT": "1",
    "JOYOMNI_CUDA_GRAPH": "1",
    "JOYOMNI_SAGE_ATTN": "1",
    "JOYOMNI_FP8_FAST_ACCUM": "1",
    "JOYOMNI_LOW_VRAM": "1",
    "JOYAI_TE_RESIDENT": "1",
    "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONUNBUFFERED": "1",
}
DEVICE = "cuda:0"
FPS = 24  # frame_meta media time, as the reference client sent it (not used by the model)
MAX_INFLIGHT_CHUNKS = 2  # server default for file input: wait for capacity
CAPACITY_TIMEOUT_S = 120.0
DRAIN_TIMEOUT_S = 120.0
CLOSE_TIMEOUT_S = 10.0
STATIC_DIFF_THRESH = 0.5  # server default (unused with freeze_kv_on_static off)


class EngineError(RuntimeError):
    pass


class FatalEngineError(RuntimeError):
    """The CUDA context or the runtime may be unusable: the manager replaces the worker."""

    fatal = True


def _snap_to_align(value: int, align: int) -> int:  # the upstream server's rule
    return max(align, (value + align // 2) // align * align)


def check_checkpoints(cfg: dict) -> None:
    """Refuse before any import or GPU work when a checkpoint is missing (nothing is ever downloaded)."""
    need = {
        "dit_ckpt": (cfg["dit_ckpt"], os.path.isfile),
        "vae_ckpt": (cfg["vae_ckpt"], os.path.exists),
        "text_encoder_ckpt": (os.path.join(cfg["text_encoder_ckpt"], "config.json"), os.path.isfile),
        "text_encoder_ckpt index": (os.path.join(cfg["text_encoder_ckpt"], "model.safetensors.index.json"), os.path.isfile),
    }
    missing = [
        f"{key} ({os.path.basename(os.path.dirname(p)) if key.startswith('text_encoder') else os.path.basename(p)})"
        for key, (p, ok) in need.items()
        if not ok(p)
    ]
    if missing:
        raise EngineError(
            "checkpoint missing in the runtime: "
            + ", ".join(missing)
            + " - install the weights as described in docs/SETUP.md; nothing is downloaded at run time"
        )


class Engine:
    def __init__(self, cfg: dict):
        check_checkpoints(cfg)
        self.cfg = cfg
        cache = cfg["cache_dir"].rstrip("/")
        os.environ.update(PROFILE_ENV)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{cache}/torchinductor"
        os.environ["TRITON_CACHE_DIR"] = f"{cache}/triton"
        os.environ["CUDA_CACHE_PATH"] = f"{cache}/nv_compute"
        bindir = os.path.dirname(cfg.get("python") or sys.executable)
        os.environ["PATH"] = f"{bindir}:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"
        for d in ("torchinductor", "triton", "nv_compute"):
            os.makedirs(f"{cache}/{d}", exist_ok=True)
        deploy = cfg["deploy_dir"]
        os.chdir(deploy)
        sys.path[:0] = [os.path.join(deploy, "xvideo", "serving"), deploy]
        t0 = time.time()

        import logging

        import torch

        self.torch = torch
        from xvideo.lowvram import apply_vram_cap_for_testing, low_vram_enabled, te_cpu_offload_enabled
        from xvideo.models import models

        from joyai_lowram.dit_loader import load_dit_streamed
        from joyai_lowram.te_loader import load_text_encoder_lowram

        if not (low_vram_enabled() and te_cpu_offload_enabled()):
            raise EngineError("the JoyAI profile needs JOYOMNI_LOW_VRAM=1 (text-encoder offload)")
        self.dit_report: dict = {}
        self.te_weights = None

        def patched_load_dit(c, device):
            blocks = []
            model = load_dit_streamed(c, device, on_block=lambda i, n: blocks.append(n))
            flags = [[bool(getattr(b, "_fp8_img_installed", False)), bool(getattr(b, "_fp8_txt_installed", False))] for b in model.double_blocks]
            self.dit_report = {
                "stats": getattr(model, "lowram_load_stats", None),
                "blocks": len(blocks),
                "fp8_img": sum(f[0] for f in flags),
                "fp8_txt": sum(f[1] for f in flags),
                "n_blocks": len(flags),
                "meta_left": sum(1 for _, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta),
            }
            return model

        def patched_load_text_encoder(**kw):
            out = load_text_encoder_lowram(**kw)
            self.te_weights = out[1].lowram_weights
            return out

        models.load_dit = patched_load_dit  # must happen before the serving module binds these names
        models.load_text_encoder = patched_load_text_encoder
        logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
        from xvideo.inductor_autotune_fix import install as _install_autotune_fix

        _install_autotune_fix()
        apply_vram_cap_for_testing(DEVICE)
        from xvideo.models.dit import dit as dit_mod
        from xvideo.serving.joyomni_streaming import JoyOmniRuntime, StreamingSettings

        self.StreamingSettings = StreamingSettings
        self.backend = dit_mod.attention_backend()
        need = cfg.get("require_attention", "sage")
        if need == "sage" and self.backend != "sage":
            raise EngineError(f"attention backend is {self.backend!r}, expected 'sage' (SageAttention not installed or FA4 present in the runtime)")
        long_, short = max(cfg["width"], cfg["height"]), min(cfg["width"], cfg["height"])
        self.runtime = JoyOmniRuntime.load(
            cfg["dit_ckpt"],
            vae_ckpt=cfg["vae_ckpt"],
            text_encoder_ckpt=cfg["text_encoder_ckpt"],
            device=DEVICE,
            vae_device=None,
            vae_encode_device=DEVICE,
            vae_decode_device=DEVICE,
            vae_pseudo_device=DEVICE,
            postprocess_device=DEVICE,
            seed=42,
            warmup_height=short,
            warmup_width=long_,
        )
        if (
            self.dit_report.get("fp8_img") != self.dit_report.get("n_blocks")
            or self.dit_report.get("fp8_txt") != self.dit_report.get("n_blocks")
            or self.dit_report.get("meta_left")
        ):
            raise EngineError(f"DiT load incomplete: {self.dit_report}")
        align = self.runtime.pipeline.vae.stem.stride * 8
        self.long = _snap_to_align(long_, align)
        self.short = _snap_to_align(short, align)
        self.load_seconds = round(time.time() - t0, 3)
        self.jobs = 0
        self.graph_captures = 0
        self.lock = threading.Lock()

    # -- reporting ------------------------------------------------------------------------------------------
    def memory_info(self) -> dict:
        torch = self.torch
        return {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        }

    def ready_info(self) -> dict:
        torch = self.torch
        import importlib.metadata as md

        def ver(name):
            try:
                return md.version(name)
            except md.PackageNotFoundError:
                return None

        te = self.te_weights
        return {
            "attention_backend": self.backend,
            "fp8_blocks": [self.dit_report.get("fp8_img"), self.dit_report.get("fp8_txt"), self.dit_report.get("n_blocks")],
            "dit_load": self.dit_report.get("stats"),
            "text_encoder_resident": bool(te is not None and te.resident),
            "text_encoder_pinned": bool(te is not None and te.pin),
            "session_size_landscape": [self.short, self.long],
            "engine_load_seconds": self.load_seconds,
            "versions": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "sageattention": ver("sageattention"),
                "transformers": ver("transformers"),
                "diffusers": ver("diffusers"),
                "python": sys.version.split()[0],
            },
            **self.memory_info(),
        }

    def status(self) -> dict:
        te = self.te_weights
        return {
            "engine": "joyai",
            "jobs": self.jobs,
            "attention_backend": self.backend,
            "graph_runners": len(self.runtime.graph_runners),
            "graph_capture_failures": dict((str(k), v) for k, v in self.runtime.graph_capture_failures.items()),
            "text_encoder_cached_bytes": te.cached_bytes() if te is not None else None,
            **self.memory_info(),
        }

    # -- one job ----------------------------------------------------------------------------------------------
    def run(self, job: dict, frames: np.ndarray, ref, *, should_cancel, on_progress):
        from PIL import Image

        from joyai_protocol import JobCancelled

        torch = self.torch
        n = frames.shape[0]
        orientation = job["orientation"]
        if orientation == "auto":
            orientation = "portrait" if job["height"] > job["width"] else "landscape"
        sh, sw = (self.long, self.short) if orientation == "portrait" else (self.short, self.long)
        rt = self.runtime
        rt.output_quality = 60  # server default ("auto"); unused in lossless mode
        rt.lossless_output = True  # file mode: frames come back as RGB uint8 arrays
        settings = self.StreamingSettings(
            height=sh,
            width=sw,
            num_inference_steps=int(job["num_inference_steps"]),
            seed=int(job["seed"]),
            max_temporal_ids=int(job["max_temporal_ids"]),
            freeze_kv_on_static=False,
            static_diff_thresh=STATIC_DIFF_THRESH,
            profile_timings=False,
            output_codec="mjpeg",
        )
        ref_image = Image.fromarray(np.ascontiguousarray(ref), "RGB") if ref is not None else None
        graphs_before = len(rt.graph_runners)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        session = rt.create_v2v_session(prompt=job["prompt"], settings=settings, ref_image=ref_image)
        out = np.empty((n, sh, sw, 3), dtype=np.uint8)
        got = np.zeros(n, dtype=bool)
        state = {"frames": 0, "chunks": 0, "error": None, "stop": False, "first_frame_s": None}

        def collect():
            try:
                while not state["stop"]:
                    res = session.wait_async_result(0.05)
                    if res is None:
                        continue
                    arrays = res.jpegs or []
                    metas = res.source_metas or [{} for _ in arrays]
                    if res.valid_count is not None:
                        arrays, metas = arrays[: res.valid_count], metas[: res.valid_count]
                    for a, m in zip(arrays, metas):
                        seq = int(m.get("seq", -1))
                        if not 0 <= seq < n or got[seq]:
                            raise EngineError(f"unexpected output frame seq {seq}")
                        a = np.asarray(a)
                        if a.dtype != np.uint8 or a.shape != (sh, sw, 3):
                            raise EngineError(f"unexpected output frame {a.dtype} {a.shape}")
                        out[seq] = a
                        got[seq] = True
                    state["frames"] = int(got.sum())
                    state["chunks"] += 1
                    if state["first_frame_s"] is None and state["frames"]:
                        state["first_frame_s"] = round(time.time() - t0, 3)
                    on_progress(state["frames"], state["chunks"])
            except BaseException as exc:  # noqa: BLE001 - handed to the job thread
                state["error"] = exc

        collector = threading.Thread(target=collect, name="joyai-collect", daemon=True)
        collector.start()
        cancelled = False
        try:
            for i in range(n):
                if should_cancel():
                    cancelled = True
                    break
                deadline = time.monotonic() + CAPACITY_TIMEOUT_S
                while session.inflight_chunks() >= MAX_INFLIGHT_CHUNKS:  # the server's file-input backpressure
                    if state["error"] is not None:
                        raise state["error"]
                    if should_cancel():
                        cancelled = True
                        break
                    if time.monotonic() >= deadline:
                        raise EngineError("timed out waiting for inference capacity")
                    time.sleep(0.01)
                if cancelled:
                    break
                if state["error"] is not None:
                    raise state["error"]
                frame = Image.fromarray(frames[i], "RGB")
                meta = {"seq": i, "t_capture_ms": i * 1000.0 / FPS, "t_server_recv_ms": time.time() * 1000.0}
                session.push_frame(frame, frame_meta=meta, drain_results=False)
            if not cancelled:
                session.flush_pending()
                deadline = time.monotonic() + DRAIN_TIMEOUT_S
                while state["frames"] < n:
                    if state["error"] is not None:
                        raise state["error"]
                    if should_cancel():
                        cancelled = True
                        break
                    if time.monotonic() >= deadline:
                        raise EngineError(f"incomplete output: {state['frames']}/{n} frames")
                    time.sleep(0.01)
        finally:
            state["stop"] = True
            collector.join(timeout=5.0)
            session.request_stop()
            try:
                session.close(timeout=CLOSE_TIMEOUT_S)
            except Exception as exc:  # the server treats this as "restart required"
                raise FatalEngineError(f"session close failed: {exc!r}") from exc
        if state["error"] is not None:
            raise state["error"]
        if cancelled:
            raise JobCancelled()
        if not got.all():
            raise EngineError(f"missing output frames: {int((~got).sum())}")
        self.jobs += 1
        graphs_after = len(rt.graph_runners)
        info = {
            "engine": "joyai",
            "session_size": [sh, sw],
            "orientation": orientation,
            "session_seconds": round(time.time() - t0, 3),
            "first_output_seconds": state["first_frame_s"],
            "chunks": state["chunks"],
            "attention_backend": self.backend,
            "graph_runners_before": graphs_before,
            "graph_runners_after": graphs_after,
            "graph_capture_failures": dict((str(k), v) for k, v in rt.graph_capture_failures.items()),
            "jobs_in_engine": self.jobs,
            "settings": {
                "num_inference_steps": settings.num_inference_steps,
                "seed": settings.seed,
                "max_temporal_ids": settings.max_temporal_ids,
                "freeze_kv_on_static": False,
                "lossless_output": True,
                "max_inflight_chunks": MAX_INFLIGHT_CHUNKS,
            },
            "reference_image": ref is not None,
            **self.memory_info(),
        }
        return out, info
