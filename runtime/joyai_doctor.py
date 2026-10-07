"""Read-only checks of a JoyAI runtime (run inside the distro with the runtime's Python; prints one JSON report).

Usage: python -I -B joyai_doctor.py <worker.json-like config> [--gpu] [--sha256 <expected.json>]

Levels (aigc-blackwell staged smoke; each PASS / FAIL / NOT_RUN):
  1 import torch + versions         2 CUDA available, capability, sm_120 in the arch list
  3 real kernels: bf16 matmul + SDPA forward on the GPU (only with --gpu)
  4 attention path the model would use: joyomni_ops.has_fp8, sageattention kernel vs SDPA, FA4 absent (only with --gpu)
Also: checkpoint files present (and sha256 against an expected list if given), upstream checkout HEAD (read from
.git without running git). Nothing is downloaded, installed or written.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo: Path) -> str | None:
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = repo / ".git" / head[5:]
            if ref.is_file():
                return ref.read_text().strip()
            packed = (repo / ".git" / "packed-refs").read_text().splitlines()
            return next((ln.split()[0] for ln in packed if ln.endswith(head[5:])), None)
        return head
    except OSError:
        return None


def main(argv: list[str]) -> int:
    cfg = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    gpu = "--gpu" in argv
    expected = {}
    if "--sha256" in argv:
        expected = json.loads(Path(argv[argv.index("--sha256") + 1]).read_text(encoding="utf-8"))
    rep: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0], "levels": {}, "checks": {}}
    deploy = Path(cfg["deploy_dir"])
    rep["checks"]["upstream_head"] = _git_head(deploy.parent)
    files = {}
    for key in ("dit_ckpt", "vae_ckpt", "text_encoder_ckpt"):
        p = Path(cfg[key])
        files[key] = {"exists": p.exists(), "is_dir": p.is_dir()}
        if p.is_file():
            files[key]["bytes"] = p.stat().st_size
    for rel, digest in expected.items():
        p = Path(rel)
        ok = p.is_file() and _sha256(p) == digest
        files.setdefault("sha256", {})[p.name] = "match" if ok else "MISMATCH_OR_MISSING"
    rep["checks"]["files"] = files
    try:
        import torch

        rep["levels"]["1_import_torch"] = "PASS"
        rep["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    except Exception as exc:  # noqa: BLE001
        rep["levels"]["1_import_torch"] = f"FAIL {type(exc).__name__}: {exc}"
        print(json.dumps(rep))
        return 1
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        rep["gpu"] = {"name": torch.cuda.get_device_name(0), "capability": list(cap), "arch_list": archs}
        rep["levels"]["2_cuda_capability"] = "PASS" if cap == (12, 0) and any(a.startswith("sm_120") for a in archs) else f"FAIL cap {cap} archs {archs}"
    else:
        rep["levels"]["2_cuda_capability"] = "FAIL cuda not available"
    if not gpu:
        rep["levels"]["3_real_kernel"] = rep["levels"]["4_attention_path"] = "NOT_RUN (no --gpu)"
        print(json.dumps(rep))
        return 0
    try:
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        mm = (a @ a).float()
        q = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
        o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        rep["levels"]["3_real_kernel"] = "PASS" if torch.isfinite(mm).all() and torch.isfinite(o).all() else "FAIL non-finite"
    except Exception as exc:  # noqa: BLE001
        rep["levels"]["3_real_kernel"] = f"FAIL {type(exc).__name__}: {exc}"
    try:
        import importlib.util

        rep["checks"]["flash_attn_cute_present"] = importlib.util.find_spec("flash_attn") is not None
        import joyomni_ops

        rep["checks"]["joyomni_ops_has_fp8"] = bool(
            getattr(joyomni_ops, "has_fp8", lambda: False)() if callable(getattr(joyomni_ops, "has_fp8", None)) else getattr(joyomni_ops, "has_fp8", False)
        )
        from sageattention import sageattn

        so = sageattn(q, q, q, tensor_layout="HND", is_causal=False)
        torch.cuda.synchronize()
        diff = (so.float() - o.float()).abs().max().item()
        rep["checks"]["sage_vs_sdpa_max_abs"] = round(diff, 4)
        ok = rep["checks"]["joyomni_ops_has_fp8"] and not rep["checks"]["flash_attn_cute_present"] and diff < 1.0
        rep["levels"]["4_attention_path"] = "PASS" if ok else "FAIL see checks"
    except Exception as exc:  # noqa: BLE001
        rep["levels"]["4_attention_path"] = f"FAIL {type(exc).__name__}: {exc}"
    rep["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
