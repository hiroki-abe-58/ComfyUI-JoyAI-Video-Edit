# Testing

CPU tests (no GPU, no weights; the fake engine stands in for the model behind the real worker, protocol and
process control): `COMFYUI_PATH=<ComfyUI v0.38.0 checkout> python -m pytest -q`.

`runtime/joyai_lowram/` is byte-identical to the placement patch used for the PATCHED_REFERENCE run
(sha256 prefixes: `__init__.py` 9173ef80, `dit_loader.py` 481e5e0f, `te_loader.py` ffe5920d; checked by a test).

GPU tests (maintainer, not CI): `scripts/gpu_e2e.py` drives a real ComfyUI queue; comparison layers L0-L4 and
the results are recorded outside the repository until a release.
## GPU results summary (local, 2026-10-07, not CI)
Real ComfyUI v0.38.0 queue, RTX 5090, WSL2 worker: 120 / 121 / 240-frame jobs complete with one worker reused across
jobs (A -> B -> A); unload / reload, cancel mid-stream / during first-frame init / during load, ComfyUI killed idle
and busy, worker killed mid-job, missing distro / checkpoint, invalid inputs, job timeout - all handled. Comparison
with the patched reference: see docs/LIMITS.md.
## v0.1 release checks (local, 2026-10-08, not CI)
Real queue on the test machine with the v0.1 surface: 120 / 121-frame jobs on one persistent worker (A -> B -> A),
an old workflow carrying the removed inputs ran with the fixed profile (the extra inputs are ignored), Unload ->
"Waiting for WSL memory release" (34 s) -> reload, and a second ComfyUI process was refused with "in use by another
ComfyUI process" while the first held the model. CPU suite: singleton (exclusive lock, 4-process start race with one
winner, owner-checked cleanup), mandatory guard, v0.1 surface, release message rate limit.