# Testing

CPU tests (no GPU, no weights; the fake engine stands in for the model behind the real worker, protocol and
process control): `COMFYUI_PATH=<ComfyUI v0.38.0 checkout> python -m pytest -q`.

`runtime/joyai_lowram/` is byte-identical to the placement patch used for the PATCHED_REFERENCE run
(sha256 prefixes: `__init__.py` 9173ef80, `dit_loader.py` 481e5e0f, `te_loader.py` ffe5920d; checked by a test).

GPU tests (maintainer, not CI): `scripts/gpu_e2e.py` drives a real ComfyUI queue; comparison layers L0-L4 and
the results are recorded outside the repository until a release.