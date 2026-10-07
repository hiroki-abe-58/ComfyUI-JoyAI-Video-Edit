# JoyAI-Video-Edit for ComfyUI (unofficial)

ComfyUI nodes for [JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit), JD's streaming
instruction-based video editing model, on an RTX 5090 (32 GB) under Windows. The model runs in a persistent worker
inside a dedicated WSL2 distribution and stays loaded between queue jobs.

Status: local development version, not released. See docs/LIMITS.md.

## Nodes
- **JoyAI Runtime** - starts the worker for an administrator-registered runtime and loads the model once.
- **JoyAI Video Edit** - IMAGE batch + text instruction -> edited IMAGE batch (one frame per input frame, 840x480
  or 480x840).
- **JoyAI Unload** - stops the worker and frees GPU and host memory.

## How it runs
The worker imports the upstream checkout unchanged and drives its in-process streaming API the way the upstream
server's file mode does (lossless frames, the RTX 5090 profile: FP8 + SageAttention + low-VRAM placement + CUDA
graphs). The only replaced parts are how the DiT and the MiMo-VL text-encoder weights are placed in memory
(`runtime/joyai_lowram`, same checkpoints, dtypes and FP8 recipe), so that it fits a 64 GB host. Results are
therefore compared with that patched reference, not with an unmodified official run.

Setup: docs/SETUP.md. Security notes: docs/SECURITY.md. Tests: docs/TESTING.md.

## License
Apache License 2.0 for this repository's code (see LICENSE and NOTICE). Model weights and upstream code are not
included and keep their own licenses.