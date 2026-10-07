# Setup

The node pack itself has no Python dependencies beyond ComfyUI. The model runs in a separate runtime that the
administrator installs once inside a dedicated WSL2 distribution; the pack never installs or downloads anything.

1. Runtime (inside the distro, as in upstream `deploy/DEPLOYMENT.md`, RTX 5090 section): Python 3.10 env with
   `deploy/requirements.txt`, CUDA nvcc 12.8, `joyomni_ops` built for `sm_120a` (CUTLASS dcf215af), SageAttention
   2.2.0 @ d1a57a5 with upstream's `sageattention-cudagraph-stream.patch`, upstream checkout at ca17e1d.
   FlashAttention 4 must not be installed (the model would silently switch backend; the worker refuses anything but
   SageAttention by default).
2. Weights (pinned revisions; check the sha256 against the Hugging Face API digests):
   `jdopensource/JoyAI-Video-Edit` @ 39491dda (`dit/joyai_video_edit_dit_0811.pth`, `vae/`),
   `XiaomiMiMo/MiMo-VL-7B-RL-2508` @ 4bfb2707 (safetensors shards, tokenizer, config).
3. Copy `examples/joyai.runtimes.example.json` to `<ComfyUI user directory>/joyai.runtimes.json` (or point
   `COMFYUI_JOYAI_CONFIG` at it) and fill in the distro name and paths. Workflows can only pick a runtime id.
4. Optional: `python -I -B runtime/joyai_doctor.py <config> --gpu` inside the distro for the staged checks.

Memory: on Windows the WSL2 worker's GPU and pinned host memory count against the system commit charge. A loaded
worker added ~55 GiB of commit on the test machine (64 GB RAM + automatic pagefile). The admission check refuses
to start unless commit + 57.7 GiB + 2 GiB stays at or below 95 % of the current commit limit.