# Limits (v0, local)

- Verified only on Windows 11 + WSL2 + RTX 5090 32 GB, profile 840x480, steps 2, max_temporal_ids 8, raw prompts,
  no reference image, no prompt enhancement, gates off.
- Up to 600 frames per job (the upstream server starts a new session every 600 frames; not replicated).
- Not bitwise repeatable, also with the same seed (upstream: VAE sampling on the global CUDA RNG, SageAttention / FP8
  fast-accumulation kernels).
- Results are compared with a PATCHED_REFERENCE (upstream server with the same low host-RAM weight placement), not
  with an unmodified official run, which did not fit this machine's memory.
- Cancelling during the model load or the first-frame initialisation takes effect only after that step (or by
  stopping the worker).