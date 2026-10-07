# Limits (v0, local)

- Verified only on Windows 11 + WSL2 + RTX 5090 32 GB, profile 840x480, steps 2, max_temporal_ids 8, raw prompts,
  no reference image, no prompt enhancement, gates off. Portrait orientation and the reference image are untested.
- Up to 600 frames per job (the upstream server starts a new session every 600 frames; not replicated).
- Not bitwise repeatable, also with the same seed. Upstream samples VAE latents from the global CUDA RNG, which is not
  reseeded per session, so a result depends on what the worker ran before; SageAttention / FP8 fast-accumulation
  kernels add further variation. In our runs, the same input, prompt and seed gave per-frame mean differences of
  about 1.0-1.5 (synthetic scene) and 4-6 (upstream example, watercolor prompt) levels out of 255 between repeats.
- Results are compared with a PATCHED_REFERENCE (upstream server with the same low host-RAM weight placement), not
  with an unmodified official run, which did not fit this machine's memory. Against that reference the input,
  transport and output layers are byte-identical; the model output differs within roughly the same range as the
  reference differs from itself, but a pre-registered strict rule (cross pairs within the range of two self pairs)
  was not met for 2 of 3 test cases. No parity claim beyond that is made.
- Memory (Windows commit charge, 64 GB RAM + automatic pagefile): a loaded worker plus ComfyUI peaked at about
  85.6 GiB of commit (84 % of the limit at the time); a model load added up to ~62 GiB. The v0 guard therefore asks
  for commit_now + 65 GiB <= 90 % of the current commit limit before a load (on the test machine: at most ~21-26 GiB
  committed before loading, depending on the limit). After the worker stops, the WSL2 VM keeps 25-30 GiB committed
  until it stops by itself (20-80 s); the Loader shows "Waiting for WSL memory release" and waits up to 120 s.
- First start compiles kernels (about 14 min without a cache); later starts load in about 80 s.
- Cancelling during the model load or the first-frame initialisation takes effect only after that step (or by
  stopping the worker).