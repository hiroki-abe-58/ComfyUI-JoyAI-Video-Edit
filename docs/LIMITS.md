# Limits (v0.1)

- Verified only on Windows 11 + WSL2 + RTX 5090 32 GB with the fixed v0.1 profile: 840x480 landscape session, 2 steps,
  8 temporal ids, raw prompts, no reference image, no prompt enhancement, gates off. The node exposes nothing else;
  taller-than-wide input is refused.
- Up to 600 frames per job (the upstream server starts a new session every 600 frames; not replicated).
- Not bitwise repeatable, also with the same seed (upstream samples part of the latent from the process-wide CUDA
  random state, not reset per job; asynchronous workers and SageAttention / FP8 fast-accumulation kernels add
  variation). Typical frame differences between two runs: ~1.1-1.3 (synthetic scene) and ~3.9-4.2 (upstream example,
  watercolor prompt) levels out of 255.
- Parity: compared with a PATCHED_REFERENCE (unmodified upstream server with the same low host-RAM weight placement),
  not with an unmodified official run (UNMODIFIED_OFFICIAL_REAL_WEIGHT_E2E = NOT_RUN_RESOURCE_BLOCKED). Input,
  transport and output layers are byte-identical; model outputs are within the observed natural variance of the
  patched reference (6 + 6 runs per case): PATCHED_REFERENCE_PARITY_WITHIN_OBSERVED_NATURAL_VARIANCE. A stricter,
  earlier rule with two repeat pairs per case was not met for 2 of 3 cases (kept on record). With the random state
  controlled by test-only instrumentation, the ComfyUI output was bit-identical to the reference in some runs.
- Memory (Windows commit charge, 64 GB RAM + automatic pagefile): peaks of 84-86 GiB commit with ComfyUI + worker;
  a model load added up to ~62 GiB. Load admission: commit_now + 65 GiB <= 90 % of the current commit limit; no new
  load or job at >= 90 %; warning at >= 85 %; a running job stops at 95 %. The guard is mandatory for WSL runtimes.
- After a worker stops, the WSL2 VM keeps 25-30 GiB committed until it stops by itself; the Loader shows "Waiting
  for WSL memory release" and waits up to 120 s until a load would be admitted (observed 30-80 s, sometimes longer
  than a minute).
- One JoyAI model per WSL distro on the machine (machine-wide lock across ComfyUI processes).
- First start compiles kernels (about 14 min without a cache on the test machine); later starts load in about 80 s.
- Cancelling during the model load or the first-frame initialisation takes effect only after that step.
- The Edit node's report is an output string (and in /history); connect a Preview Any node to see it.
- Not verified: portrait sessions, reference images, other step counts / temporal ids, > 600 frames, the CUDA OOM
  path, other GPUs, Linux hosts.