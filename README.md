# JoyAI-Video-Edit for ComfyUI (unofficial)

ComfyUI nodes for [JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit), JD's streaming
instruction-based video editing model. The model runs in a persistent worker inside a dedicated WSL2 distribution on
Windows and stays loaded between queue jobs. This is a community integration; it is not affiliated with or endorsed by
JD.com / jd-opensource, Xiaomi or Comfy Org.

**Status: v0.1.0 pre-release.** It was built and tested on one machine profile (below). Read *Requirements*,
*Memory* and *Known limitations* before installing.

## Supported profile (v0.1 - the only configuration that was run on real hardware)
| item | v0.1 |
|---|---|
| Host | Windows 11, ComfyUI v0.38.0 (Python 3.12), 64 GB RAM with an automatic pagefile |
| Worker | WSL2, a dedicated Ubuntu distribution, Python 3.10, torch 2.9.1+cu128 |
| GPU | NVIDIA GeForce RTX 5090 32 GB (SM120), driver 595.95 |
| Model | JoyAI-Video-Edit `ca17e1d` (code), DiT `joyai_video_edit_dit_0811.pth` + VAE (HF revision `39491dda`), MiMo-VL-7B-RL-2508 text encoder (revision `4bfb2707`) |
| Runtime profile | upstream "RTX 5090" profile: SageAttention 2.2.0, FP8 (img + txt, fast accumulation), low-VRAM placement, CUDA graphs |
| Edit settings (fixed) | 840x480 landscape session, 2 inference steps, 8 temporal ids, raw prompt (no prompt enhancement), no reference image, detector gates off |
| Input | IMAGE batch, RGB, landscape (width >= height), 1..600 frames |
Settings outside this profile (portrait video, reference images, other step counts) are not exposed in v0.1.

## Requirements
- Windows 11 with WSL2 and an RTX 5090 (32 GB). Other GPUs and Linux hosts were not tested.
- A dedicated WSL2 distribution for the runtime (see *Security*), with the upstream runtime built inside it:
  Python 3.10 env with upstream `deploy/requirements.txt`, CUDA nvcc 12.8, `joyomni_ops` built for `sm_120a`
  (CUTLASS `dcf215af`), SageAttention 2.2.0 @ `d1a57a5` with upstream's `sageattention-cudagraph-stream.patch`.
  FlashAttention 4 must not be installed in that env.
- About 65 GiB of free Windows commit charge when the model loads (see *Memory*).
- Disk: ~50 GB for the weights inside the distro, plus the runtime and kernel caches.
The node pack itself has no Python dependencies beyond ComfyUI and never downloads, installs or builds anything.

## Setup
1. Build the runtime and download the weights inside the dedicated distro, as described in
   [docs/SETUP.md](docs/SETUP.md) (upstream `deploy/DEPLOYMENT.md`, RTX 5090 section). Check the weight files against
   the Hugging Face SHA-256 digests of the pinned revisions.
2. Install this node pack into `ComfyUI/custom_nodes` (ComfyUI Manager or `git clone`).
3. Copy [examples/joyai.runtimes.example.json](examples/joyai.runtimes.example.json) to
   `<ComfyUI user directory>/joyai.runtimes.json` (or set `COMFYUI_JOYAI_CONFIG`) and fill in the distro name and paths.
   Workflows can only pick a runtime id from this file.
4. Optional check inside the distro: `python -I -B runtime/joyai_doctor.py <config> --gpu` (staged import / CUDA /
   kernel / attention-path checks).
5. Load [workflows/joyai_video_edit_v01.json](workflows/joyai_video_edit_v01.json).

## Nodes
- **JoyAI Runtime** - starts the worker of a configured runtime and loads the model once. The first start compiles
  kernels (about 14 minutes without a cache on the test machine); later starts load in about 80 seconds.
  `idle_unload_minutes` stops an idle worker.
- **JoyAI Video Edit** - IMAGE batch + text instruction -> edited IMAGE batch, one 840x480 frame per input frame.
  Inputs are resized like the upstream server (bicubic, no crop). Returns a JSON report (timings, worker, memory).
- **JoyAI Unload** - stops the worker and frees GPU and host memory.

## Lifecycle
The worker is one process per WSL distro on the machine (a machine-wide lock refuses a second ComfyUI process with
"in use by another ComfyUI process"). Jobs run one at a time on the loaded model. Cancel (ComfyUI's Cancel button)
stops a job between frames; cancelling during the model load or the first-frame initialisation takes effect after
that step. If ComfyUI exits or crashes, the worker exits too (Windows Job Object + stdin EOF).
States: STOPPED -> STARTING -> LOADING -> READY -> (RUNNING -> READY) -> STOPPING -> RELEASING -> STOPPED.

## Memory
On Windows the WSL2 worker's GPU allocations and pinned host memory count against the system commit charge.
- A model load starts only if `commit now + 65 GiB <= 90 %` of the current commit limit (read at every check).
- No new load or job at >= 90 %; a warning at >= 85 %; a running job is stopped at 95 %.
- Measured peaks on the test machine (64 GB RAM, commit limit 101.4 GiB): 84-86 GiB commit (83-85 %) with ComfyUI +
  worker during 120-240-frame jobs; up to ~27.9 GB of the 32 GB GPU.
- The guard is mandatory for WSL runtimes and can only be made stricter in the config.
On the test machine the system commit charge was ~23 GiB with ComfyUI idle, against a start cap of ~26 GiB - other
memory-heavy applications make the load refuse with a clear message instead of starting.

## RELEASING (after Unload)
After the worker stops, the WSL2 VM keeps 25-30 GiB committed for a while. The JoyAI Runtime node then shows
"Waiting for WSL memory release" and starts the next load only when the memory is back and the load would be
admitted. Expect 30-80 seconds, sometimes a little more than a minute; after 120 seconds it stops with an error that
says what to do. Starting a load on residual memory is never attempted.

## Nondeterminism
Outputs are not bitwise repeatable, also with the same input, prompt and seed. Upstream samples part of the latent
from the process-wide CUDA random state (not reset per job), and its asynchronous pipeline and fast attention / FP8
kernels add run-to-run variation. Typical frame difference between two runs: ~1.1-1.3 (synthetic scene) and
~3.9-4.2 (watercolor style on the upstream example video) levels out of 255; individual pixels at moving edges can
differ much more.

## Parity scope
The model could not be run unmodified on the test machine: upstream's low-VRAM path builds the bf16 DiT in host RAM
(about 30 GiB), which did not fit together with the rest. This pack therefore loads the same checkpoints with a low
host-memory placement (`runtime/joyai_lowram`: same parameters, dtypes and FP8 recipe; weights streamed block by block
to the GPU, text-encoder weights read once and pinned). The results are compared with a **patched reference** - the
unmodified upstream server using the same placement - not with an unmodified official run
(UNMODIFIED_OFFICIAL_REAL_WEIGHT_E2E = NOT_RUN_RESOURCE_BLOCKED).
- Input, transport and output layers are byte-identical between ComfyUI and the patched reference.
- With 6 + 6 runs per test case (120, 121 and 240 frames), the differences between the reference and ComfyUI were of
  the same size as the differences between repeated runs of either path:
  **PATCHED_REFERENCE_PARITY_WITHIN_OBSERVED_NATURAL_VARIANCE**.
- In a controlled validation (test-only instrumentation that set the same random state in both paths; not part of the
  pack) ComfyUI produced output bit-identical to the reference in some runs of two cases. This shows the integration
  adds no difference of its own; it does not make production runs repeatable.
- An earlier, stricter pre-registered rule (cross-path spread within the range of two repeat pairs) was not met for 2
  of 3 cases; that result is kept in the project record.

## Known artifacts
- A faint reddish trail or halo can remain next to a recoloured moving object (seen in the synthetic test scene), in
  the patched reference and in ComfyUI alike.
- Strong styles (e.g. watercolor) flicker in fine textures from frame to frame.

## Known limitations
- Landscape input only (taller-than-wide input is refused); output is always 840x480.
- At most 600 frames per job (upstream starts a new session every 600 frames; not replicated).
- One JoyAI model per WSL distro on the machine; jobs run one at a time.
- Not verified: portrait sessions, reference images, other step counts / temporal ids, other GPUs or hosts, the CUDA
  out-of-memory path.
- The Edit node's report is an output string; it is not drawn on the node.

## Security
- Workflows choose a runtime id only. The distro, Linux user, Python, upstream folder and checkpoints come from the
  administrator's config (strict schema). No input accepts an executable, path, distro, URL or command.
- No network listener: the worker talks over the `wsl.exe` stdin/stdout pipes and raw frame files in a job folder.
- The worker runs offline (Hugging Face offline mode), drops credential-like variables and receives no Windows
  environment; nothing is downloaded or built at run time (Triton / inductor compile kernels into the configured
  cache folder on first use).
- **Root worker / dedicated distro:** the tested setup runs the worker as root inside a WSL2 distribution used for
  nothing else. Any WSL user - root or not - can reach the Windows drives under `/mnt` and start Windows programs
  through WSL interop with the rights of the Windows user who started ComfyUI, so use a dedicated distro and treat
  the runtime like any other code you run. A non-root setup is described in [docs/SECURITY.md](docs/SECURITY.md)
  (not tested).

## Troubleshooting
| message | meaning / what to do |
|---|---|
| `memory guard: load the JoyAI model needs 65.0 GiB of commit headroom ...` | close memory-heavy applications or wait, then run again |
| `Waiting for WSL memory release ...` | a previous worker just stopped; the load starts by itself when memory is back |
| `WSL memory was not released within 120 s ...` | wait a minute and retry; check other WSL distros / applications |
| `... is in use by another ComfyUI process` | another ComfyUI on this machine has the model loaded; use JoyAI Unload there |
| `WSL distro '<name>' is not installed` | fix `distro` in the runtime config (`wsl.exe -l -v`) |
| `checkpoint missing in the runtime: ...` | install the named weights inside the distro (docs/SETUP.md) |
| `attention backend is 'sdpa', expected 'sage'` | SageAttention is missing in the runtime env, or FlashAttention 4 is installed |
| `... taller than wide; v0.1 supports landscape input only` | rotate or crop the video |
| first start takes > 10 minutes | kernel compilation on first use; later starts use the cache |

## Attribution and licenses
This repository's code: Apache License 2.0 (LICENSE). See NOTICE for attribution, including the small parts adapted
from JoyAI-Video-Edit. Upstream code, model weights and the text encoder are not included and keep their own
licenses (JoyAI-Video-Edit code and weights: Apache License 2.0; MiMo-VL-7B-RL-2508: MIT License).
Further documents: [docs/LIMITS.md](docs/LIMITS.md), [docs/SECURITY.md](docs/SECURITY.md),
[docs/TESTING.md](docs/TESTING.md).
