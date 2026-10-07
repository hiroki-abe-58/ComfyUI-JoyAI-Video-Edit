# Security notes

- Workflows choose a runtime id only. The distro, user, Python, upstream folder and checkpoints come from the
  administrator's config file (strict schema, unknown keys refused, absolute Linux paths without `..`).
- The worker is started with a fixed argv list (`wsl.exe -d <distro> [-u <user>] --cd / --exec <python> -I -B
  <frozen copy of runtime/joyai_worker.py> <worker.json>`); no shell anywhere. Workflow values travel only inside
  JSON job files.
- No network listener: control over the wsl.exe stdin/stdout pipes, frames as raw files in a fresh job folder under
  a per-user folder outside ComfyUI (%LOCALAPPDATA%\ComfyUI-JoyAI-Video-Edit\jobs by default; ComfyUI deletes its own
  temp folder at start-up and exit). Folders carry an owner record; cleanup only touches folders of a ComfyUI process
  that is gone. The worker refuses job paths outside the jobs root and links.
- Environment: the launcher gets a minimal Windows environment with `WSLENV` empty; the worker drops variables that
  look like credentials, sets Hugging Face offline mode and empties the prompt-enhancement variables.
- Process ownership: `wsl.exe` runs in a Job Object with KILL_ON_JOB_CLOSE; the worker leads its own process group,
  exits on stdin EOF, and a kill helper (pid + start time + command line checks) removes leftovers of ComfyUI
  processes that are gone (owner pid + start time recorded); workers of a live ComfyUI process are never touched.
- One model per distro on the machine: an exclusive lock-file handle (kernel-enforced, released by the OS when the
  process ends) is held from admission until the worker's memory is released.
- Memory guard (v0.1, mandatory for WSL runtimes): a model load needs commit_now + 65 GiB <= 90 % of the current commit limit; no new load or job at
  >= 90 %; warning at >= 85 %; a running job is stopped as soon as commit reaches 95 %. After a worker stops, the
  Loader waits (up to 120 s, shown as "Waiting for WSL memory release") until the commit charge is back near its
  pre-load level instead of starting a load on residual memory. The configuration can make this stricter, never looser.
- Messages and reports redact absolute paths and the user name.
This is isolation of configuration and arguments, not a sandbox: the runtime runs with the rights of the user that
started ComfyUI (root inside the dedicated distro in the tested setup).
## The worker runs as root inside a dedicated distro
- Why root today: the tested runtime (Python env, compiled kernels, inductor / Triton caches, checkpoints under
  `/opt/joyai`) was installed and validated as root; the optional `memory_max` cgroup scope uses the system
  `systemd-run --scope`, which needs root. The model itself does not need root.
- Blast radius: root controls the dedicated distro only. Note that any user in a WSL distro - root or not - can read and
  write the Windows drives mounted under `/mnt/<drive>` and start Windows programs through WSL interop, with the rights
  of the Windows user who started ComfyUI. Running the worker as a non-root Linux user therefore narrows access inside
  the distro, not the access to Windows files.
- What limits exposure: no network listener (stdin/stdout pipes + job files; verified with `ss` while loaded and after a
  job); a dedicated distro used for nothing else; workflows cannot choose an executable, path, distro or user (admin
  config only, strict schema); offline mode and credential-like variables dropped; the Windows side owns the process
  tree (Job Object) and a checked kill helper removes leftovers.
- Possible non-root route: create a `joyai` user in the distro, give it read access to the checkpoints and write access
  to the cache folder, set `"user": "joyai"` in the runtime config and leave `memory_max` unset (or use a user systemd
  instance). Optionally restrict the dedicated distro further in its own `/etc/wsl.conf` (automount / interop) - a
  distro setting the administrator decides; not tested here.