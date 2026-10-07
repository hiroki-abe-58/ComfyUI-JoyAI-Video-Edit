# Security notes

- Workflows choose a runtime id only. The distro, user, Python, upstream folder and checkpoints come from the
  administrator's config file (strict schema, unknown keys refused, absolute Linux paths without `..`).
- The worker is started with a fixed argv list (`wsl.exe -d <distro> [-u <user>] --cd / --exec <python> -I -B
  <frozen copy of runtime/joyai_worker.py> <worker.json>`); no shell anywhere. Workflow values travel only inside
  JSON job files.
- No network listener: control over the wsl.exe stdin/stdout pipes, frames as raw files in a fresh job folder under
  the ComfyUI temp directory. The worker refuses job paths outside the jobs root and links.
- Environment: the launcher gets a minimal Windows environment with `WSLENV` empty; the worker drops variables that
  look like credentials, sets Hugging Face offline mode and empties the prompt-enhancement variables.
- Process ownership: `wsl.exe` runs in a Job Object with KILL_ON_JOB_CLOSE; the worker leads its own process group,
  exits on stdin EOF, and a kill helper (pid + start time + command line checks) removes leftovers.
- Memory guard: admission before start and before each job, stop at >= 95 % of the commit limit for 5 s or >= 97 %
  once; the configuration can make it stricter, never looser.
- Messages and reports redact absolute paths and the user name.
This is isolation of configuration and arguments, not a sandbox: the runtime runs with the rights of the user that
started ComfyUI (root inside the dedicated distro in the tested setup).