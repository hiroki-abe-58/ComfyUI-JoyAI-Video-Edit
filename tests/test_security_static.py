"""Static checks of the shipped code (SECURITY.md rows 1, 2, 4, 8, 11, 12, 15, 17)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from joyai_comfy import config
from joyai_comfy import manager as M
from joyai_comfy.redact import redact

REPO = Path(__file__).resolve().parents[1]
SHIPPED = sorted(p for p in list((REPO / "joyai_comfy").glob("*.py")) + list((REPO / "runtime").glob("*.py")) + [REPO / "__init__.py"])
OUR_RUNTIME = [p for p in SHIPPED if "joyai_lowram" not in str(p)]


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "pattern",
    [
        r"shell\s*=\s*True",
        r"\bos\.system\(",
        r"\bos\.popen\(",
        r"subprocess\.\w+\(\s*f[\"']",
        r"\beval\(",
        r"\bexec\(",
        r"\bpickle\b",
        r"allow_pickle\s*=\s*True",
        r"weights_only\s*=\s*False",
        r"\.bind\(",
        r"\.listen\(",
        r"0\.0\.0\.0",
        r"\buvicorn\b",
        r"\bsocket\b",
        r"\burlopen\b",
        r"\bimport (requests|httpx|aiohttp)\b|\bfrom (requests|httpx|aiohttp)\b",
        r"hf_hub_download",
        r"snapshot_download",
        r"\bpip\b install",
        r"\bgit\b (clone|fetch|pull)",
        r"wsl(\.exe)?\W+--(shutdown|terminate|unregister)",
    ],
)
def test_forbidden_patterns_absent(pattern):
    hits = [f"{p.relative_to(REPO)}" for p in SHIPPED if re.search(pattern, _src(p))]
    assert not hits, f"{pattern!r} found in {hits}"


def test_lowram_patch_is_byte_identical_to_the_reference_copy():
    # the placement patch is shipped unchanged from the PATCHED_REFERENCE run (sha256 recorded in docs/TESTING.md)
    import hashlib

    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in (REPO / "runtime" / "joyai_lowram").glob("*.py")}
    assert digests == {"__init__.py": "9173ef808e8a97fb", "dit_loader.py": "481e5e0f68371518", "te_loader.py": "ffe5920dbcd73764"}


def test_no_upstream_files_or_weights_in_tree():
    bad = [
        p
        for p in REPO.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "_comfyui" not in p.parts
        and (p.suffix in (".pth", ".pt", ".safetensors", ".onnx", ".ckpt", ".bin") or "xvideo" in p.parts)
    ]
    assert not bad


def test_launcher_argv_is_a_fixed_list_without_shell():
    rt = config.parse_runtime(
        "r",
        {
            "kind": "wsl",
            "distro": "JoyAI-Ubuntu",
            "user": "root",
            "python": "/opt/joyai/env/bin/python",
            "deploy_dir": "/opt/d",
            "dit_ckpt": "/opt/a",
            "vae_ckpt": "/opt/b",
            "text_encoder_ckpt": "/opt/c",
            "cache_dir": "/opt/e",
            "memory_max": "64G",
        },
    )
    argv = M.launcher_argv(rt, "/mnt/e/x/code/joyai_worker.py", "/mnt/e/x/worker.json", scope=True)
    assert argv[0].lower().endswith("\\system32\\wsl.exe")
    assert argv[1:] == [
        "-d",
        "JoyAI-Ubuntu",
        "-u",
        "root",
        "--cd",
        "/",
        "--exec",
        "/usr/bin/systemd-run",
        "--scope",
        "--quiet",
        "-p",
        "MemoryMax=64G",
        "--",
        "/opt/joyai/env/bin/python",
        "-I",
        "-B",
        "/mnt/e/x/code/joyai_worker.py",
        "/mnt/e/x/worker.json",
    ]
    assert "bash" not in argv and "sh" not in argv and "-c" not in argv


def test_child_env_has_no_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    rt = config.parse_runtime(
        "r",
        {"kind": "wsl", "distro": "D", "python": "/p", "deploy_dir": "/d", "dit_ckpt": "/a", "vae_ckpt": "/b", "text_encoder_ckpt": "/c", "cache_dir": "/e"},
    )
    env = M.child_env(rt)
    assert env["WSLENV"] == ""
    assert not {k for k in env if re.search("TOKEN|KEY|SECRET|PASSWORD", k, re.I)}
    assert set(env) <= {"SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC", "PROGRAMDATA", "LOCALAPPDATA", "USERPROFILE", "PATH", "WSLENV"}


def test_redact_removes_paths_and_user(monkeypatch):
    monkeypatch.setenv("USERNAME", "alicesmith")
    s = redact(r"failed at E:\AI\jobs\x.u8 and /opt/joyai/deps/checkpoints/dit.pth and /mnt/e/AI/x for alicesmith in \\srv\share\y")
    assert "E:\\" not in s and "/opt" not in s and "/mnt" not in s and "alicesmith" not in s and "\\\\srv" not in s
    assert s.count("<path>") == 4


def test_no_workflow_input_accepts_paths_or_commands():
    src = _src(REPO / "joyai_comfy" / "nodes.py")
    for word in ("path", "executable", "command", "distro", "url", "python"):
        assert not re.search(rf'Input\("[^"]*{word}', src, re.I), word
