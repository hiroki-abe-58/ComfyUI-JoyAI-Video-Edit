"""Host memory snapshot, admission and the sustained-pressure stop (Windows commit charge).

On the machine this was measured on, CUDA allocations of the WSL2 worker counted against the Windows commit charge
almost 1:1, and the commit limit follows the automatic pagefile (it changes) - so the limit is read at every check,
never cached. A value that cannot be measured is "unknown" and is never treated as unlimited: admission refuses.
Rules (v0; never loosened by configuration, see config.GUARD_DEFAULTS / GUARD_BOUNDS):
- model load: commit_now + 65 GiB (expected_worker_gib) + margin <= 90 % of the current limit, commit_now itself below
  90 %, and physical available >= min_available_gib;
- job on a loaded worker: commit_now + job_factor x job estimate + margin <= 90 % of the current limit;
- warning (status / log) at >= 85 %; a running job is stopped as soon as commit reaches 95 %.
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass, field

GIB = 1024**3


def snapshot() -> dict:
    snap = {"measurable": False, "source": None, "t": time.monotonic()}
    try:
        if os.name == "nt":
            from ctypes import wintypes

            class PI(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("CommitTotal", ctypes.c_size_t),
                    ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t),
                    ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t),
                    ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_size_t),
                    ("HandleCount", wintypes.DWORD),
                    ("ProcessCount", wintypes.DWORD),
                    ("ThreadCount", wintypes.DWORD),
                ]

            pi = PI()
            pi.cb = ctypes.sizeof(pi)
            if ctypes.WinDLL("psapi").GetPerformanceInfo(ctypes.byref(pi), pi.cb):
                ps = pi.PageSize
                snap.update(
                    source="GetPerformanceInfo",
                    total_gib=pi.PhysicalTotal * ps / GIB,
                    available_gib=pi.PhysicalAvailable * ps / GIB,
                    commit_gib=pi.CommitTotal * ps / GIB,
                    commit_limit_gib=pi.CommitLimit * ps / GIB,
                    nonpaged_gib=pi.KernelNonpaged * ps / GIB,
                )
        elif os.path.exists("/proc/meminfo"):
            vals = {}
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    k, _, rest = line.partition(":")
                    vals[k] = int(rest.split()[0]) * 1024
            snap.update(
                source="/proc/meminfo",
                total_gib=vals["MemTotal"] / GIB,
                available_gib=vals["MemAvailable"] / GIB,
                commit_gib=vals["Committed_AS"] / GIB,
                commit_limit_gib=vals["CommitLimit"] / GIB,
            )
    except Exception:  # noqa: BLE001 - any failure means "unknown", never "unlimited"
        snap["source"] = None
    if snap.get("commit_limit_gib"):
        snap["commit_pct"] = 100.0 * snap["commit_gib"] / snap["commit_limit_gib"]
        snap["measurable"] = True
    snap["commit_enforced"] = os.name == "nt"
    for k in ("total_gib", "available_gib", "commit_gib", "commit_limit_gib", "commit_pct", "nonpaged_gib"):
        if k in snap:
            snap[k] = round(snap[k], 3)
    return snap


def job_estimate_gib(frames: int, in_h: int, in_w: int, out_h: int, out_w: int) -> float:
    """Extra commit of one job on a loaded worker (TEST_PLAN section 6): Windows input uint8 copy + output uint8 +
    output float32 IMAGE; WSL input uint8 + collected output uint8 (per-frame PIL objects are transient)."""
    win = frames * (in_h * in_w * 3 + out_h * out_w * 3 + out_h * out_w * 3 * 4)
    wsl = frames * (in_h * in_w * 3 + out_h * out_w * 3)
    return (win + wsl) / GIB


def check_admission(snap: dict, guard: dict, add_gib: float, *, start: bool) -> tuple[bool, str]:
    """May a model load (``start=True``) or a job begin, adding about ``add_gib`` of commit?"""
    if not guard["enabled"]:
        return True, "memory guard disabled by the runtime config"
    what = "load the JoyAI model" if start else "start the job"
    if not snap.get("measurable"):
        return False, f"memory guard: cannot measure the commit charge; refusing to {what}"
    if not snap.get("commit_enforced"):
        need = guard["min_available_gib"] if start else 1.0
        if snap["available_gib"] < need:
            return False, f"memory guard: only {snap['available_gib']:.1f} GiB available (needs {need:.1f} GiB to {what})"
        return True, f"ok: available {snap['available_gib']:.1f} GiB (commit not enforced on this system)"
    limit = snap["commit_limit_gib"]
    cap = guard["max_commit_pct"] / 100.0 * limit
    projected = snap["commit_gib"] + add_gib + guard["margin_gib"]
    if snap["commit_gib"] >= cap:
        return False, (
            f"memory guard: commit is already {100.0 * snap['commit_gib'] / limit:.1f} % of the current limit {limit:.1f} GiB "
            f"(no new large-memory stage at >= {guard['max_commit_pct']:.0f} %)"
        )
    if projected > cap:
        return False, (
            f"memory guard: {what} needs {add_gib + guard['margin_gib']:.1f} GiB of commit headroom; {snap['commit_gib']:.1f} GiB now "
            f"-> {projected:.1f} GiB > {guard['max_commit_pct']:.0f} % of the current limit {limit:.1f} GiB ({cap:.1f} GiB). "
            "Close other GPU/RAM users and retry"
        )
    if start and snap["available_gib"] < guard["min_available_gib"]:
        return False, f"memory guard: only {snap['available_gib']:.1f} GiB physical memory available (needs {guard['min_available_gib']:.1f} GiB)"
    note = " (warning: at or above the warning level)" if 100.0 * projected / limit >= guard["warn_commit_pct"] else ""
    return True, f"ok: projected {projected:.1f} GiB <= {cap:.1f} GiB ({guard['max_commit_pct']:.0f} % of {limit:.1f} GiB){note}"


def released(snap: dict, baseline_gib: float | None, guard: dict) -> bool:
    """After a worker stopped: has the commit charge come back (to the pre-load level + tolerance)? Unknown -> no."""
    if not snap.get("measurable"):
        return False
    if not snap.get("commit_enforced"):
        return True
    if baseline_gib is None:
        return True
    return snap["commit_gib"] <= baseline_gib + guard["release_tolerance_gib"]


@dataclass
class PressureWatch:
    """Stop a running job as soon as commit reaches hard_stop_commit_pct (or memory became unmeasurable for 5 s);
    remember whether the warning level was crossed."""

    guard: dict
    since_unknown: float | None = None
    peak_pct: float = 0.0
    warned: bool = False
    last: dict = field(default_factory=dict)
    reason: str = ""

    def update(self, snap: dict) -> bool:
        self.last = snap
        if not self.guard["enabled"]:
            return False
        now = snap.get("t", time.monotonic())
        if not snap.get("measurable"):
            if self.since_unknown is None:
                self.since_unknown = now
            if now - self.since_unknown >= 5.0:
                self.reason = "commit charge could not be measured for 5 s"
                return True
            return False
        self.since_unknown = None
        pct = snap["commit_pct"] if snap.get("commit_enforced") else 0.0
        self.peak_pct = max(self.peak_pct, pct)
        if pct >= self.guard["warn_commit_pct"]:
            self.warned = True
        if pct >= self.guard["hard_stop_commit_pct"]:
            self.reason = f"commit reached {pct:.1f} % (hard stop at {self.guard['hard_stop_commit_pct']:.0f} %)"
            return True
        return False
