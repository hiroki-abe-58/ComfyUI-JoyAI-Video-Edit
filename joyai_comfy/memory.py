"""Host memory snapshot, admission and the sustained-pressure stop (Windows commit charge).

On the machine this was measured on, CUDA allocations of the WSL2 worker counted against the Windows commit charge
almost 1:1, and the commit limit follows the automatic pagefile (it changes) - so the limit is read at every check,
never cached. A value that cannot be measured is "unknown" and is never treated as unlimited: admission refuses.
Rules (never loosened by configuration, see config.GUARD_DEFAULTS):
- admission before a worker start: commit_now + expected_worker + margin <= max_commit_pct of the current limit,
  and physical available >= min_available_gib;
- admission before a job: commit_now + job_factor x job estimate + margin <= max_commit_pct of the current limit;
- while a job runs: stop at >= stop_commit_pct for stop_sustain_seconds, or at >= hard_stop_commit_pct once.
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
    """May a worker start (``start=True``) or a job begin, adding about ``add_gib`` of commit?"""
    if not guard["enabled"]:
        return True, "memory guard disabled by the runtime config"
    what = "start the JoyAI runtime" if start else "start the job"
    if not snap.get("measurable"):
        return False, f"memory guard: cannot measure the commit charge; refusing to {what}"
    if not snap.get("commit_enforced"):
        need = guard["min_available_gib"] if start else 1.0
        if snap["available_gib"] < need:
            return False, f"memory guard: only {snap['available_gib']:.1f} GiB available (needs {need:.1f} GiB to {what})"
        return True, f"ok: available {snap['available_gib']:.1f} GiB (commit not enforced on this system)"
    limit = snap["commit_limit_gib"]
    projected = snap["commit_gib"] + add_gib + guard["margin_gib"]
    cap = guard["max_commit_pct"] / 100.0 * limit
    if projected > cap:
        return False, (
            f"memory guard: projected commit {projected:.1f} GiB ({snap['commit_gib']:.1f} now + {add_gib:.1f} + "
            f"{guard['margin_gib']:.1f} margin) > {guard['max_commit_pct']:.0f} % of the current limit {limit:.1f} GiB "
            f"({cap:.1f} GiB) - close other GPU/RAM users and retry"
        )
    if start and snap["available_gib"] < guard["min_available_gib"]:
        return False, f"memory guard: only {snap['available_gib']:.1f} GiB physical memory available (needs {guard['min_available_gib']:.1f} GiB)"
    return True, f"ok: projected {projected:.1f} GiB <= {cap:.1f} GiB ({guard['max_commit_pct']:.0f} % of {limit:.1f} GiB)"


@dataclass
class PressureWatch:
    """True once commit stayed >= stop_commit_pct for stop_sustain_seconds, or reached hard_stop_commit_pct once.
    Unmeasurable counts as high."""

    guard: dict
    since: float | None = None
    peak_pct: float = 0.0
    last: dict = field(default_factory=dict)
    reason: str = ""

    def update(self, snap: dict) -> bool:
        self.last = snap
        if not self.guard["enabled"]:
            return False
        now = snap.get("t", time.monotonic())
        if snap.get("measurable"):
            pct = snap["commit_pct"] if snap.get("commit_enforced") else 0.0
            self.peak_pct = max(self.peak_pct, pct)
            if pct >= self.guard["hard_stop_commit_pct"]:
                self.reason = f"commit reached {pct:.1f} % (hard stop {self.guard['hard_stop_commit_pct']:.0f} %)"
                return True
            high = pct >= self.guard["stop_commit_pct"]
        else:
            high = True
        if not high:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        if now - self.since >= self.guard["stop_sustain_seconds"]:
            self.reason = f"commit stayed at or above {self.guard['stop_commit_pct']:.0f} % for {self.guard['stop_sustain_seconds']:.0f} s"
            return True
        return False
