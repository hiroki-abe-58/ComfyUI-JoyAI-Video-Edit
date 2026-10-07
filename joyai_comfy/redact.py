"""Remove absolute paths and the user name from text shown in the UI or written into reports."""

from __future__ import annotations

import getpass
import os
import re

_WIN = re.compile(r"(?i)\b[a-z]:[\\/][^\s'\"<>|]*")
_UNC = re.compile(r"\\\\[^\s'\"<>|]+")
_POSIX = re.compile(r"(?<![\w.])/(?:mnt|home|opt|root|tmp|usr|var|srv|data|media|Users)(?:/[^\s'\"<>|:]*)?")


def _user_names() -> set[str]:
    names = set()
    for get in (getpass.getuser, lambda: os.environ.get("USERNAME", ""), lambda: os.environ.get("USER", "")):
        try:
            n = get()
        except Exception:  # noqa: BLE001
            n = ""
        if n and len(n) >= 3:
            names.add(n)
    return names


def redact(text) -> str:
    s = str(text)
    s = _UNC.sub("<path>", s)
    s = _WIN.sub("<path>", s)
    s = _POSIX.sub("<path>", s)
    for name in _user_names():
        s = re.sub(re.escape(name), "<user>", s, flags=re.I)
    return s
