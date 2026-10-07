"""The wire format, loaded from runtime/joyai_protocol.py (the runtime imports the same file: one source of truth)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "runtime" / "joyai_protocol.py"
_spec = importlib.util.spec_from_file_location("joyai_comfy._joyai_protocol", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

PROTOCOL_VERSION = _mod.PROTOCOL_VERSION
MAX_LINE = _mod.MAX_LINE
MAX_REPLY = _mod.MAX_REPLY
MAX_JSON_FILE = _mod.MAX_JSON_FILE
ID_RE = _mod.ID_RE
MAX_FRAMES = _mod.MAX_FRAMES
MIN_SIDE = _mod.MIN_SIDE
MAX_SIDE = _mod.MAX_SIDE
MAX_PROMPT_CHARS = _mod.MAX_PROMPT_CHARS
STEPS_RANGE = _mod.STEPS_RANGE
TEMPORAL_IDS_RANGE = _mod.TEMPORAL_IDS_RANGE
SEED_RANGE = _mod.SEED_RANGE
ORIENTATIONS = _mod.ORIENTATIONS
JOB_FILE = _mod.JOB_FILE
INPUT_FILE = _mod.INPUT_FILE
OUTPUT_FILE = _mod.OUTPUT_FILE
RESULT_FILE = _mod.RESULT_FILE
CANCEL_FILE = _mod.CANCEL_FILE
ProtocolError = _mod.ProtocolError
u8_nbytes = _mod.u8_nbytes
check_prompt = _mod.check_prompt
validate_job = _mod.validate_job
validate_result = _mod.validate_result
SOURCE_PATH = _PATH
