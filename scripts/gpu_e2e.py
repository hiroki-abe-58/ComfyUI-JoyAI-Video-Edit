"""Maintainer GPU test: drive a running ComfyUI (dedicated test install) through its HTTP queue.

usage: python scripts/gpu_e2e.py --url http://127.0.0.1:8197 --runtime <id> --video <file in ComfyUI/input> --prompt TEXT
       [--seed 42] [--prefix name] [--interrupt-after S] [--unload] [--out result.json] [--timeout 1800]

Workflow (API format, core nodes only + this pack): LoadVideo -> GetVideoComponents -> JoyAI Runtime -> JoyAI Video
Edit -> SaveImage (PNG frames), optionally JoyAI Unload. The input should be a lossless video (e.g. PNG-in-MOV, rgb24)
so the IMAGE batch equals the source bytes. Prints and writes a JSON record: prompt id, status, timings, the Edit
report (from the node's UI text), saved image names, ComfyUI error messages. Nothing else is touched.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid


def _req(url: str, data: dict | None = None, timeout: float = 30.0):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:  # noqa: S310 - localhost test server given by the maintainer
        raw = resp.read()
    return json.loads(raw) if raw else None


def workflow(a) -> dict:
    wf = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": a.video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "JoyAIRuntimeLoader", "inputs": {"runtime": a.runtime, "idle_unload_minutes": 0}},
        "4": {"class_type": "JoyAIVideoEdit", "inputs": {"runtime": ["3", 0], "images": ["2", 0], "prompt": a.prompt, "seed": a.seed}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": a.prefix}},
    }
    if a.unload:
        wf["6"] = {"class_type": "JoyAIUnload", "inputs": {"runtime": ["3", 0], "trigger": ["4", 1]}}  # after the edit report
    return wf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8197")
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="joyai_e2e")
    ap.add_argument("--interrupt-after", type=float, default=None, help="POST /interrupt this many seconds after execution starts")
    ap.add_argument("--unload", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--workflow-json", default=None, help="submit this API-format workflow instead (for error-path tests)")
    a = ap.parse_args()
    wf = json.loads(open(a.workflow_json, encoding="utf-8").read()) if a.workflow_json else workflow(a)
    rec: dict = {"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": {k: v for k, v in vars(a).items() if k != "url"}}
    t0 = time.time()
    try:
        r = _req(f"{a.url}/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    except urllib.error.HTTPError as exc:
        rec.update(status="rejected", http_status=exc.code, body=json.loads(exc.read() or b"{}"))
        print(json.dumps(rec, indent=1))
        if a.out:
            open(a.out, "w", encoding="utf-8").write(json.dumps(rec, indent=1))
        return 2
    pid = r["prompt_id"]
    rec["prompt_id"] = pid
    started = None
    interrupted = False
    while time.time() - t0 < a.timeout:
        hist = _req(f"{a.url}/history/{pid}")
        if hist and pid in hist and hist[pid].get("status", {}).get("completed") is not None:
            h = hist[pid]
            if h["status"].get("status_str") in ("success", "error") or h["status"].get("completed"):
                break
        q = _req(f"{a.url}/queue")
        running = [x for x in q.get("queue_running", []) if x[1] == pid]
        if running and started is None:
            started = time.time()
            rec["queue_wait_s"] = round(started - t0, 2)
        if a.interrupt_after is not None and started is not None and not interrupted and time.time() - started >= a.interrupt_after:
            _req(f"{a.url}/interrupt", {})
            interrupted = True
            rec["interrupt_sent_after_s"] = round(time.time() - started, 2)
        time.sleep(0.5)
    rec["wall_s"] = round(time.time() - t0, 2)
    hist = _req(f"{a.url}/history/{pid}") or {}
    h = hist.get(pid, {})
    st = h.get("status", {})
    rec["status"] = st.get("status_str", "timeout")
    rec["messages"] = [m for m in st.get("messages", []) if m[0] in ("execution_error", "execution_interrupted", "execution_success", "execution_cached")]
    outs = h.get("outputs", {})
    edit_ui = outs.get("4", {})
    if "text" in edit_ui:
        rec["report"] = json.loads(edit_ui["text"][0])
    rec["images"] = [im["filename"] for im in outs.get("5", {}).get("images", [])]
    if "6" in outs:
        rec["unload"] = json.loads(outs["6"]["text"][0])
    print(json.dumps({k: v for k, v in rec.items() if k not in ("images",)}, indent=1)[:6000])
    print("images:", len(rec["images"]))
    if a.out:
        open(a.out, "w", encoding="utf-8").write(json.dumps(rec, indent=1))
    return 0 if rec["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
