from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
from conftest import frames

from joyai_comfy import jobs
from joyai_comfy import protocol as P
from joyai_comfy.process import ProcessTree


def test_new_dirs_are_unique_and_confined(make_rt):
    rt = make_rt()
    a, pa = jobs.new_dir(rt, "job")
    b, pb = jobs.new_dir(rt, "job")
    root = jobs.jobs_root(rt)
    assert a != b and pa.parent == root and pb.parent == root and P.ID_RE.fullmatch(a)


def test_child_names_are_fixed(make_rt):
    _, d = jobs.new_dir(make_rt(), "job")
    assert jobs.child(d, P.OUTPUT_FILE) == d / "output.u8"
    for bad in ("../x", "output.u8/../../y", "job.json.bak", "C:\\x"):
        with pytest.raises(jobs.JobFilesError):
            jobs.child(d, bad)


def test_write_u8_atomic_and_digest(make_rt):
    _, d = jobs.new_dir(make_rt(), "job")
    f = frames(3)
    digest = jobs.write_u8(d / P.INPUT_FILE, f)
    data = (d / P.INPUT_FILE).read_bytes()
    assert digest == hashlib.sha256(data).hexdigest() and len(data) == f.nbytes
    assert not (d / (P.INPUT_FILE + ".tmp")).exists()
    with pytest.raises(jobs.JobFilesError):
        jobs.write_u8(d / P.INPUT_FILE, f[:, :, ::2])  # not C-contiguous


def _result_for(d, job, out):
    digest = jobs.write_u8(d / P.OUTPUT_FILE, out)
    res = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "frames": out.shape[0],
        "height": out.shape[1],
        "width": out.shape[2],
        "output_bytes": out.nbytes,
        "output_sha256": digest,
    }
    jobs.write_json(d / P.RESULT_FILE, res)
    return res


def _job(jid, n=2):
    return {
        "schema_version": 1,
        "job_id": jid,
        "frames": n,
        "height": 64,
        "width": 64,
        "input_sha256": "0" * 64,
        "prompt": "p",
        "seed": 1,
        "num_inference_steps": 2,
        "max_temporal_ids": 8,
        "orientation": "landscape",
        "reference": None,
    }


def test_read_result_checks(make_rt):
    jid, d = jobs.new_dir(make_rt(), "job")
    job = _job(jid)
    out = frames(2, 64, 96)
    _result_for(d, job, out)
    res, got = jobs.read_result(d, job)
    assert np.array_equal(got, out) and res["frames"] == 2
    # tampered bytes
    raw = bytearray((d / P.OUTPUT_FILE).read_bytes())
    raw[0] ^= 1
    (d / P.OUTPUT_FILE).write_bytes(bytes(raw))
    with pytest.raises(jobs.JobFilesError, match="sha256"):
        jobs.read_result(d, job)
    # truncated
    (d / P.OUTPUT_FILE).write_bytes(bytes(raw[:-3]))
    with pytest.raises(jobs.JobFilesError, match="bytes"):
        jobs.read_result(d, job)


def test_read_result_wrong_job_or_count(make_rt):
    jid, d = jobs.new_dir(make_rt(), "job")
    _result_for(d, _job(jid), frames(2, 64, 96))
    with pytest.raises(jobs.JobFilesError):
        jobs.read_result(d, _job("job-20990101-000000-000000000000"))
    with pytest.raises(jobs.JobFilesError):
        jobs.read_result(d, _job(jid, n=3))


def test_missing_result(make_rt):
    jid, d = jobs.new_dir(make_rt(), "job")
    with pytest.raises(jobs.JobFilesError, match="result.json"):
        jobs.read_result(d, _job(jid))


def test_remove_refuses_foreign_paths(make_rt, tmp_path):
    rt = make_rt()
    other = tmp_path / "keep-me-000000"
    other.mkdir()
    jobs.remove(rt, other)
    assert other.exists()
    _, d = jobs.new_dir(rt, "job")
    jobs.remove(rt, d)
    assert not d.exists()


@pytest.mark.skipif(os.name != "nt", reason="junctions are a Windows feature")
def test_junction_job_folder_is_detected(make_rt, tmp_path):
    rt = make_rt()
    root = jobs.jobs_root(rt)
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = root / "job-20990101-000000-aaaaaaaaaaaa"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    try:
        assert jobs._is_link(link)
        jobs.remove(rt, link)
        assert target.exists()
    finally:
        os.rmdir(link)


# -- process tree ---------------------------------------------------------------------------------------------

CHILD = "import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); print(p.pid, flush=True); time.sleep(120)"


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_terminate_ends_the_whole_tree(tmp_path):
    tree = ProcessTree([sys.executable, "-c", CHILD], cwd=tmp_path, env=dict(os.environ), stdout=None, stderr=tmp_path / "err.log")
    grandchild = int(tree.proc.stdout.readline())
    assert _alive(grandchild)
    if os.name == "nt":
        assert tree.started_suspended and set(tree.job_pids()) >= {tree.pid, grandchild}
    rep = tree.terminate()
    tree.close()
    assert rep["active_after"] == 0
    deadline = time.time() + 10
    while time.time() < deadline and _alive(grandchild):
        time.sleep(0.2)
    assert not _alive(grandchild)


@pytest.mark.skipif(os.name != "nt", reason="Job Object semantics")
def test_parent_death_kills_the_tree(tmp_path):
    """A parent that owns the Job Object dies (taskkill /F): the OS closes the job and ends its children."""
    script = tmp_path / "parent.py"
    script.write_text(
        "import os, sys, time\n"
        f"sys.path.insert(0, {str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))!r})\n"
        "from joyai_comfy.process import ProcessTree\n"
        f"t = ProcessTree([sys.executable, '-c', {CHILD!r}], cwd={str(tmp_path)!r}, env=dict(os.environ), stdout=None, stderr={str(tmp_path / 'e.log')!r})\n"
        "print(t.pid, t.proc.stdout.readline().decode().strip(), flush=True)\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    child, grandchild = (int(x) for x in parent.stdout.readline().split())
    assert _alive(child) and _alive(grandchild)
    subprocess.run(["taskkill", "/PID", str(parent.pid), "/F"], capture_output=True)
    parent.wait(10)
    deadline = time.time() + 15
    while time.time() < deadline and (_alive(child) or _alive(grandchild)):
        time.sleep(0.2)
    assert not _alive(child) and not _alive(grandchild)


def test_result_json_size_cap(make_rt):
    jid, d = jobs.new_dir(make_rt(), "job")
    (d / P.RESULT_FILE).write_text(json.dumps({"pad": "x" * (P.MAX_JSON_FILE + 10)}), encoding="utf-8")
    with pytest.raises(jobs.JobFilesError):
        jobs.read_result(d, _job(jid))
