from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from joyai_comfy import protocol as P
from joyai_comfy.validate import InputError, images_to_u8, require_landscape, u8_to_images, validate_images, validate_prompt


def test_every_byte_value_round_trips_exactly():
    u8 = np.arange(256, dtype=np.uint8).repeat(3).reshape(1, 16, 16, 3)
    img = u8_to_images(u8)
    assert img.dtype == torch.float32 and float(img.min()) == 0.0 and float(img.max()) == 1.0
    back = images_to_u8(img)
    assert back.dtype == np.uint8 and back.flags["C_CONTIGUOUS"]
    assert np.array_equal(back, u8)


def test_random_batch_round_trip_and_channel_order():
    rng = np.random.default_rng(1)
    u8 = rng.integers(0, 256, size=(5, 64, 80, 3), dtype=np.uint8)
    u8[:, :8, :8] = (255, 0, 0)  # pure red patch must stay in channel 0
    img = u8_to_images(u8)
    assert float(img[0, 0, 0, 0]) == 1.0 and float(img[0, 0, 0, 1]) == 0.0
    assert np.array_equal(images_to_u8(img), u8)


def test_conversion_of_non_u8_float_images_rounds():
    img = torch.tensor([[[[0.5, 0.25, 1 / 255 * 0.49]]]]).expand(1, 64, 64, 3).contiguous()
    out = images_to_u8(img)
    assert tuple(out[0, 0, 0]) == (128, 64, 0)


@pytest.mark.parametrize(
    "tensor, msg",
    [
        (torch.zeros(2, 64, 64, 4), "convert to RGB"),
        (torch.zeros(2, 64, 64, 1), "convert to RGB"),
        (torch.zeros(64, 64, 3), "IMAGE batch"),
        (torch.zeros(0, 64, 64, 3), "empty"),
        (torch.zeros(2, 63, 64, 3), "within"),
        (torch.zeros(2, 64, 64, 3, dtype=torch.uint8), "float"),
        (torch.full((1, 64, 64, 3), 1.5), "within 0..1"),
        (torch.full((1, 64, 64, 3), -0.1), "within 0..1"),
        (torch.full((1, 64, 64, 3), float("nan")), "finite"),
        (torch.full((1, 64, 64, 3), float("inf")), "finite"),
    ],
)
def test_images_refused(tensor, msg):
    with pytest.raises(InputError, match=msg):
        validate_images(tensor, max_frames=600)


def test_too_many_frames_refused():
    with pytest.raises(InputError, match="at most 4 frames"):
        validate_images(torch.zeros(5, 64, 64, 3), max_frames=4)


def test_not_a_tensor():
    with pytest.raises(InputError):
        validate_images(np.zeros((1, 64, 64, 3)), max_frames=4)


@pytest.mark.parametrize("prompt", ["", "   \n\t", None, "x" * (P.MAX_PROMPT_CHARS + 1), "a\0b"])
def test_prompt_refused(prompt):
    with pytest.raises(InputError):
        validate_prompt(prompt)


def test_prompt_ok():
    assert validate_prompt("make it snow") == "make it snow"


def test_landscape_only():
    require_landscape(720, 1280)
    require_landscape(512, 512)  # square counts as landscape (the verified 840x480 session)
    with pytest.raises(InputError, match="landscape input only"):
        require_landscape(1280, 720)


def test_v01_profile_constants():
    assert P.STEPS_RANGE == (2, 2) and P.TEMPORAL_IDS_RANGE == (8, 8) and P.ORIENTATIONS == ("landscape",)


def job(**over):
    j = {
        "schema_version": 1,
        "job_id": "job-20261007-000000-abcdef012345",
        "frames": 2,
        "height": 64,
        "width": 64,
        "input_sha256": hashlib.sha256(b"x").hexdigest(),
        "prompt": "p",
        "seed": 42,
        "num_inference_steps": 2,
        "max_temporal_ids": 8,
        "orientation": "landscape",
        "reference": None,
    }
    j.update(over)
    return j


def test_job_schema():
    assert P.validate_job(job())["frames"] == 2
    for bad in (
        {"frames": 0},
        {"frames": 601},
        {"job_id": "../../x"},
        {"seed": -1},
        {"num_inference_steps": 9},
        {"orientation": "up"},
        {"input_sha256": "ABC"},
        {"extra": 1},
        {"reference": {"height": 64}},
        {"prompt": " "},
        {"frames": True},
    ):
        with pytest.raises(P.ProtocolError):
            P.validate_job(job(**bad))


def test_result_schema():
    j = job()
    r = {"schema_version": 1, "job_id": j["job_id"], "frames": 2, "height": 480, "width": 840, "output_bytes": 2 * 480 * 840 * 3, "output_sha256": "0" * 64}
    assert P.validate_result(r, j) is r
    for bad in ({"job_id": "job-other-000000000000"}, {"frames": 3}, {"output_bytes": 5}, {"output_sha256": "z"}):
        with pytest.raises(P.ProtocolError):
            P.validate_result({**r, **bad}, j)


def test_protocol_module_is_the_runtime_file():
    from pathlib import Path

    assert P.SOURCE_PATH == Path(__file__).resolve().parents[1] / "runtime" / "joyai_protocol.py"
