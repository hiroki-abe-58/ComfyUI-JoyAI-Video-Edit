"""ComfyUI V3 nodes: JoyAI Runtime (load), JoyAI Video Edit, JoyAI Unload."""

from __future__ import annotations

import json
import math

from comfy_api.latest import ComfyExtension, io

from . import config, memory
from . import protocol as P
from .manager import MANAGER, AdmissionRefused, JobCancelled, JobTimeout, ReleaseTimeout, WorkerError
from .redact import redact
from .validate import InputError, images_to_u8, orientation_for, u8_to_images, validate_images, validate_prompt

JoyAIRuntime = io.Custom("JOYAI_RUNTIME")
CATEGORY = "JoyAI/Video Edit"
NO_RUNTIME = "(no runtime configured)"


def _runtime_ids() -> list[str]:
    try:
        ids = sorted(config.load_runtimes())
    except config.ConfigError:
        ids = []
    return ids or [NO_RUNTIME]


def _get_runtime(rid: str) -> config.Runtime:
    try:
        runtimes = config.load_runtimes()
    except config.ConfigError as exc:
        raise RuntimeError(f"JoyAI runtime config is invalid: {redact(exc)}") from None
    if not runtimes:
        raise RuntimeError(f"JoyAI runtime config not found: set {config.CONFIG_ENV} or create <ComfyUI user dir>/{config.CONFIG_FILENAME} (see docs/SETUP.md)")
    if rid not in runtimes:
        raise RuntimeError(f"JoyAI runtime {rid!r} is not defined in {config.CONFIG_FILENAME}")
    return runtimes[rid]


def _interrupted() -> bool:
    import comfy.model_management as mm

    return mm.processing_interrupted()


def _notifier(node_id):
    """Show a status line on the node (e.g. while waiting for the WSL memory release); also logged."""

    def notify(text: str) -> None:
        import logging

        logging.info("[JoyAI] %s", text)
        try:
            from server import PromptServer

            if node_id is not None and PromptServer.instance is not None:
                PromptServer.instance.send_progress_text(text, node_id)
        except Exception:  # noqa: BLE001 - status text is best effort
            pass

    return notify


def _raise_interrupt_or(msg: str):
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()  # the user pressed Cancel: ComfyUI's own interrupt exception
    raise RuntimeError(msg)


class JoyAIRuntimeLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyAIRuntimeLoader",
            display_name="JoyAI Runtime",
            category=CATEGORY,
            description=(
                "Starts the JoyAI-Video-Edit worker of an administrator-registered runtime (WSL2) and loads the model once "
                "(DiT 0811 FP8 + MiMo-VL text encoder, RTX 5090 profile). The first start compiles kernels and can take "
                "10+ minutes; later starts use the cache. The worker stays loaded until Unload, idle timeout or ComfyUI exit."
            ),
            inputs=[
                io.Combo.Input("runtime", options=_runtime_ids(), tooltip=f"A runtime id from {config.CONFIG_FILENAME} (set up by the administrator)"),
                io.Int.Input(
                    "idle_unload_minutes",
                    default=30,
                    min=0,
                    max=240,
                    step=1,
                    tooltip="Stop the worker after this many minutes without a job (0 = keep until Unload / ComfyUI exit)",
                ),
            ],
            outputs=[JoyAIRuntime.Output(display_name="runtime"), io.String.Output(display_name="status")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, runtime, idle_unload_minutes):
        # re-run after Unload / worker death (the generation changes), otherwise the cached handle stays valid
        return f"{MANAGER.generation}:{MANAGER.state}"

    @classmethod
    def execute(cls, runtime, idle_unload_minutes) -> io.NodeOutput:
        rt = _get_runtime(runtime)
        try:
            status = MANAGER.load(
                rt, idle_minutes=int(idle_unload_minutes), interrupted=_interrupted, notify=_notifier(getattr(getattr(cls, "hidden", None), "unique_id", None))
            )
        except JobCancelled:
            _raise_interrupt_or("JoyAI runtime start was cancelled")
        except ReleaseTimeout as exc:
            raise RuntimeError(f"Waiting for WSL memory release timed out: {redact(exc)}") from None
        except (AdmissionRefused, WorkerError, JobTimeout) as exc:
            raise RuntimeError(redact(exc)) from None
        snap = memory.snapshot()
        status["memory"] = {k: snap.get(k) for k in ("commit_gib", "commit_limit_gib", "commit_pct", "available_gib")}
        handle = {"runtime_id": rt.id, "generation": MANAGER.generation}
        return io.NodeOutput(handle, json.dumps(status, indent=1, default=str))


class JoyAIVideoEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyAIVideoEdit",
            display_name="JoyAI Video Edit",
            category=CATEGORY,
            description=(
                "Edits a video (IMAGE batch, RGB) with a text instruction using JoyAI-Video-Edit's streaming model, frame by "
                "frame in 8-frame chunks. Output = one frame per input frame at the runtime's session size (840x480 or "
                "480x840); inputs are resized like the upstream server (bicubic, no crop). Not bitwise repeatable."
            ),
            inputs=[
                JoyAIRuntime.Input("runtime"),
                io.Image.Input("images", tooltip=f"Video frames, RGB, 1..{P.MAX_FRAMES} frames"),
                io.String.Input("prompt", multiline=True, default="", tooltip="Edit instruction (raw prompt; prompt enhancement is not used)"),
                io.Int.Input("seed", default=42, min=P.SEED_RANGE[0], max=P.SEED_RANGE[1], control_after_generate=False),
                io.Int.Input(
                    "num_inference_steps",
                    default=2,
                    min=P.STEPS_RANGE[0],
                    max=P.STEPS_RANGE[1],
                    advanced=True,
                    tooltip="Upstream default 2 (the only value verified with this pack)",
                ),
                io.Int.Input(
                    "max_temporal_ids",
                    default=8,
                    min=P.TEMPORAL_IDS_RANGE[0],
                    max=P.TEMPORAL_IDS_RANGE[1],
                    advanced=True,
                    tooltip="Upstream default 8 (the only value verified with this pack)",
                ),
                io.Combo.Input(
                    "orientation",
                    options=list(P.ORIENTATIONS),
                    default="auto",
                    advanced=True,
                    tooltip="auto = portrait when the input is taller than wide (upstream server rule)",
                ),
                io.Image.Input(
                    "reference_image", optional=True, tooltip="Optional reference image (first image of the batch is used); not verified with this pack"
                ),
            ],
            outputs=[io.Image.Output(display_name="images"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, runtime, images, prompt, seed, num_inference_steps=2, max_temporal_ids=8, orientation="auto", reference_image=None) -> io.NodeOutput:
        import comfy.utils

        if not isinstance(runtime, dict) or "runtime_id" not in runtime:
            raise RuntimeError("JoyAI runtime is not loaded (connect the JoyAI Runtime node)")
        rt = _get_runtime(runtime["runtime_id"])
        try:
            n, h, w = validate_images(images, max_frames=rt.max_frames)
            prompt = validate_prompt(prompt)
            ref = None
            if reference_image is not None:
                validate_images(reference_image[:1], max_frames=1, what="reference_image")
                ref = images_to_u8(reference_image[:1])
        except InputError as exc:
            raise ValueError(str(exc)) from None
        frames = images_to_u8(images)
        orient = orientation_for(orientation, h, w)
        bar = comfy.utils.ProgressBar(n)
        try:
            result, out = MANAGER.edit(
                rt,
                frames,
                prompt=prompt,
                seed=int(seed),
                steps=int(num_inference_steps),
                temporal_ids=int(max_temporal_ids),
                orientation=orient,
                reference=ref,
                interrupted=_interrupted,
                on_progress=lambda done, total: bar.update_absolute(done, total),
            )
        except JobCancelled:
            _raise_interrupt_or("JoyAI job was cancelled")
        except JobTimeout as exc:
            raise RuntimeError(f"JoyAI job timed out: {redact(exc)}") from None
        except (AdmissionRefused, WorkerError) as exc:
            raise RuntimeError(redact(exc)) from None
        images_out = u8_to_images(out)
        report = {
            "frames_in": n,
            "frames_out": int(out.shape[0]),
            "input_size": [h, w],
            "output_size": [int(out.shape[1]), int(out.shape[2])],
            "orientation": orient,
            "worker": result.get("worker"),
            "engine": result.get("engine"),
            "manager": result.get("manager"),
        }
        text = json.dumps(json.loads(redact(json.dumps(report, default=str))), indent=1)
        return io.NodeOutput(images_out, text, ui={"text": [text]})


class JoyAIUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoyAIUnload",
            display_name="JoyAI Unload",
            category=CATEGORY,
            description="Stops the JoyAI worker (frees the GPU and the host memory it holds). Connect 'trigger' to run it after another node.",
            inputs=[
                JoyAIRuntime.Input("runtime", optional=True),
                io.AnyType.Input("trigger", optional=True, tooltip="Optional: any output, only to order this node after it"),
            ],
            outputs=[io.String.Output(display_name="status")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def fingerprint_inputs(cls, runtime=None, trigger=None):
        return math.nan  # always runs

    @classmethod
    def execute(cls, runtime=None, trigger=None) -> io.NodeOutput:
        before = memory.snapshot()
        report = MANAGER.unload("Unload node")
        after = memory.snapshot()
        report["commit_gib_before"] = before.get("commit_gib")
        report["commit_gib_after"] = after.get("commit_gib")
        report["commit_limit_gib"] = after.get("commit_limit_gib")
        text = json.dumps(report, indent=1, default=str)
        return io.NodeOutput(text, ui={"text": [text]})


class JoyAIExtension(ComfyExtension):
    async def get_node_list(self):
        return [JoyAIRuntimeLoader, JoyAIVideoEdit, JoyAIUnload]


async def comfy_entrypoint() -> JoyAIExtension:
    return JoyAIExtension()
