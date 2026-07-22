"""Modal deploy entry for Boogu-Image-0.1.

Boogu-Image is an Apache-2.0 unified image generation + editing model family
(https://github.com/boogu-project/Boogu-Image). This plugin exposes two ABI
slots backed by two *separate* checkpoints, each in its own ``@deploy`` class so
they scale independently and never sit co-resident on one GPU:

  - ``image_gen``  (T2I)  -> Boogu-Image-0.1-Base   — dense bilingual text rendering
  - ``image_edit`` (TI2I) -> Boogu-Image-0.1-Edit   — single-reference image editing

The upstream ``boogu`` package is not on PyPI, so the image ``git clone``s the
repo and ``pip install -e``'s it (same approach as the flux2 plugin).

Deploy:
  modal deploy deploy.py

Pre-fetch weights:
  modal run download.py::download
"""

from __future__ import annotations

import io
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Optional

import modal
from tongflow import deploy
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot


_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}

# Checkpoints — default to the published ``*-fp8`` repos to roughly halve
# resident VRAM (fits A100-40GB). The fp8 load path goes through the
# checkpoint's own trust_remote_code loader and should be validated on first
# deploy; if it errors, swap to the bf16 repos (drop the ``-fp8`` suffix) and
# bump the GPU tier to A100-80GB below.
GEN_REPO_ID = str(_hf.get("genRepoId") or "Boogu/Boogu-Image-0.1-Base-fp8")
EDIT_REPO_ID = str(_hf.get("editRepoId") or "Boogu/Boogu-Image-0.1-Edit-fp8")
GEN_MODEL_DIR = f"/models/{GEN_REPO_ID}"
EDIT_MODEL_DIR = f"/models/{EDIT_REPO_ID}"

# Upstream inference source (provides the `boogu` python package).
BOOGU_GIT = str(_hf.get("boatGit") or "https://github.com/boogu-project/Boogu-Image.git")
BOOGU_DIR = "/opt/boogu"

# Diffusion sampling defaults — plugin-internal, not part of the ABI contract.
# Base/Edit are 25~50 step models; CFG per the model zoo (Base ~4.0, Edit ~5.0).
DEFAULT_NUM_INFERENCE_STEPS = 50
GEN_TEXT_GUIDANCE_SCALE = 4.0
EDIT_TEXT_GUIDANCE_SCALE = 5.0
EDIT_IMAGE_GUIDANCE_SCALE = 1.0
# Native model ceiling is 2K; Boogu resizes the generation latent to honour
# these caps, so keep them aligned with the requested output size.
MAX_INPUT_SIDE = 2048

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)


# ── app / image ───────────────────────────────────────────────────────────────

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    # *-devel (not runtime): Boogu pulls Triton / torchao kernels that JIT-compile
    # and need a C compiler. Same choice as the flux2 plugin.
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    .apt_install("git", "build-essential")
    .run_commands(
        f"git clone --depth 1 {BOOGU_GIT} {BOOGU_DIR}",
        f"cd {BOOGU_DIR} && pip install -e .",
        # flash-attn: Boogu uses it; this helper auto-fetches a matching wheel and
        # falls back to a source build. Best-effort — `|| true` so a wheel miss
        # doesn't fail the whole image build.
        f"cd {BOOGU_DIR} && python utils/get_flash_attn.py || true",
    )
    .pip_install("tongflow==0.2.13", "fastapi[standard]")
    .env(
        {
            "device": "cuda:0",  # required by the Boogu pipeline
            "HF_HOME": "/models/hf",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

with image.imports():
    import torch
    from PIL import Image
    from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline


def _pil_to_png_bytes(img: Any) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── image_gen (T2I) — Boogu-Image-0.1-Base ────────────────────────────────────


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="A100-40GB",  # fp8 10B + Qwen3-VL mllm; use A100-80GB if running bf16 repos at 2K
    volumes={"/models": volume},
    timeout=1200,
)
class GenInference:
    @modal.enter()
    def load(self):
        self.pipe = BooguImagePipeline.from_pretrained(
            GEN_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.pipe.to("cuda:0")

    def _t2i_png_bytes(
        self, prompt: str, width: int, height: int, seed: int
    ) -> bytes:
        # Boogu wants 16-aligned output sizes.
        height -= height % 16
        width -= width % 16
        out = self.pipe(
            instruction=prompt,
            negative_instruction="",
            height=height,
            width=width,
            max_input_image_pixels=height * width,
            max_input_image_side_length=2 * max(height, width),
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            text_guidance_scale=GEN_TEXT_GUIDANCE_SCALE,
            generator=torch.Generator("cuda:0").manual_seed(seed),
        ).images[0]
        return _pil_to_png_bytes(out)

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN)
    def image_gen(self, input: ImageGenInput) -> ImageGenOutput:
        text = (input.text or "").strip()
        if not text:
            return ImageGenOutput(success=False, error="Missing text prompt")
        raw = self._t2i_png_bytes(
            prompt=text,
            width=input.width if input.width is not None else 1024,
            height=input.height if input.height is not None else 1024,
            seed=int(input.seed) if input.seed is not None else 0,
        )
        return ImageGenOutput(success=True, image=asset(raw, mime="image/png"))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-geninference-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )



# ── image_edit (TI2I) — Boogu-Image-0.1-Edit ──────────────────────────────────


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="A100-40GB",  # fp8 10B + Qwen3-VL mllm; use A100-80GB if running bf16 repos at 2K
    volumes={"/models": volume},
    timeout=1200,
)
class EditInference:
    @modal.enter()
    def load(self):
        self.pipe = BooguImagePipeline.from_pretrained(
            EDIT_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.pipe.to("cuda:0")

    def _edit_png_bytes(
        self,
        prompt: str,
        image_bytes: bytes,
        seed: int,
        match_input_size: bool,
        width: Optional[int],
        height: Optional[int],
    ) -> bytes:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # The Boogu TI2I path takes file *paths* alongside the loaded images.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil.save(tmp, format="PNG")
            ref_path = tmp.name

        # align_res=True + batch_size=1 => output follows the reference image and
        # height/width may stay None. When the caller pins a size, honour it.
        if match_input_size:
            out_h, out_w, align_res = None, None, True
        else:
            out_h = (height - height % 16) if height is not None else None
            out_w = (width - width % 16) if width is not None else None
            align_res = False

        try:
            out = self.pipe(
                instruction=[prompt],
                input_image_paths=[[ref_path]],
                input_images=[[pil]],
                negative_instruction="",
                height=out_h,
                width=out_w,
                max_input_image_pixels=MAX_INPUT_SIDE * MAX_INPUT_SIDE,
                max_input_image_side_length=MAX_INPUT_SIDE * 2,
                align_res=align_res,
                num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                text_guidance_scale=EDIT_TEXT_GUIDANCE_SCALE,
                image_guidance_scale=EDIT_IMAGE_GUIDANCE_SCALE,
                generator=torch.Generator("cuda:0").manual_seed(seed),
            ).images[0]
        finally:
            try:
                os.unlink(ref_path)
            except OSError:
                pass
        return _pil_to_png_bytes(out)

    @modal.method()
    @node_slot(NodeSlots.IMAGE_EDIT)
    def image_edit(self, input: ImageEditInput) -> ImageEditOutput:
        if input.image is None:
            return ImageEditOutput(success=False, error="Missing image")
        raw = self._edit_png_bytes(
            prompt=input.text or "",
            image_bytes=prompt_media_to_bytes(input.image),
            seed=int(input.seed) if input.seed is not None else 0,
            match_input_size=(
                input.match_input_size if input.match_input_size is not None else True
            ),
            width=input.width,
            height=input.height,
        )
        return ImageEditOutput(success=True, image=asset(raw, mime="image/png"))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-editinference-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

