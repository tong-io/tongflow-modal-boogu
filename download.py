"""Modal download entry for Boogu-Image-0.1.

Run:
  modal run download.py::download

Pulls both checkpoints used by deploy.py (Base for image_gen, Edit for
image_edit) into the shared ``models`` volume. Self-contained: Modal remote
execution may mount only this file, so do not import other local modules.
"""

from __future__ import annotations

import os
from typing import Any

import modal


_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}

# Keep these in sync with deploy.py (fp8 by default; drop the suffix for bf16).
GEN_REPO_ID = str(_hf.get("genRepoId") or "Boogu/Boogu-Image-0.1-Base-fp8")
EDIT_REPO_ID = str(_hf.get("editRepoId") or "Boogu/Boogu-Image-0.1-Edit-fp8")
REPO_IDS = [GEN_REPO_ID, EDIT_REPO_ID]

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "huggingface_hub==1.6.0",
    ),
    volumes={"/models": volume},
    timeout=3600,
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    for repo_id in REPO_IDS:
        model_dir = f"/models/{repo_id}"
        if os.path.exists(model_dir) and os.listdir(model_dir):
            print(f"Model already exists at {model_dir}, skipping")
            continue
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"Model downloaded to {model_dir}")
    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
