# tongflow-modal-boogu

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text-to-image generation and single-reference image editing with **[Boogu-Image-0.1](https://github.com/boogu-project/Boogu-Image)** (Apache-2.0, 10B unified understanding-and-generation model), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Image generation** (`image-gen`) — text-to-image with **Boogu-Image-0.1-Base**. Strongest on dense, layout-heavy bilingual (Chinese/English) text rendering — posters, documents, brand guides. Supports up to 2K.
- **Image editing** (`image-edit`) — single-reference instruction editing with **Boogu-Image-0.1-Edit** (object insert/replace/remove, attribute/background changes, in-image text editing).

Each slot runs as its own `@deploy` class loading its own checkpoint, so they scale independently and never co-reside on one GPU.

> For fast/photorealistic general text-to-image, the lighter sibling **Z-Image-Turbo** (`tongflow-modal-z-image`, 6B) is usually the cheaper default. Boogu's differentiated value here is dense text rendering (Base) and single-image editing (Edit).

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

Boogu weights (`Boogu/Boogu-Image-0.1-Base`, `Boogu/Boogu-Image-0.1-Edit`) are public — no `HF_TOKEN` required.

## Cost / VRAM

Defaults to the published **fp8** checkpoints (`…-Base-fp8` / `…-Edit-fp8`) on **A100-40GB** — fp8 roughly halves resident VRAM versus bf16. Validate the fp8 load path on first deploy; if it errors, swap `GEN_REPO_ID` / `EDIT_REPO_ID` (in `deploy.py` **and** `download.py`) to the bf16 repos (drop the `-fp8` suffix) and bump the GPU to **A100-80GB**.

## Deploy

```bash
modal run download.py::download   # pre-fetch both checkpoints into the `models` volume
modal deploy deploy.py
```

The TongFlow runtime does the above automatically via `entry.py`.
