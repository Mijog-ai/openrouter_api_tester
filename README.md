# OpenRouter API Tester

A PyQt6 desktop app that loads **every model available on
[OpenRouter](https://openrouter.ai)**, groups them by type, and gives you an
**adaptive playground** and a **batch tester** to exercise any of them through
the OpenRouter API.

Audio models are intentionally excluded — every other model type (text,
vision/multimodal, image generation, document/file) is supported.

## Features

- **Bundled preload** — the full model list ships in
  `openrouter_tester/data/models.json`, so the app is populated **instantly on
  launch (even offline)**. It then refreshes from the live
  `GET /api/v1/models` in the background and rewrites the preload.
- **Categorised catalog** — each model is sorted by its input/output
  modalities:
  - **Text** — text-in / text-out chat models
  - **Vision / Multimodal** — accept image or video input
  - **Image Generation** — produce images as output
  - **Document / File** — accept file/PDF input
- **Audio filtered out** — any model with audio input or output is dropped.
- **Adaptive playground** — the chat panel reshapes itself to the selected
  model:
  - **Text** → plain streaming chat.
  - **Vision/Multimodal** → an **Attach image** button; the image is sent
    inline as a base64 data URL and shown in the transcript.
  - **Document/File** → an **Attach file** button (PDF, txt, csv, json…) sent
    as an OpenRouter `file` content part.
  - **Image Generation** → the button becomes **Generate**; the request asks
    for image output (`modalities: ["image","text"]`) and returned images are
    **rendered inline** in the transcript.
- **Batch Test tab** — pick a category (or *All*), cap how many models to hit,
  and run one small prompt against each. Results stream into a table with
  **pass/fail, latency, and a response snippet** (image models are checked for
  a returned image). Runs sequentially to stay rate-limit friendly, and can be
  stopped mid-run.
- **Search & filter** — filter by name/id, or show only free models.
- **Rich model info** — hover a model to see its id, context length, pricing,
  and description.

## Setup

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide your OpenRouter API key (optional — can also be typed in the UI)
export OPENROUTER_API_KEY="sk-or-..."   # Windows: set OPENROUTER_API_KEY=...

# 4. Run
python main.py
```

Get an API key at https://openrouter.ai/keys.

> The model list loads without a key (from the preload or the public models
> endpoint), but running a chat or a batch test requires one.

### Refreshing the bundled preload

The app updates the preload automatically after each successful live refresh.
To regenerate it manually (e.g. to commit an updated snapshot):

```bash
python scripts/fetch_models.py
```

## Project layout

```
main.py                       # entry point
scripts/fetch_models.py       # regenerate the bundled model preload
openrouter_tester/
├── client.py                 # OpenRouter REST client (models, streaming + non-streaming chat)
├── catalog.py                # model normalisation, categorisation, preload cache
├── workers.py                # QThread workers (fetch, streaming chat, image-gen, batch test)
├── data/models.json          # bundled model preload
└── ui/
    ├── main_window.py        # tabs, model browser, preload/refresh wiring
    ├── chat_widget.py        # adaptive chat playground (+ inline image rendering)
    └── batch_test_widget.py  # batch tester table
```

## How categorisation works

Each OpenRouter model reports its `architecture.input_modalities` and
`architecture.output_modalities` (e.g. `["text", "image"]`). The app assigns a
single category using this priority:

1. Outputs an image → **Image Generation**
2. Accepts image/video input → **Vision / Multimodal**
3. Accepts file input → **Document / File**
4. Otherwise → **Text**

Any model whose input *or* output modalities include `audio` is excluded.

## Notes

- Streaming chats use the OpenAI-compatible
  `POST /api/v1/chat/completions` endpoint with `stream: true`; image
  generation and batch probes use the same endpoint non-streamed.
- Network calls run on background `QThread`s so the UI stays responsive, and
  in-flight streaming generations can be cancelled with **Stop**.
- Your API key is only held in memory / read from the environment; it is never
  written to disk by the app.
- The **Max models** cap in the Batch Test tab exists to protect your credits —
  raise it deliberately before probing hundreds of paid models.
