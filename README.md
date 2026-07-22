# OpenRouter API Tester

A PyQt6 desktop app that loads **every model available on
[OpenRouter](https://openrouter.ai)**, groups them by type, and gives you a
chat playground to test any of them through the OpenRouter API.

Audio models are intentionally excluded — every other model type (text,
vision/multimodal, image generation, document/file) is supported.

## Features

- **Live model catalog** — pulls the full model list from
  `GET /api/v1/models` and sorts each model into a category based on its
  input/output modalities:
  - **Text** — text-in / text-out chat models
  - **Vision / Multimodal** — accept image or video input
  - **Image Generation** — produce images as output
  - **Document / File** — accept file/PDF input
- **Audio filtered out** — any model with audio input or output is dropped.
- **Search & filter** — filter by name/id, or show only free models.
- **Chat playground** — pick a model and chat with it. Responses stream token
  by token; temperature and max-tokens are adjustable.
- **Image attachments** — for vision-capable models, attach an image that is
  sent inline (base64 data URL) with your next message.
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

> The model list loads without a key, but running a chat requires one.

## Project layout

```
main.py                       # entry point
openrouter_tester/
├── client.py                 # OpenRouter REST client (models + streaming chat)
├── catalog.py                # model normalisation + categorisation (audio excluded)
├── workers.py                # QThread workers (non-blocking network I/O)
└── ui/
    ├── main_window.py        # model browser + top bar wiring
    └── chat_widget.py        # chat playground panel
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

- Chat requests use OpenRouter's OpenAI-compatible
  `POST /api/v1/chat/completions` endpoint with `stream: true`.
- Network calls run on background `QThread`s so the UI stays responsive, and
  in-flight generations can be cancelled with **Stop**.
- Your API key is only held in memory / read from the environment; it is never
  written to disk by the app.
