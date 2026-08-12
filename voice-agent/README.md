# voice-agent

A LiveKit voice AI agent (STT → LLM → TTS pipeline) for the candy shop robot. Managed as a `uv` project.

## Stack

All models are served through [LiveKit Inference](https://docs.livekit.io/agents/models/), so no per-provider API keys are required:

| Component      | Model                                        |
| -------------- | -------------------------------------------- |
| STT            | `deepgram/nova-3` (multilingual)             |
| LLM            | `google/gemini-3.6-flash`                    |
| TTS            | `inworld/inworld-tts-2` (voice `Ashley`)     |
| Turn detection | `inference.TurnDetector()`                   |
| Noise removal  | `ai_coustics` QUAIL_VF_S audio enhancement   |

## Setup

```shell
uv sync
```

Provide LiveKit credentials in `.env.local` (see `.env.example`). Let the LiveKit CLI write them:

```shell
lk cloud auth
```

Or copy the template and fill it in from your [LiveKit Cloud project settings](https://cloud.livekit.io):

```shell
cp .env.example .env.local
```

```
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=<your-api-key>
LIVEKIT_API_SECRET=<your-api-secret>
```

## Run

Talk to the agent directly in your terminal:

```shell
lk agent console
```

Development mode (reloads on file change, for a frontend or telephony client):

```shell
uv run voice-agent dev
```

Production mode:

```shell
uv run voice-agent start
```

Pre-download model weights (useful in Docker builds):

```shell
uv run voice-agent download-files
```

## Development

```shell
uv run ruff check src/       # lint
uv run ruff format src/      # format
uv run pytest                # tests
```
