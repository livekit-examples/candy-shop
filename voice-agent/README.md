# voice-agent

A LiveKit voice AI agent built on the STT → LLM → TTS pipeline, following the
[LiveKit voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai/).

Managed as a `uv` project.

## Stack

All models are served through [LiveKit Inference](https://docs.livekit.io/agents/models/), so
no per-provider API keys are required:

| Component      | Model                                        |
| -------------- | -------------------------------------------- |
| STT            | `deepgram/nova-3` (multilingual)             |
| LLM            | `google/gemma-4-31b-it`                      |
| TTS            | `cartesia/sonic-3`                           |
| Turn detection | `inference.TurnDetector()`                   |
| Noise removal  | `ai_coustics` QUAIL_VF_S audio enhancement   |

## Setup

Install dependencies:

```shell
uv sync
```

Provide LiveKit credentials in `.env.local` (see `.env.example` for the shape). The
easiest path is to let the LiveKit CLI write them for you:

```shell
lk cloud auth
```

Or copy the template and fill it in manually from your
[LiveKit Cloud project settings](https://cloud.livekit.io):

```shell
cp .env.example .env.local
```

```
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=<your-api-key>
LIVEKIT_API_SECRET=<your-api-secret>
```

## Run

Talk to the agent directly in your terminal — no frontend needed:

```shell
lk agent console
```

> The agent's own `uv run voice-agent console` still works but prints a deprecation
> warning on `livekit-agents` 1.6.9; `lk agent console` is the supported path.

Development mode, so a frontend or telephony client can connect (reloads on file change):

```shell
uv run voice-agent dev
```

Production mode:

```shell
uv run voice-agent start
```

Pre-download any model weights the plugins need (useful in Docker builds):

```shell
uv run voice-agent download-files
```

## Connect a frontend

`console` mode is enough to verify the agent talks. To use a real client, start the agent
in `dev` mode and connect one of the LiveKit
[starter frontends](https://docs.livekit.io/agents/start/frontend/) — web, iOS, Android,
Flutter, or React Native — or wire up
[telephony](https://docs.livekit.io/agents/start/telephony/) for a phone number.

## Layout

```
src/voice_agent/
  __init__.py    # exports main()
  agent.py       # Assistant definition + AgentServer session handler
```

The agent's persona and voice output rules live in the `instructions` string in
`Assistant.__init__` — edit that to change its behavior.

## Development

```shell
uv run ruff check src/       # lint
uv run ruff format src/      # format
uv run pytest                # tests
```
