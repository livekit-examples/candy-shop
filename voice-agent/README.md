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

## Deploy as a hosted agent

LiveKit Cloud builds the image from the `Dockerfile` here and runs `voice-agent
start` on its own infrastructure. Run all of these from this directory — the
build context is `voice-agent/`, not the repo root.

Already done once — the agent is `CA_eGcynJ2axibZ` in the `robotics-examples`
project, region `us-east`, recorded in `livekit.toml`. To stand up another one:

```shell
lk agent create --project <livekit-cloud-project> --region us-east .
```

`--region` is mandatory whenever stdin isn't a terminal, and only `us-east`
(Virginia), `eu-central` (Frankfurt), and `ap-south` (Mumbai) exist. It is fixed
for the life of the agent — moving means delete and recreate.

Afterwards:

```shell
lk agent deploy     # build and roll out a new version
lk agent status     # replicas, CPU, memory
lk agent logs       # tail runtime logs (--log-type=build for the image build)
lk agent rollback   # back to the previous version
```

**No secrets to upload.** Every model is served through LiveKit Inference, and
`LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` are injected by the
platform. Do not point `--secrets-file` at the repo-root `.env`: it is full of
rig-only config (serial ports, checkpoint paths) that the agent never reads.

**The rig has to live in the same LiveKit project.** A hosted agent joins rooms
in the Cloud project it was deployed to and reaches the robot and the operators
by participant identity over RPC, so it cannot talk to a rig registered against
a self-hosted server. Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and
`LIVEKIT_API_SECRET` in the repo-root `.env` to that Cloud project before
starting `robot` and the operators.

**Dispatch is explicit.** The worker registers under
`agent_name="candy-shop-assistant"`, so it is not auto-assigned to every room.
Either name it in the room configuration of the token your frontend mints, or
dispatch it by hand:

```shell
lk dispatch create --room candy-shop --agent-name candy-shop-assistant
```

## Development

```shell
uv run ruff check src/       # lint
uv run ruff format src/      # format
uv run pytest                # tests
```
