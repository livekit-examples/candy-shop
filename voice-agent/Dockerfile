# syntax=docker/dockerfile:1
#
# Image for the LiveKit Cloud hosted agent. The build context is this directory,
# not the repo root: the voice agent is a self-contained uv project, and nothing
# else in candy-shop (robot, operators, datasets) belongs in the container.
#
# Constraints LiveKit Cloud imposes on the image: a glibc base (no Alpine), a
# non-root user, and a fixed CMD that runs the worker's `start` command. No
# LIVEKIT_* variables and no .env files are baked in — the platform injects
# credentials and secrets as environment variables at runtime.

ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Unbuffered so a crash still flushes its logs; bytecode precompiled at install
# time so the first import doesn't pay for it during a cold start.
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

FROM base AS build

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, without the project itself: this is a src layout, and
# uv_build refuses to build voice_agent before src/voice_agent exists. Keeping
# the layer independent of the source also means a code edit doesn't re-resolve
# the whole lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# The hook plugins use to fetch model weights, run before the source copy so it
# caches with the dependency layer. A no-op today (the ai-coustics enhancer
# ships its weights in the wheel, STT/LLM/TTS/turn detection are all served by
# LiveKit Inference), but a plugin with local weights must not download them
# on the first job — cold start has a five-minute health check budget.
RUN uv run --no-sync --module livekit.agents download-files

COPY . .
RUN uv sync --locked --no-dev

FROM base

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

WORKDIR /app

# One layer, correct ownership, and the build stage's compilers stay behind.
COPY --from=build --chown=appuser:appuser /app /app

USER appuser

# The venv's console script directly rather than `uv run`, which would want a
# writable cache and could try to re-sync against the lockfile at startup.
ENV PATH="/app/.venv/bin:$PATH"
CMD ["voice-agent", "start"]
