# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A monorepo containing two Python AI/ML microservices and Docker Compose infrastructure for various self-hosted services. All Python projects use **Poetry** and target **Python 3.12+**.

## Commands

### Docker Compose (from repo root)

```bash
make init                              # Start Portainer + create ingress network
make down                              # Tear down Portainer + ingress network
STACK_NAME=<name> make service-up      # Start a service stack (requires .env)
STACK_NAME=<name> make service-build   # Build a service stack
STACK_NAME=<name> make service-down    # Stop a service stack
STACK_NAME=<name> make service-logs    # View logs
STACK_NAME=<name> make service-restart # Restart a service stack
```

Stack names correspond to YAML files: `ai-agent`, `subtitle-generator`, `network`, `n8n`, `nocodb`, etc.

### Subtitle Generator

```bash
cd src/subtitle_generator
poetry install
poetry run python app.py               # Starts Gradio UI on port 7860
```

### Korean Teacher Agent

```bash
cd src/korean-teacher-agent
poetry install
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
poetry run alembic upgrade head        # Run database migrations
```

### No test or lint tooling is configured in either project.

## Architecture

### Subtitle Generator (`src/subtitle_generator/`)

Gradio web app with four tabs: translate subtitles, generate from audio, edit translations, compare subtitles.

- **app.py** — Gradio UI entry point, wires together all processing modules
- **src/generate_subtitle.py** — Audio transcription via OpenAI Whisper API with concurrent chunk processing
- **src/translate_subtitle.py** — SRT translation using OpenAI GPT. Processes subtitles in blocks (`BLOCK_SIZE`, default 100) with a sliding window context (`SUB_WINDOW_RADIUS`) for quality
- **src/vad_onnx.py** — Voice Activity Detection using ONNX Runtime (`models/model.onnx`) to identify speech segments before transcription
- **src/subtitle_utils.py** — CJK-aware text splitting, overlap detection/removal between chunks, proportional timing

Key design: async pipeline. Audio → VAD segmentation → Whisper transcription → block-based translation with context windows.

### Korean Teacher Agent (`src/korean-teacher-agent/`)

FastAPI REST service using LangGraph for multi-agent orchestration.

- **src/main.py** — FastAPI app. Key endpoints: `POST /writing-homework` (async, returns 202), `POST /send-writing-homework`
- **src/agents/** — LangGraph state-machine agents: `writing_homework_agent.py` (homework generation), `notion_agent.py` (Notion page evaluation + commenting), `homework_feedback_agent.py`
- **src/prompts/prompt_manager.py** — Singleton that fetches prompts from Arize Phoenix (not hardcoded), caches locally, supports SIGUSR1 refresh
- **src/database/** — Async SQLAlchemy with SQLite (aiosqlite). Alembic for migrations (`alembic.ini` at project root)
- **src/tools/notion_tools.py** — LangGraph tools for Notion API operations
- **src/services/** — Background task handling and external service integration (Heartbeat)

Key design: prompts are externalized in Phoenix, agents use Pydantic structured outputs, homework generation runs as a background task.

## Environment Variables

Both services require `OPENAI_API_KEY`. The Korean Teacher Agent additionally needs `NOTION_TOKEN` and optionally `PHOENIX_COLLECTOR_ENDPOINT` / `PHOENIX_API_KEY` for observability. The root `env` file is a template for infrastructure variables.
