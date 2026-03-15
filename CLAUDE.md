# qwen-testing

Local testing project for Qwen3-0.6B running via Ollama. All inference happens on-machine — no cloud, no API keys.

## What this is

This repo contains test scripts for Qwen3-0.6B (Alibaba's tiny 0.6B parameter model) running locally through Ollama. It's designed for environments with no GPU and limited RAM (~4GB), using the quantized model (~400MB).

## Requirements

- [Ollama](https://ollama.com) installed
- `qwen3:4b` model pulled
- Python 3.x (stdlib only — no pip dependencies)

## Setup

### Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull the model
```bash
ollama pull qwen3:4b
```

### Start Ollama server (for API-based scripts)
```bash
ollama serve &
```

## Running the tests

### Basic prompt test (subprocess CLI)
```bash
python3 test_basic.py
```
Sends a few prompts to the model via `ollama run` and prints responses.

### Multi-turn chat test (subprocess CLI)
```bash
python3 test_chat.py
```
Interactive REPL — type messages, get responses. Type `quit` to exit.

### API streaming test (requires ollama serve)
```bash
python3 test_api.py
```
Uses Ollama's local REST API at `http://localhost:11434` with streaming.

## Branch structure

- `main` — stable releases
- `dev` — active development (default working branch)

---

## Research Agent

`research_agent.py` is a headless ReAct-style research agent that:
1. Searches DuckDuckGo for the query
2. Asks Qwen to pick the best result
3. Browses the page with headless Chromium
4. Evaluates whether the page answers the query (follows links if not, up to 3 hops)
5. Synthesizes a concise answer using the local model

### Additional dependencies

This script requires pip packages (first use of external dependencies in this project):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium
```

### Usage

```bash
# Basic usage
.venv/bin/python3 research_agent.py "what promotions are taking place at the Goose, fulham pub"

# With JSON schema — returns validated JSON matching the schema shape
.venv/bin/python3 research_agent.py "find events at the Goose Fulham" \
  --schema '[{"description": "", "dates": ""}]'

# Schema from a file
.venv/bin/python3 research_agent.py "find events at the Goose Fulham" \
  --schema ./my_schema.json

# Interactive (no query argument)
.venv/bin/python3 research_agent.py
```

### `--schema` flag

Pass a JSON schema as an inline string or a path to a `.json` file. The agent will:
1. Instruct the LLM to reply with JSON matching the schema
2. Validate the response with `jsonschema`
3. Retry up to 3 times on validation failure, feeding the error back to the model
4. Fall back gracefully if all retries fail

The schema can be a plain example instance (e.g. `[{"description": "", "dates": ""}]`) — it will be coerced into a proper JSON Schema automatically.

### Model constraints

Uses `qwen3:1.7b` — each LLM call is stateless (no cross-step history) to avoid context overflow. Max raw page content per browse: 8000 chars. LLM-extracted relevant content per page: 2000 chars. Max synthesis input: 4000 chars.
