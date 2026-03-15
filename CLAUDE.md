# qwen-testing

Local testing project for Qwen3-0.6B running via Ollama. All inference happens on-machine — no cloud, no API keys.

## What this is

This repo contains test scripts for Qwen3-0.6B (Alibaba's tiny 0.6B parameter model) running locally through Ollama. It's designed for environments with no GPU and limited RAM (~4GB), using the quantized model (~400MB).

## Requirements

- [Ollama](https://ollama.com) installed
- `qwen3:0.6b` model pulled
- Python 3.x (stdlib only — no pip dependencies)

## Setup

### Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull the model
```bash
ollama pull qwen3:0.6b
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
