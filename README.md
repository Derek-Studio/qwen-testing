# qwen-testing

Test scripts for running **Qwen3-4B** locally via [Ollama](https://ollama.com). No GPU required, no cloud, no API keys — everything runs on your machine.

## Quick start

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model (~2.6GB quantized)
ollama pull qwen3:4b

# Run a basic test
python3 test_basic.py

# Interactive chat
python3 test_chat.py
```

## Scripts

| Script | Description |
|---|---|
| `test_basic.py` | Sends predefined prompts via `ollama run`, prints responses |
| `test_chat.py` | Interactive multi-turn chat loop |
| `test_api.py` | Streaming via Ollama's local REST API (requires `ollama serve`) |

## System requirements

- ~2.6GB disk for the model
- ~3GB RAM at inference time
- No GPU needed
- Python 3.x stdlib only (no pip)
