#!/usr/bin/env python3
"""
Streaming API test for Qwen3-0.6B via Ollama's local REST API.
Requires `ollama serve` to be running on http://localhost:11434.

Uses only Python stdlib (urllib + json) — no pip dependencies.
"""

import json
import sys
import urllib.request
from urllib.error import URLError

MODEL = "qwen3:1.7b"
BASE_URL = "http://localhost:11434"


def check_server() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=3) as resp:
            return resp.status == 200
    except URLError:
        return False


def stream_generate(prompt: str) -> str:
    payload = json.dumps({"model": MODEL, "prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    full_response = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            chunk = json.loads(line.decode())
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            full_response.append(token)
            if chunk.get("done"):
                break
    print()  # newline after streaming
    return "".join(full_response)


def main():
    if not check_server():
        print("Error: Ollama server not running. Start it with: ollama serve &", file=sys.stderr)
        sys.exit(1)

    prompts = [
        "Explain what a large language model is in two sentences.",
        "What are the planets in our solar system?",
    ]

    print(f"Testing streaming API with model: {MODEL}\n{'=' * 50}")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] Prompt: {prompt}")
        print("-" * 40)
        print("Response: ", end="")
        stream_generate(prompt)

    print("\n" + "=" * 50)
    print("Streaming tests complete.")


if __name__ == "__main__":
    main()
