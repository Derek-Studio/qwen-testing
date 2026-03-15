#!/usr/bin/env python3
"""
Basic prompt/response test for Qwen3-0.6B via Ollama CLI.
Uses subprocess to call `ollama run` — no server or network needed.
"""

import subprocess
import sys

MODEL = "qwen3:1.7b"

PROMPTS = [
    "What is 2 + 2? Answer in one sentence.",
    "Write a haiku about a mountain.",
    "Name three programming languages and one word that describes each.",
]


def run_prompt(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", MODEL, prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    print(f"Testing model: {MODEL}\n{'=' * 50}")
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}] Prompt: {prompt}")
        print("-" * 40)
        response = run_prompt(prompt)
        print(f"Response: {response}")
    print("\n" + "=" * 50)
    print("All tests passed.")


if __name__ == "__main__":
    main()
