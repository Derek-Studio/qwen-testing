#!/usr/bin/env python3
"""
Interactive multi-turn chat test for Qwen3-0.6B via Ollama CLI.
Each turn calls `ollama run` with the full conversation history
formatted as a prompt so the model has context.

Type 'quit' or press Ctrl-C to exit.
"""

import subprocess
import sys

MODEL = "qwen3:1.7b"


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


def build_prompt(history: list[tuple[str, str]], user_input: str) -> str:
    """Format conversation history + new input as a single prompt string."""
    lines = []
    for user_msg, assistant_msg in history:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    lines.append(f"User: {user_input}")
    lines.append("Assistant:")
    return "\n".join(lines)


def main():
    print(f"Qwen3-0.6B Chat (model: {MODEL})")
    print("Type 'quit' to exit.\n")

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        prompt = build_prompt(history, user_input)
        print("Assistant: ", end="", flush=True)
        response = run_prompt(prompt)
        print(response)

        history.append((user_input, response))


if __name__ == "__main__":
    main()
