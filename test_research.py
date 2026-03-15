#!/usr/bin/env python3
"""
Manual validation script for research_agent.py.

Runs the agent against a fixed query and schema, saving:
  outputs/<timestamp>_debug.log  — full print trace (terminal + file)
  outputs/<timestamp>_output.json — final JSON answer

Inspect the output files to verify the agent is working correctly.
"""

import json
import os
import sys
from datetime import datetime

from research_agent import (
    BASE_URL,
    MODEL,
    BrowserTool,
    OllamaClient,
    ResearchAgent,
    SearchTool,
    _coerce_to_schema,
)

QUERY = "find the events at the goose fulham"
SCHEMA_EXAMPLE = [{"description": "", "dates": ""}]


class Tee:
    def __init__(self, file):
        self._file = file
        self._stdout = sys.__stdout__

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)

    log_path = f"outputs/{ts}_debug.log"
    out_path = f"outputs/{ts}_output.json"

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(log_file)

    schema = _coerce_to_schema(SCHEMA_EXAMPLE)

    llm = OllamaClient()
    if not llm.is_available():
        print("Error: Ollama server not running. Start it with: ollama serve &")
        sys.stdout = sys.__stdout__
        log_file.close()
        sys.exit(1)

    print(f"\nModel : {MODEL}")
    print(f"Query : {QUERY}")
    print(f"Schema: {json.dumps(SCHEMA_EXAMPLE)}")
    print("=" * 60)

    agent = ResearchAgent(llm=llm, search=SearchTool(), browser=BrowserTool(), schema=schema)
    answer = agent.run(QUERY)

    print("=" * 60)
    print(answer)
    print("=" * 60)

    sys.stdout = sys.__stdout__
    log_file.close()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer)

    print(f"\nOutput files:")
    print(f"  {out_path}")
    print(f"  {log_path}")


if __name__ == "__main__":
    main()
