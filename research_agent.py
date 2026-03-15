#!/usr/bin/env python3
"""
ReAct-style research agent for Qwen3-0.6B via Ollama.
Searches the web with DuckDuckGo, browses pages with headless Chromium,
and synthesizes an answer using the local model.

Requires: pip install -r requirements.txt && playwright install chromium
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.error import URLError

from ddgs import DDGS
from playwright.sync_api import sync_playwright

try:
    import jsonschema
    _JSONSCHEMA_AVAILABLE = True
except ImportError:
    _JSONSCHEMA_AVAILABLE = False

MODEL = "qwen3:1.7b"
BASE_URL = "http://localhost:11434"
MAX_SEARCH_RESULTS = 5
MAX_BROWSE_ITERATIONS = 3
MAX_PAGE_CHARS = 8000
EXTRACT_CHARS = 2000
MAX_SYNTHESIS_CHARS = 4000
MAX_LINKS_TO_SHOW = 10
OLLAMA_TIMEOUT = 120


def _coerce_to_schema(raw):
    if isinstance(raw, dict) and ("type" in raw or "$schema" in raw):
        return raw  # already a proper schema
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        props = {k: {"type": "string"} for k in raw[0]}
        return {"type": "array", "items": {"type": "object", "properties": props}}
    return {"type": "object"}


class OllamaClient:
    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=3) as resp:
                return resp.status == 200
        except URLError:
            return False

    def chat(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        content = data["message"]["content"]
        # Strip <think>...</think> blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.strip()


class SearchTool:
    def search(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    })
            return results
        except Exception:
            return []


class BrowserTool:
    def fetch_page(self, url: str) -> dict:
        browser = None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 900},
                )
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                text = page.inner_text("body")

                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass  # fall through silently on timeout

                # Scroll to trigger lazy-loaded content
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1500)

                text_after = page.inner_text("body")
                text = text_after if len(text_after) > len(text) else text
                text = text[:MAX_PAGE_CHARS]

                raw_links = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => ({text: e.innerText.trim(), href: e.getAttribute('href')}))"
                )
                links = []
                seen = set()
                for link in raw_links:
                    href = link.get("href", "")
                    if not href or href.startswith("#") or href.startswith("javascript:"):
                        continue
                    abs_url = urllib.parse.urljoin(url, href)
                    if abs_url not in seen:
                        seen.add(abs_url)
                        links.append({"text": link.get("text", "")[:80], "url": abs_url})
                    if len(links) >= MAX_LINKS_TO_SHOW:
                        break

                browser.close()
                return {"url": url, "text": text, "links": links}
        except Exception as e:
            if browser:
                try:
                    browser.close()
                except Exception:
                    pass
            return {"url": url, "text": f"Error fetching page: {e}", "links": []}


class ResearchAgent:
    def __init__(self, llm: OllamaClient, search: SearchTool, browser: BrowserTool, schema=None):
        self.llm = llm
        self.search = search
        self.browser = browser
        self.schema = schema

    def _extract_relevant(self, query: str, page_text: str, page_url: str) -> str:
        system = (
            "You are a research assistant extracting relevant passages from a web page.\n"
            "Reply with ONLY the relevant text copied or lightly paraphrased from the page.\n"
            "If the page contains NO information relevant to the query, reply with exactly: NOTHING_RELEVANT\n"
            "Do not add commentary, headers, or explanation."
        )
        user = (
            f"Query: {query}\n"
            f"Page URL: {page_url}\n\n"
            f"Page content:\n{page_text}\n\n"
            "Extract every passage relevant to the query. Be concise."
        )
        response = self.llm.chat(system, user)
        if response.strip().upper() == "NOTHING_RELEVANT":
            return ""
        return response[:EXTRACT_CHARS]

    def run(self, query: str) -> str:
        gathered_info: list[str] = []

        # Step 1: Plan search queries
        print("[1/8] Planning search queries...")
        system1 = (
            "You are a research assistant planning web searches. "
            "Reply with ONLY 3 search query variations, one per line. No numbering, no explanation."
        )
        user1 = (
            f"I want to find: {query}\n\n"
            "Write 3 different search queries that together maximise the chance of finding the "
            "official source or most relevant page. Vary the phrasing and keywords across queries."
        )
        raw_queries = self.llm.chat(system1, user1)
        search_queries = [q.strip() for q in raw_queries.strip().splitlines() if q.strip()][:3]
        if not search_queries:
            search_queries = [query]
        for q in search_queries:
            print(f"      - {q}")

        # Step 2: Search all queries in parallel
        print("[2/8] Searching DuckDuckGo (parallel)...")
        seen_urls: set[str] = set()
        results: list[dict] = []

        def run_search(q: str) -> list[dict]:
            return self.search.search(q)

        with ThreadPoolExecutor(max_workers=len(search_queries)) as executor:
            futures = {executor.submit(run_search, q): q for q in search_queries}
            for future in as_completed(futures):
                for r in future.result():
                    if r["url"] not in seen_urls:
                        seen_urls.add(r["url"])
                        results.append(r)

        if not results:
            return "No search results found."
        print(f"      Found {len(results)} unique result(s):")
        for i, r in enumerate(results):
            print(f"      {i+1}. {r['title']}")
            print(f"         {r['url']}")

        # Step 3: Select best URL
        print("[3/8] Asking Qwen to select best URL...")
        results_text = "\n".join(
            f"{i+1}. {r['title']} — {r['snippet']}\n   URL: {r['url']}"
            for i, r in enumerate(results)
        )
        system2 = "You are a research assistant. Reply with ONLY the number of your choice. No explanation."
        user2 = f"Query: {query}\n\n{results_text}\n\nWhich result best answers the query?"
        response2 = self.llm.chat(system2, user2)
        try:
            idx = int(response2.strip()) - 1
            if idx < 0 or idx >= len(results):
                idx = 0
        except ValueError:
            idx = 0
        selected_url = results[idx]["url"]
        print(f"      Selected result {idx+1}: {results[idx]['title']}")

        # Step 4: Browse selected URL
        print(f"[4/8] Browsing: {selected_url}")
        page_data = self.browser.fetch_page(selected_url)
        raw_chars = len(page_data["text"])
        extracted = self._extract_relevant(query, page_data["text"], page_data["url"])
        print(f"      Retrieved {raw_chars} chars raw, extracted {len(extracted)} chars relevant")
        if extracted:
            gathered_info.append(f"Source: {page_data['url']}\n{extracted}")

        # Steps 5–7: Evaluate sufficiency (up to MAX_BROWSE_ITERATIONS hops)
        hop = 0
        current_page = page_data
        current_extracted = extracted
        while hop < MAX_BROWSE_ITERATIONS:
            step_label = hop + 5
            print(f"[{step_label}/8] Evaluating page content...")

            if not current_extracted:
                # Nothing relevant — force follow without LLM call
                if current_page["links"]:
                    follow_url = current_page["links"][0]["url"]
                    print(f"      No relevant content — following first link: {follow_url}")
                    current_page = self.browser.fetch_page(follow_url)
                    raw_chars = len(current_page["text"])
                    current_extracted = self._extract_relevant(query, current_page["text"], current_page["url"])
                    print(f"      Retrieved {raw_chars} chars raw, extracted {len(current_extracted)} chars relevant")
                    if current_extracted:
                        gathered_info.append(f"Source: {current_page['url']}\n{current_extracted}")
                    hop += 1
                    continue
                else:
                    print("      No relevant content and no links to follow")
                    break

            links_text = "\n".join(
                f"{i+1}. {l['text']} — {l['url']}"
                for i, l in enumerate(current_page["links"])
            )
            system5 = "You are a research assistant. Reply with exactly: DONE or FOLLOW:<number>"
            user5 = (
                f"Query: {query}\n\n"
                f"Extracted relevant content:\n{current_extracted}\n\n"
                f"Links on this page:\n{links_text}\n\n"
                "Does the extracted content fully answer the query? Reply DONE if yes, FOLLOW:<number> for more detail."
            )
            response5 = self.llm.chat(system5, user5)
            print(f"      Model decision: {response5[:60]}")

            if response5.startswith("DONE"):
                print("      Page is sufficient — moving to synthesis")
                break
            elif response5.startswith("FOLLOW:"):
                try:
                    link_idx = int(response5.split(":")[1].strip()) - 1
                    if 0 <= link_idx < len(current_page["links"]):
                        follow_url = current_page["links"][link_idx]["url"]
                        print(f"[{step_label + 1}/8] Following link: {follow_url}")
                        current_page = self.browser.fetch_page(follow_url)
                        raw_chars = len(current_page["text"])
                        current_extracted = self._extract_relevant(query, current_page["text"], current_page["url"])
                        print(f"      Retrieved {raw_chars} chars raw, extracted {len(current_extracted)} chars relevant")
                        if current_extracted:
                            gathered_info.append(f"Source: {current_page['url']}\n{current_extracted}")
                        hop += 1
                        continue
                except (ValueError, IndexError):
                    pass
            print("      No further links to follow")
            break

        # Step 8: Synthesize
        print("[8/8] Synthesizing answer...")
        combined = "\n\n---\n\n".join(gathered_info)
        if len(combined) > MAX_SYNTHESIS_CHARS:
            combined = combined[:MAX_SYNTHESIS_CHARS]

        if self.schema:
            schema_str = json.dumps(self.schema)
            system_synth = (
                "You are a research assistant. Summarize findings to answer the query. "
                "Reply with JSON only matching this schema: " + schema_str
            )
        else:
            system_synth = (
                "You are a research assistant. Summarize findings to answer the query. "
                "Be concise. Only use the provided content."
            )

        user_synth = f"Query: {query}\n\nResearch:\n{combined}"

        if self.schema and _JSONSCHEMA_AVAILABLE:
            coerced = _coerce_to_schema(self.schema)
            for attempt in range(3):
                response = self.llm.chat(system_synth, user_synth)
                clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())
                try:
                    jsonschema.validate(json.loads(clean), coerced)
                    return clean  # valid
                except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                    if attempt < 2:
                        user_synth += f"\n\nPrevious attempt failed: {e}\nReturn ONLY valid JSON."
            return response  # graceful fallback
        else:
            return self.llm.chat(system_synth, user_synth)


DEFAULT_QUERY = "what promotions are taking place at the Goose, fulham pub"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--schema", default=None,
                        help="JSON schema as inline string or path to .json file")
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        print(f"Research query [{DEFAULT_QUERY}]: ", end="", flush=True)
        user_input = input().strip()
        query = user_input if user_input else DEFAULT_QUERY

    schema = None
    if args.schema:
        if os.path.isfile(args.schema):
            with open(args.schema) as f:
                schema = json.load(f)
        else:
            try:
                schema = json.loads(args.schema)
            except json.JSONDecodeError as e:
                print(f"Error: --schema is not a valid file or JSON: {e}", file=sys.stderr)
                sys.exit(1)

    llm = OllamaClient()
    if not llm.is_available():
        print("Error: Ollama server not running. Start it with: ollama serve &", file=sys.stderr)
        sys.exit(1)

    print(f"\nModel : {MODEL}")
    print(f"Query : {query}")
    if schema:
        print(f"Schema: {json.dumps(schema)}")
    print("=" * 60)
    agent = ResearchAgent(llm=llm, search=SearchTool(), browser=BrowserTool(), schema=schema)
    answer = agent.run(query)
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
