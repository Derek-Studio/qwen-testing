#!/usr/bin/env python3
"""
ReAct-style research agent. Searches the web with DuckDuckGo, browses pages
with headless Chromium, and synthesizes an answer using a local or cloud LLM.

Providers:
  --provider ollama   Local Qwen via Ollama (default)
  --provider claude   Claude Haiku via Anthropic API (requires ANTHROPIC_API_KEY)

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

try:
    import anthropic as _anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

OLLAMA_MODEL = "qwen3:1.7b"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
BASE_URL = "http://localhost:11434"
MAX_SEARCH_RESULTS = 5
MAX_BROWSE_ITERATIONS = 1
MAX_PAGE_CHARS = 6000
EXTRACT_CHARS = 2000
MAX_SYNTHESIS_CHARS = 4000
MAX_LINKS_TO_SHOW = 10
OLLAMA_TIMEOUT = 180
MAX_RESULTS_FOR_SELECTION = 6
NETWORKIDLE_TIMEOUT = 5000   # ms; raise to 15000 for production
SCROLL_DELAY = 500           # ms; raise to 1500 for production


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
            "model": OLLAMA_MODEL,
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


class ClaudeClient:
    def __init__(self, api_key: str):
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        self._client = _anthropic_sdk.Anthropic(api_key=api_key)

    def is_available(self) -> bool:
        return True  # API availability checked at construction time

    def chat(self, system: str, user: str) -> str:
        message = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text.strip()


def make_llm_client(provider: str) -> OllamaClient | ClaudeClient:
    if provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
            print("Export it before running: export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
            sys.exit(1)
        return ClaudeClient(api_key=api_key)
    else:
        client = OllamaClient()
        if not client.is_available():
            print("Error: Ollama server not running. Start it with: ollama serve &", file=sys.stderr)
            sys.exit(1)
        return client


class BraveSearchTool:
    """Brave Search API — requires BRAVE_SEARCH_API_KEY. Free tier: 2000 queries/month."""
    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
        try:
            params = urllib.parse.urlencode({"q": query, "count": max_results})
            req = urllib.request.Request(
                f"{self.API_URL}?{params}",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.api_key,
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                import gzip
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode())
            results = []
            for r in data.get("web", {}).get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("description", ""),
                    "url": r.get("url", ""),
                })
            return results
        except Exception as e:
            print(f"      [Brave] search error: {e} — falling back to DDG")
            return DDGSearchTool().search(query, max_results)


class DDGSearchTool:
    """DuckDuckGo search — no API key needed."""
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


# Keep SearchTool as an alias for backwards compatibility
SearchTool = DDGSearchTool


def make_search_tool(provider: str) -> BraveSearchTool | DDGSearchTool:
    if provider == "brave":
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not api_key:
            print("Warning: BRAVE_SEARCH_API_KEY not set — falling back to DuckDuckGo", file=sys.stderr)
            return DDGSearchTool()
        return BraveSearchTool(api_key=api_key)
    return DDGSearchTool()


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
                    page.wait_for_load_state("networkidle", timeout=NETWORKIDLE_TIMEOUT)
                except Exception:
                    pass  # fall through silently on timeout

                # Scroll to trigger lazy-loaded content
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(SCROLL_DELAY)

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
            "You are a research assistant extracting pub promotion details from a web page.\n"
            "Extract ANY of the following: deals, discounts, happy hours, quiz nights, events, "
            "special pricing, drink specials, weekly offers, or recurring promotions.\n"
            "Also extract: drink prices (cocktails, pints, shots), event days/times, "
            "recurring weekly specials, and any 'what's on' or 'offers' information.\n"
            "Reply with ONLY the relevant text copied or lightly paraphrased from the page.\n"
            "Include specific details: prices, percentages, days of the week, time ranges.\n"
            "Only reply with exactly: NOTHING_RELEVANT if the page has truly no pub info at all "
            "(e.g. a privacy policy page, a completely unrelated site).\n"
            "Do not add commentary, headers, or explanation."
        )
        user = (
            f"Query: {query}\n"
            f"Page URL: {page_url}\n\n"
            f"Page content:\n{page_text}\n\n"
            "Extract every passage about pub drinks, pricing, promotions, deals, discounts, "
            "happy hours, quiz nights, weekly specials, or events. Include prices, days, and times."
        )
        response = self.llm.chat(system, user)
        if response.strip().upper() == "NOTHING_RELEVANT":
            return ""
        return response[:EXTRACT_CHARS]

    def run(self, query: str, start_url: str | None = None) -> str:
        gathered_info: list[str] = []

        # Optional: skip search and start directly from a known URL
        if start_url:
            print(f"[SKIP SEARCH] Using provided start URL: {start_url}")
            page_data = self.browser.fetch_page(start_url)
            extracted = self._extract_relevant(query, page_data["text"], page_data["url"])
            print(f"      Retrieved {len(page_data['text'])} chars raw, {len(extracted)} chars relevant")
            if extracted:
                gathered_info.append(f"[SOURCE_URL: {page_data['url']}]\n{extracted}")
            # Proceed directly to synthesis
            print("[8/8] Synthesizing answer...")
            if not gathered_info:
                print("      No relevant content found — returning empty result")
                return "[]" if self.schema else "No relevant information found."
            combined = "\n\n---\n\n".join(gathered_info)
            if len(combined) > MAX_SYNTHESIS_CHARS:
                combined = combined[:MAX_SYNTHESIS_CHARS]
            if self.schema:
                schema_str = json.dumps(self.schema)
                system_synth = (
                    "You are a research assistant extracting pub promotion data. "
                    "Return a JSON array where each element is a separate promotion or deal. "
                    "Each promotion object must match this schema: " + schema_str + "\n\n"
                    "Field meanings:\n"
                    "- description: what the promotion is\n"
                    "- discount: the price or saving (e.g. '£6', '50% off', 'variable')\n"
                    "- days: which days of the week it runs\n"
                    "- time: the hours it is active\n"
                    "- source_url: copy from [SOURCE_URL: <url>] tag above the relevant content\n\n"
                    "Reply with ONLY the JSON array, no explanation."
                )
                user_synth = f"Query: {query}\n\nResearch:\n{combined}"
                if _JSONSCHEMA_AVAILABLE:
                    coerced = _coerce_to_schema(self.schema)
                    for attempt in range(3):
                        response = self.llm.chat(system_synth, user_synth)
                        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())
                        try:
                            jsonschema.validate(json.loads(clean), coerced)
                            return clean
                        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                            if attempt < 2:
                                user_synth += f"\n\nPrevious attempt failed: {e}\nReturn ONLY valid JSON."
                    return response
                else:
                    return self.llm.chat(system_synth, user_synth)
            else:
                return self.llm.chat(
                    "You are a research assistant. Summarize pub promotion findings. Be concise.",
                    f"Query: {query}\n\nResearch:\n{combined}"
                )

        # Step 1: Plan search queries
        print("[1/8] Planning search queries...")
        system1 = (
            "You are a research assistant planning web searches for pub promotions. "
            "Reply with ONLY 3 search query variations, one per line. No numbering, no explanation."
        )
        user1 = (
            f"I want to find: {query}\n\n"
            "Write 3 short, distinct search queries (max 8 words each) to find pub promotions, "
            "deals, happy hours, quiz nights, and discounts at this specific pub. "
            "Use the PUB NAME and AREA/CITY only — do NOT include full street addresses or postcodes. "
            "Vary the terms: use 'happy hour', 'deals', 'promotions', 'offers', 'events' across queries."
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

        # Step 3: Select best URL (and pick top 3 candidates to browse)
        print("[3/8] Asking Qwen to select best URL...")
        selection_pool = results[:MAX_RESULTS_FOR_SELECTION]
        results_text = "\n".join(
            f"{i+1}. {r['title']} — {r['snippet']}\n   URL: {r['url']}"
            for i, r in enumerate(selection_pool)
        )
        system2 = (
            "You are a research assistant helping find pub promotions. "
            "Reply with ONLY the number of your choice. No explanation."
        )
        user2 = (
            f"Query: {query}\n\n{results_text}\n\n"
            "Which result is most likely to have specific pub promotion details "
            "(happy hours, deals, quiz nights, discounts)? Prefer official pub websites "
            "or event/listing pages over aggregator homepages."
        )
        response2 = self.llm.chat(system2, user2)
        try:
            idx = int(response2.strip()) - 1
            if idx < 0 or idx >= len(selection_pool):
                idx = 0
        except ValueError:
            idx = 0
        print(f"      Selected result {idx+1}: {selection_pool[idx]['title']}")

        # Build ordered list of URLs to browse: best pick first, then alternatives
        browse_indices = [idx]
        for i in range(min(len(selection_pool), 3)):
            if i not in browse_indices:
                browse_indices.append(i)
        browse_indices = browse_indices[:3]

        # Step 4: Browse top URLs in parallel
        print(f"[4/8] Browsing top {len(browse_indices)} result(s) in parallel...")
        urls_to_browse = [selection_pool[bidx]["url"] for bidx in browse_indices]
        for url in urls_to_browse:
            print(f"      - {url}")

        fetched_pages: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=len(urls_to_browse)) as executor:
            future_to_url = {executor.submit(self.browser.fetch_page, url): url for url in urls_to_browse}
            for future in as_completed(future_to_url):
                pd = future.result()
                fetched_pages[pd["url"]] = pd

        # Extract relevant content sequentially (Ollama is single-threaded)
        page_data = None
        for url in urls_to_browse:
            pd = fetched_pages.get(url, {"url": url, "text": "", "links": []})
            raw_chars = len(pd["text"])
            extracted = self._extract_relevant(query, pd["text"], pd["url"])
            print(f"      {pd['url']}: {raw_chars} chars raw, {len(extracted)} chars relevant")
            if extracted:
                gathered_info.append(f"[SOURCE_URL: {pd['url']}]\n{extracted}")
            if page_data is None:
                page_data = pd  # use best-pick as starting page for hop loop

        if page_data is None:
            page_data = self.browser.fetch_page(selection_pool[idx]["url"])
        extracted = self._extract_relevant(query, page_data["text"], page_data["url"])

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
                        gathered_info.append(f"[SOURCE_URL: {current_page['url']}]\n{current_extracted}")
                    hop += 1
                    continue
                else:
                    print("      No relevant content and no links to follow")
                    break

            links_text = "\n".join(
                f"{i+1}. {l['text']} — {l['url']}"
                for i, l in enumerate(current_page["links"])
            )
            system5 = (
                "You are a research assistant. Reply with exactly: DONE or FOLLOW:<number>\n"
                "Reply DONE if the content already includes pub promotion details (deals, "
                "happy hours, discounts, quiz nights, prices, days, times).\n"
                "Reply FOLLOW:<number> only if a specific link clearly leads to more promotion details."
            )
            user5 = (
                f"Query: {query}\n\n"
                f"Extracted relevant content:\n{current_extracted}\n\n"
                f"Links on this page:\n{links_text}\n\n"
                "Does the extracted content contain specific pub promotion details? "
                "Reply DONE if yes, FOLLOW:<number> if a link clearly leads to more promotion info."
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
                            gathered_info.append(f"[SOURCE_URL: {current_page['url']}]\n{current_extracted}")
                        hop += 1
                        continue
                except (ValueError, IndexError):
                    pass
            print("      No further links to follow")
            break

        # Step 8: Synthesize
        print("[8/8] Synthesizing answer...")
        if not gathered_info:
            print("      No relevant content found — returning empty result")
            return "[]" if self.schema else "No relevant information found."
        combined = "\n\n---\n\n".join(gathered_info)
        if len(combined) > MAX_SYNTHESIS_CHARS:
            combined = combined[:MAX_SYNTHESIS_CHARS]

        if self.schema:
            schema_str = json.dumps(self.schema)
            system_synth = (
                "You are a research assistant extracting pub promotion data. "
                "Return a JSON array where each element is a separate promotion or deal. "
                "Each promotion object must match this schema: " + schema_str + "\n\n"
                "Field meanings:\n"
                "- description: what the promotion is (e.g. 'Thirsty Thursdays happy hour — double spirit + mixer')\n"
                "- discount: the price or saving (e.g. '£6', '50% off', '£75 bar tab', 'variable')\n"
                "- days: which days of the week it runs (e.g. 'Thursday', 'Friday, Saturday')\n"
                "- time: the hours it is active (e.g. '22:00-00:00', '17:30-close', 'all day')\n"
                "- source_url: the EXACT URL shown in [SOURCE_URL: ...] above the content where this promotion appeared\n\n"
                "IMPORTANT: For source_url, copy the URL exactly from [SOURCE_URL: <url>] tags in the research. "
                "Do not invent URLs. Reply with ONLY the JSON array, no explanation."
            )
        else:
            system_synth = (
                "You are a research assistant. Summarize pub promotion findings to answer the query. "
                "Be concise. Only use the provided content."
            )

        user_synth = f"Query: {query}\n\nResearch (each section tagged with its source URL):\n{combined}"

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
    parser.add_argument("--start-url", default=None,
                        help="Skip search and start browsing from this URL directly")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "claude"],
                        help="LLM provider: 'ollama' (local Qwen, default) or 'claude' (Haiku via API)")
    parser.add_argument("--search-provider", default="brave", choices=["brave", "ddg"],
                        help="Search provider: 'brave' (default, requires BRAVE_SEARCH_API_KEY) or 'ddg' (DuckDuckGo)")
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

    llm = make_llm_client(args.provider)
    search = make_search_tool(args.search_provider)
    model_label = CLAUDE_MODEL if args.provider == "claude" else OLLAMA_MODEL

    print(f"\nLLM     : {args.provider} ({model_label})")
    print(f"Search  : {args.search_provider} ({type(search).__name__})")
    print(f"Query   : {query}")
    if schema:
        print(f"Schema  : {json.dumps(schema)}")
    print("=" * 60)
    agent = ResearchAgent(llm=llm, search=search, browser=BrowserTool(), schema=schema)
    answer = agent.run(query, start_url=args.start_url)
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
