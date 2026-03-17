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

CANDIDATE_SELECTORS = ["article", "[class*='event']", "[class*='card']", "section", "li"]

OLLAMA_MODEL = "qwen3:1.7b"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MINIMAX_MODEL = "MiniMax-Text-01"
MINIMAX_API_URL = "https://api.minimaxi.chat/v1/chat/completions"
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
SCREENSHOT_MIN_AREA_PCT = 0.05   # bbox must be ≥ 5% of viewport to be a useful card
SCREENSHOT_MAX_AREA_PCT = 0.60   # bbox must be ≤ 60% to avoid grabbing whole page
SCREENSHOT_PADDING = 16
SCREENSHOT_LEVELS = 10


def _extract_text(html: str) -> str:
    """Strip noise tags and return clean text from HTML using BeautifulSoup."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    main = next(
        (t for t in (soup.find("main"), soup.find("article"), soup.find("body"), soup)
         if t and t.get_text(strip=True)),
        soup,
    )
    return "\n".join(l for l in main.get_text("\n", strip=True).splitlines() if l)


def _extract_links(page, base_url: str) -> list[dict]:
    """Extract absolute links from a Playwright page."""
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
        abs_url = urllib.parse.urljoin(base_url, href)
        if abs_url not in seen:
            seen.add(abs_url)
            links.append({"text": link.get("text", "")[:80], "url": abs_url})
        if len(links) >= MAX_LINKS_TO_SHOW:
            break
    return links


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


class MiniMaxClient:
    """MiniMax API — OpenAI-compatible. Requires MINIMAX_API_KEY."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def is_available(self) -> bool:
        return True

    def chat(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": MINIMAX_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1024,
        }).encode()
        req = urllib.request.Request(
            MINIMAX_API_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()


def make_llm_client(provider: str) -> OllamaClient | ClaudeClient | MiniMaxClient:
    if provider == "minimax":
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            print("Error: MINIMAX_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
        return MiniMaxClient(api_key=api_key)
    elif provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
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
    """Persistent Playwright browser session. Use as a context manager or call start()/close()."""

    _USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, screenshot_dir: str | None = None):
        self._pw = self._browser = self._context = None
        self.screenshot_dir = screenshot_dir

    def start(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            user_agent=self._USER_AGENT,
            viewport={"width": 1280, "height": 900},
        )

    def close(self):
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass
        self._pw = self._browser = self._context = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()

    def _screenshot_promo_cropped(self, page, url: str, promo_index: int, keyword: str) -> str | None:
        """Take a cropped screenshot of the element best matching keyword, with DOM-walk bbox selection."""
        from pathlib import Path
        slug = re.sub(r"[^a-zA-Z0-9]", "_", url)[:60].strip("_")
        out_path = str(Path(self.screenshot_dir) / f"{slug}_{promo_index}.png")

        # Find the element matching the keyword
        el_handle = None
        for selector in CANDIDATE_SELECTORS:
            try:
                loc = page.locator(selector).filter(has_text=keyword)
                if loc.count() > 0:
                    loc.first.scroll_into_view_if_needed()
                    el_handle = loc.first.element_handle()
                    break
            except Exception:
                continue

        if el_handle is None:
            page.screenshot(path=out_path, full_page=True)
            return out_path

        # Walk up DOM collecting viewport-relative bboxes
        boxes = page.evaluate("""el => {
            const vw = window.innerWidth, vh = window.innerHeight;
            const result = [];
            let cur = el;
            for (let i = 0; i < 8 && cur && cur.tagName !== 'BODY'; i++) {
                const r = cur.getBoundingClientRect();
                result.push({ x: r.x, y: r.y, width: r.width, height: r.height,
                              pct: (r.width * r.height) / (vw * vh) });
                cur = cur.parentElement;
            }
            return result;
        }""", el_handle)

        # Pick first ancestor >= MIN_AREA_PCT and <= MAX_AREA_PCT; fall back to largest
        best = next((b for b in boxes if b["pct"] >= SCREENSHOT_MIN_AREA_PCT
                                     and b["pct"] <= SCREENSHOT_MAX_AREA_PCT), None)
        if best is None:
            best = max(boxes, key=lambda b: b["pct"]) if boxes else None
        if best is None:
            page.screenshot(path=out_path, full_page=True)
            return out_path

        vw = page.viewport_size["width"]
        vh = page.viewport_size["height"]
        clip = {
            "x":      max(0, best["x"] - SCREENSHOT_PADDING),
            "y":      max(0, best["y"] - SCREENSHOT_PADDING),
            "width":  min(vw - max(0, best["x"] - SCREENSHOT_PADDING),
                          best["width"]  + 2 * SCREENSHOT_PADDING),
            "height": min(vh - max(0, best["y"] - SCREENSHOT_PADDING),
                          best["height"] + 2 * SCREENSHOT_PADDING),
        }
        page.screenshot(path=out_path, clip=clip)
        return out_path

    def _screenshot_promo_all_levels(self, page, url, promo_index, char_offset, segments):
        from pathlib import Path
        slug = re.sub(r"[^a-zA-Z0-9]", "_", url)[:60].strip("_")
        base = Path(self.screenshot_dir)

        def fallback():
            out = str(base / f"{slug}_{promo_index}_lv0.png")
            try:
                page.screenshot(path=out, full_page=True)
            except Exception:
                pass
            return (out, [out])

        if char_offset is None or not segments:
            return fallback()

        # Binary search for segment containing char_offset
        lo, hi, target_seg = 0, len(segments) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            seg = segments[mid]
            if seg["end"] < char_offset:
                lo = mid + 1
            elif seg["start"] > char_offset:
                hi = mid - 1
            else:
                target_seg = seg
                break
        if target_seg is None:
            return fallback()

        # Find live DOM element at char offset (must use evaluate_handle — DOM elements can't be JSON-serialized)
        try:
            element = page.evaluate_handle("""({targetStart}) => {
                const SKIP = new Set(['SCRIPT','STYLE','NAV','FOOTER','HEADER','ASIDE','NOSCRIPT']);
                const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
                    acceptNode(node) {
                        let el = node.parentElement;
                        while (el) { if (SKIP.has(el.tagName)) return NodeFilter.FILTER_REJECT; el = el.parentElement; }
                        return node.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
                    }
                });
                let offset = 0, node;
                while ((node = walker.nextNode())) {
                    const txt = node.textContent.trim(); if (!txt) continue;
                    if (offset <= targetStart && targetStart <= offset + txt.length)
                        return node.parentElement;
                    offset += txt.length + 1;
                }
                return null;
            }""", {"targetStart": target_seg["start"]})
            if element.as_element() is None:
                element = None
        except Exception:
            element = None
        if element is None:
            return fallback()

        # Walk up SCREENSHOT_LEVELS ancestors
        try:
            ancestor_boxes = page.evaluate(f"""(el) => {{
                const vw = window.innerWidth, vh = window.innerHeight;
                const results = []; let cur = el;
                for (let i = 0; i < {SCREENSHOT_LEVELS} && cur && cur.tagName !== 'BODY'; i++) {{
                    const r = cur.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0)
                        results.push({{ level: i, x: r.x, y: r.y, width: r.width, height: r.height }});
                    cur = cur.parentElement;
                }}
                return results;
            }}""", element)
        except Exception:
            return fallback()
        if not ancestor_boxes:
            return fallback()

        # Screenshot each level
        vw, vh = page.viewport_size["width"], page.viewport_size["height"]
        all_paths = []
        successful: list[tuple[str, float, float]] = []  # (path, clip_width, clip_area)
        for box in ancestor_boxes:
            n = box["level"]
            out = str(base / f"{slug}_{promo_index}_lv{n}.png")
            x0 = max(0.0, box["x"] - SCREENSHOT_PADDING)
            y0 = max(0.0, box["y"] - SCREENSHOT_PADDING)
            cw = min(float(vw) - x0, box["width"]  + 2 * SCREENSHOT_PADDING)
            ch = min(float(vh) - y0, box["height"] + 2 * SCREENSHOT_PADDING)
            clip = {"x": x0, "y": y0, "width": cw, "height": ch}
            try:
                page.screenshot(path=out, clip=clip)
                all_paths.append(out)
                successful.append((out, cw, cw * ch))
            except Exception as e:
                print(f"      [screenshot] lv{n} failed: {e}")

        if not successful:
            return fallback()

        # Pick first level ≥ 80% viewport width; fall back to largest area
        primary_path = next(
            (p for p, cw, _ in successful if cw >= 0.8 * vw),
            max(successful, key=lambda t: t[2])[0]
        )
        return (primary_path, all_paths)

    def _build_js_segments(self, page) -> tuple[str, list[dict]]:
        try:
            segments = page.evaluate("""() => {
                const SKIP = new Set(['SCRIPT','STYLE','NAV','FOOTER','HEADER','ASIDE','NOSCRIPT']);
                const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
                    acceptNode(node) {
                        let el = node.parentElement;
                        while (el) { if (SKIP.has(el.tagName)) return NodeFilter.FILTER_REJECT; el = el.parentElement; }
                        return node.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
                    }
                });
                const results = []; let offset = 0; let node;
                while ((node = walker.nextNode())) {
                    const txt = node.textContent.trim(); if (!txt) continue;
                    const el = node.parentElement; if (!el) continue;
                    const r = el.getBoundingClientRect();
                    if (r.width === 0 && r.height === 0) continue;
                    results.push({ text: txt, start: offset, end: offset + txt.length,
                                   x: r.x, y: r.y, width: r.width, height: r.height });
                    offset += txt.length + 1;
                }
                return results;
            }""")
            if not segments:
                return ("", [])
            return ("\n".join(s["text"] for s in segments), segments)
        except Exception as e:
            print(f"      [segments] JS TreeWalker failed: {e}")
            return ("", [])

    def _try_http_fetch(self, url: str) -> str | None:
        """Fast HTTP fetch via requests. Returns extracted text or None on failure."""
        try:
            import requests
            r = requests.get(
                url,
                headers={"User-Agent": self._USER_AGENT},
                timeout=8,
            )
            r.raise_for_status()
            return _extract_text(r.text)
        except Exception:
            return None

    def fetch_page(self, url: str) -> dict:
        # 1. Try fast HTTP fetch first (skip when screenshot_dir set — need Playwright for segments)
        if not self.screenshot_dir:
            text = self._try_http_fetch(url)
            if text and len(text) >= 500:
                return {"url": url, "text": text[:MAX_PAGE_CHARS], "links": [], "segments": []}

        # 2. Fall back to Playwright
        if self._context is None:
            self.start()
        page = self._context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=NETWORKIDLE_TIMEOUT)
            except Exception:
                pass
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(SCROLL_DELAY)
            js_text, segments = self._build_js_segments(page)
            html = page.content()
            bs4_text = _extract_text(html)
            primary_text = js_text if js_text else bs4_text
            links = _extract_links(page, url)
            return {"url": url, "text": primary_text[:MAX_PAGE_CHARS], "links": links, "segments": segments}
        except Exception as e:
            return {"url": url, "text": f"Error fetching page: {e}", "links": [], "segments": []}
        finally:
            page.close()


def debug_fetch(url: str) -> None:
    import os, time
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = f"outputs/debug_{ts}_{domain}"
    os.makedirs(outdir, exist_ok=True)

    # --- HTTP fetch ---
    http_html = ""
    try:
        import requests
        r = requests.get(url, headers={"User-Agent": BrowserTool._USER_AGENT}, timeout=8)
        r.raise_for_status()
        http_html = r.text
    except Exception as e:
        print(f"[debug] HTTP fetch failed: {e}")
    http_text = _extract_text(http_html) if http_html else ""
    print(f"[debug] HTTP raw HTML:        {len(http_html):,} chars")
    print(f"[debug] HTTP extracted text:  {len(http_text):,} chars")
    with open(f"{outdir}/http_raw.html", "w", encoding="utf-8") as f:
        f.write(http_html)
    with open(f"{outdir}/http_extracted.txt", "w", encoding="utf-8") as f:
        f.write(http_text)

    # --- Playwright fetch ---
    pw_html = ""
    pw_text = ""
    browser = BrowserTool()
    browser.start()
    try:
        page = browser._context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=NETWORKIDLE_TIMEOUT)
            except Exception:
                pass
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(SCROLL_DELAY)
            pw_html = page.content()
            pw_text = _extract_text(pw_html)
        finally:
            page.close()
    finally:
        browser.close()
    print(f"[debug] Playwright raw HTML:  {len(pw_html):,} chars")
    print(f"[debug] Playwright extracted: {len(pw_text):,} chars")
    with open(f"{outdir}/playwright_raw.html", "w", encoding="utf-8") as f:
        f.write(pw_html)
    with open(f"{outdir}/playwright_extracted.txt", "w", encoding="utf-8") as f:
        f.write(pw_text)

    print(f"\nSaved to: {outdir}/")


def _batch_worker(args: tuple) -> dict:
    """Top-level worker for ProcessPoolExecutor — must be picklable (no closures)."""
    pub, provider, search_provider, schema, screenshot_dir = args
    name = pub.get("name", pub.get("url", "unknown"))
    query = pub.get("query") or f"what promotions are taking place at {name}"
    url = pub.get("url") or None
    print(f"[{name}] starting", flush=True)
    try:
        llm = make_llm_client(provider)
        search = make_search_tool(search_provider)
        agent = ResearchAgent(llm=llm, search=search, browser=BrowserTool(screenshot_dir=screenshot_dir), schema=schema)
        result = agent.run(query, start_url=url)
        print(f"[{name}] done", flush=True)
        return {"pub": name, "query": query, "result": result, "error": None}
    except Exception as e:
        print(f"[{name}] ERROR: {e}", flush=True)
        return {"pub": name, "query": query, "result": None, "error": str(e)}


def run_batch(batch_file: str, provider: str, search_provider: str, schema, workers: int, screenshot_dir: str | None = None) -> None:
    import time
    from concurrent.futures import ProcessPoolExecutor
    with open(batch_file) as f:
        pubs = json.load(f)

    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    outfile = f"outputs/batch_{ts}.json"

    print(f"Batch   : {len(pubs)} pubs, {workers} workers")
    print(f"Provider: {provider}  Search: {search_provider}")
    print(f"Output  : {outfile}")
    print("=" * 60)

    results = []
    worker_args = [(pub, provider, search_provider, schema, screenshot_dir) for pub in pubs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_batch_worker, a) for a in worker_args]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            # Write incrementally so partial results aren't lost on crash
            with open(outfile, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Saved {len(results)} results to {outfile}")


class ResearchAgent:
    def __init__(self, llm: OllamaClient, search: SearchTool, browser: BrowserTool, schema=None):
        self.llm = llm
        self.search = search
        self.browser = browser
        self.schema = schema
        self._url_segments: dict[str, tuple[str, list[dict]]] = {}  # url → (page_text, segments)

    def _extract_relevant(self, query: str, page_text: str, page_url: str) -> str:
        system = (
            "You are a research assistant extracting pub promotion details from a web page.\n"
            "Extract ANY of the following: deals, discounts, happy hours, quiz nights, events, "
            "special pricing, drink specials, weekly offers, or recurring promotions.\n"
            "Reply with ONLY the relevant text copied or lightly paraphrased from the page.\n"
            "Include specific details: prices, percentages, days of the week, time ranges.\n"
            "Only reply with exactly: NOTHING_RELEVANT if the page has truly no pub info at all.\n"
            "Do not add commentary, headers, or explanation."
        )
        user = (
            f"Query: {query}\nPage URL: {page_url}\n\nPage content:\n{page_text}\n\n"
            "Extract every passage about pub drinks, pricing, promotions, deals, discounts, "
            "happy hours, quiz nights, weekly specials, or events. Include prices, days, and times."
        )
        response = self.llm.chat(system, user)
        if response.strip().upper() == "NOTHING_RELEVANT":
            return ""
        return response[:EXTRACT_CHARS]

    def _add_promo_screenshots(self, json_str: str) -> str:
        """Post-process synthesized JSON: take one cropped screenshot per promotion."""
        try:
            promos = json.loads(json_str)
            if not isinstance(promos, list):
                return json_str
        except (json.JSONDecodeError, AttributeError):
            return json_str

        from pathlib import Path
        Path(self.browser.screenshot_dir).mkdir(parents=True, exist_ok=True)

        # Group by source_url — one page load per URL
        url_groups: dict[str, list[tuple[int, dict]]] = {}
        for i, promo in enumerate(promos):
            url = promo.get("source_url", "")
            if url:
                url_groups.setdefault(url, []).append((i, promo))

        for url, group in url_groups.items():
            if self.browser._context is None:
                self.browser.start()
            page = self.browser._context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                try:
                    page.wait_for_load_state("networkidle", timeout=NETWORKIDLE_TIMEOUT)
                except Exception:
                    pass
                page_text, segments = self._url_segments.get(url, ("", []))
                for i, promo in group:
                    anchor = promo.get("source_text_anchor", "")
                    char_offset = page_text.find(anchor) if anchor else -1
                    if char_offset < 0:
                        char_offset = None
                    primary_path, all_paths = self.browser._screenshot_promo_all_levels(
                        page, url, i, char_offset, segments
                    )
                    if primary_path:
                        promos[i]["screenshot_path"] = primary_path
                        print(f"  [screenshot] promo {i}: {primary_path} ({len(all_paths)} levels)")
            except Exception as e:
                print(f"  [screenshot] failed for {url}: {e}")
            finally:
                page.close()

        return json.dumps(promos, ensure_ascii=False)

    def run(self, query: str, start_url: str | None = None) -> str:
        gathered_info: list[str] = []
        self._url_segments = {}
        self.browser.start()
        try:
            result = self._run(query, start_url, gathered_info)
            if self.browser.screenshot_dir and self.schema:
                result = self._add_promo_screenshots(result)   # browser still open here
            return result
        finally:
            self.browser.close()

    def _run(self, query: str, start_url: str | None, gathered_info: list[str]) -> str:
        # Optional: skip search and start directly from a known URL
        if start_url:
            print(f"[SKIP SEARCH] Using provided start URL: {start_url}")
            page_data = self.browser.fetch_page(start_url)
            self._url_segments[page_data["url"]] = (page_data.get("text", ""), page_data.get("segments", []))
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
                    "- source_url: copy from [SOURCE_URL: <url>] tag above the relevant content\n"
                    "- source_text_anchor: a verbatim 20-30 character snippet copied EXACTLY from the page text\n"
                    "  near where this promotion appears; used to locate the element for screenshotting\n\n"
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

        # Step 4: Browse top URLs
        print(f"[4/8] Browsing top {len(browse_indices)} result(s)...")
        urls_to_browse = [selection_pool[bidx]["url"] for bidx in browse_indices]
        for url in urls_to_browse:
            print(f"      - {url}")

        # Sequential: sync_playwright is not thread-safe (greenlet-bound)
        fetched_pages: dict[str, dict] = {}
        for url in urls_to_browse:
            pd = self.browser.fetch_page(url)
            fetched_pages[pd["url"]] = pd
            self._url_segments[pd["url"]] = (pd.get("text", ""), pd.get("segments", []))

        # Extract relevant content
        page_data = None
        for url in urls_to_browse:
            pd = fetched_pages.get(url, {"url": url, "text": "", "links": [], "segments": []})
            raw_chars = len(pd["text"])
            extracted = self._extract_relevant(query, pd["text"], pd["url"])
            print(f"      {pd['url']}: {raw_chars} chars raw, {len(extracted)} chars relevant")
            if extracted:
                gathered_info.append(f"[SOURCE_URL: {pd['url']}]\n{extracted}")
            if page_data is None:
                page_data = pd  # use best-pick as starting page for hop loop

        if page_data is None:
            page_data = self.browser.fetch_page(selection_pool[idx]["url"])
            self._url_segments[page_data["url"]] = (page_data.get("text", ""), page_data.get("segments", []))
        extracted = self._extract_relevant(query, page_data["text"], page_data["url"])

        # Steps 5–7: Evaluate sufficiency (up to MAX_BROWSE_ITERATIONS hops)
        hop = 0
        current_page = page_data
        current_extracted = extracted  # extracted from the page_data re-extraction above
        while hop < MAX_BROWSE_ITERATIONS:
            step_label = hop + 5
            print(f"[{step_label}/8] Evaluating page content...")

            if not current_extracted:
                # Nothing relevant — force follow without LLM call
                if current_page["links"]:
                    follow_url = current_page["links"][0]["url"]
                    print(f"      No relevant content — following first link: {follow_url}")
                    current_page = self.browser.fetch_page(follow_url)
                    self._url_segments[current_page["url"]] = (current_page.get("text", ""), current_page.get("segments", []))
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
                        self._url_segments[current_page["url"]] = (current_page.get("text", ""), current_page.get("segments", []))
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
                "- source_url: the EXACT URL shown in [SOURCE_URL: ...] above the content where this promotion appeared\n"
                "- source_text_anchor: a verbatim 20-30 character snippet copied EXACTLY from the page text\n"
                "  near where this promotion appears; used to locate the element for screenshotting\n\n"
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
    parser.add_argument("--provider", default="minimax", choices=["minimax", "claude", "ollama"],
                        help="LLM provider: 'minimax' (default), 'claude' (Haiku), 'ollama' (local Qwen)")
    parser.add_argument("--search-provider", default="brave", choices=["brave", "ddg"],
                        help="Search provider: 'brave' (default, requires BRAVE_SEARCH_API_KEY) or 'ddg' (DuckDuckGo)")
    parser.add_argument("--debug-url", default=None,
                        help="Fetch URL and save all intermediate content for debugging")
    parser.add_argument("--batch-file", default=None,
                        help="JSON file with list of {name, url?, query?} objects to research in parallel")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel workers for --batch-file (default: 5)")
    parser.add_argument("--screenshot-dir", default=None,
                        help="Directory to save page screenshots; adds screenshot_path to each promotion in JSON output")
    args = parser.parse_args()

    if args.debug_url:
        debug_fetch(args.debug_url)
        sys.exit(0)

    # Parse schema (needed for both batch and single modes)
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

    if args.batch_file:
        run_batch(args.batch_file, args.provider, args.search_provider, schema, args.workers, args.screenshot_dir)
        sys.exit(0)

    if args.query:
        query = args.query
    else:
        print(f"Research query [{DEFAULT_QUERY}]: ", end="", flush=True)
        user_input = input().strip()
        query = user_input if user_input else DEFAULT_QUERY

    llm = make_llm_client(args.provider)
    search = make_search_tool(args.search_provider)
    model_label = {"minimax": MINIMAX_MODEL, "claude": CLAUDE_MODEL, "ollama": OLLAMA_MODEL}.get(args.provider, args.provider)

    print(f"\nLLM     : {args.provider} ({model_label})")
    print(f"Search  : {args.search_provider} ({type(search).__name__})")
    print(f"Query   : {query}")
    if schema:
        print(f"Schema  : {json.dumps(schema)}")
    if args.screenshot_dir:
        print(f"Screenshots: {args.screenshot_dir}")
    print("=" * 60)
    agent = ResearchAgent(llm=llm, search=search, browser=BrowserTool(screenshot_dir=args.screenshot_dir), schema=schema)
    answer = agent.run(query, start_url=args.start_url)
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
