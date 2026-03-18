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
import asyncio
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
MINIMAX_MODEL = "MiniMax-M2.5"
MINIMAX_API_URL = "https://api.minimaxi.chat/v1/chat/completions"
BASE_URL = "http://localhost:11434"
MAX_SEARCH_RESULTS = 5
MAX_BROWSE_ITERATIONS = 1
MAX_PAGE_CHARS = 6000
EXTRACT_CHARS = 2000
MAX_SYNTHESIS_CHARS = 4000
MAX_LINKS_TO_SHOW = 10
OLLAMA_TIMEOUT = 180
MAX_SUBPAGES        = 4       # sub-pages beyond home
MAX_SUBPAGE_CHARS   = 3000    # per-page char cap before concat
MAX_TOTAL_CHARS     = 18000   # hard cap on total context to LLM
MAX_LINKS_TO_SCORE  = 40      # links to pull from home for scoring
SUBPAGE_KEYWORDS    = [
    "event", "whats-on", "whats_on", "offer", "deal",
    "promot", "happy", "about", "facilit", "menu",
]
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


def _extract_links(page, base_url: str, max_links: int = MAX_LINKS_TO_SHOW) -> list[dict]:
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
        if len(links) >= max_links:
            break
    return links


def _score_links(links: list[dict], base_url: str) -> list[dict]:
    """Filter and score links for sub-page crawling. Returns sorted list (best first)."""
    base_domain = urllib.parse.urlparse(base_url).netloc
    home_path = urllib.parse.urlparse(base_url).path.rstrip("/")
    asset_exts = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".mp4", ".webp"}

    scored = []
    for link in links:
        url = link.get("url", "")
        text = link.get("text", "")
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            continue
        # Filter external domains
        if parsed.netloc and parsed.netloc != base_domain:
            continue
        # Filter asset URLs
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in asset_exts):
            continue
        # Filter non-HTTP schemes
        if parsed.scheme and parsed.scheme not in ("http", "https", ""):
            continue
        # Filter same-page-as-home
        clean_path = parsed.path.rstrip("/")
        if clean_path == home_path or clean_path == "":
            continue
        combined = (path + " " + text).lower()
        score = sum(1 for kw in SUBPAGE_KEYWORDS if kw in combined)
        scored.append({"url": url, "text": text, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


class SearchCache:
    """JSON-backed cache for Brave search results and crawl subpage URL ordering.

    Cache file structure:
    {
      "searches":  { "<query>": [<result>, ...] },
      "subpages":  { "<home_url>": ["<sub_url>", ...] }
    }
    """

    def __init__(self, path: str):
        self.path = path
        self._data: dict = {"searches": {}, "subpages": {}}
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                self._data = json.load(f)
            self._data.setdefault("searches", {})
            self._data.setdefault("subpages", {})
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get_search(self, query: str) -> list | None:
        return self._data["searches"].get(query)

    def set_search(self, query: str, results: list):
        self._data["searches"][query] = results
        self._save()

    def get_subpages(self, home_url: str) -> list[str] | None:
        return self._data["subpages"].get(home_url)

    def set_subpages(self, home_url: str, urls: list[str]):
        self._data["subpages"][home_url] = urls
        self._save()


class CachedSearchTool:
    """Wraps any search tool with a SearchCache — same .search() interface."""

    def __init__(self, tool, cache: SearchCache):
        self._tool = tool
        self._cache = cache

    def search(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> list:
        cached = self._cache.get_search(query)
        if cached is not None:
            print(f"      [cache] search hit: {query[:60]}")
            return cached
        results = self._tool.search(query, max_results)
        self._cache.set_search(query, results)
        return results


def _coerce_to_schema(raw):
    def _infer(val):
        if isinstance(val, dict):
            if "type" in val or "$schema" in val:
                return val  # already a proper schema node
            return {
                "type": "object",
                "properties": {k: _infer(v) for k, v in val.items()},
            }
        if isinstance(val, list):
            if val and isinstance(val[0], dict):
                return {"type": "array", "items": _infer(val[0])}
            return {"type": "array"}
        return {"type": "string"}

    if isinstance(raw, dict) and ("type" in raw or "$schema" in raw):
        return raw  # already a proper schema
    if isinstance(raw, dict):
        return _infer(raw)
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

    def _dismiss_popups(self, page) -> None:
        """Click common cookie/popup dismiss buttons."""
        try:
            page.evaluate("""() => {
                // 1. Known cookie platform IDs first (most reliable)
                const ids = [
                    'onetrust-accept-btn-handler',
                    'CybotCookiebotDialogBodyButtonAccept',
                    'cookie-accept', 'accept-cookies', 'accept-all-cookies',
                    'cookieAccept', 'cookieConsent', 'gdpr-accept',
                ];
                for (const id of ids) {
                    const el = document.getElementById(id);
                    if (el) { el.click(); return; }
                }
                // 2. Keyword match on all clickable elements
                const keywords = [
                    'accept all', 'accept cookies', 'allow all', 'allow cookies',
                    'i accept', 'i agree', 'agree to all',
                    'accept', 'agree', 'got it', 'ok', 'close', 'dismiss',
                    'reject all', 'decline all', 'continue',
                ];
                const els = document.querySelectorAll(
                    'button, a[role="button"], [class*="cookie"] button, ' +
                    '[class*="consent"] button, [id*="cookie"] button, ' +
                    '[id*="consent"] button, [class*="banner"] button'
                );
                for (const el of els) {
                    const t = el.textContent.trim().toLowerCase();
                    if (keywords.some(k => t === k || t.startsWith(k))) { el.click(); return; }
                }
            }""")
            page.wait_for_timeout(400)
        except Exception:
            pass

    def start(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            viewport={"width": 390, "height": 844},
            is_mobile=True,
            device_scale_factor=3,
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

    async def _fetch_subpages_async(self, urls: list[str]) -> list[dict]:
        """Fetch multiple sub-pages concurrently using async Playwright (JS-rendered).
        Returns list of {url, text, segments} dicts. Pages are closed after extraction."""
        from playwright.async_api import async_playwright

        async def _dismiss_popups_async(page) -> None:
            try:
                await page.evaluate("""() => {
                    const ids = [
                        'onetrust-accept-btn-handler',
                        'CybotCookiebotDialogBodyButtonAccept',
                        'cookie-accept', 'accept-cookies', 'accept-all-cookies',
                        'cookieAccept', 'cookieConsent', 'gdpr-accept',
                    ];
                    for (const id of ids) {
                        const el = document.getElementById(id);
                        if (el) { el.click(); return; }
                    }
                    const keywords = [
                        'accept all', 'accept cookies', 'allow all', 'allow cookies',
                        'i accept', 'i agree', 'agree to all',
                        'accept', 'agree', 'got it', 'ok', 'close', 'dismiss',
                        'reject all', 'decline all', 'continue',
                    ];
                    const els = document.querySelectorAll(
                        'button, a[role="button"], [class*="cookie"] button, ' +
                        '[class*="consent"] button, [id*="cookie"] button, ' +
                        '[id*="consent"] button, [class*="banner"] button'
                    );
                    for (const el of els) {
                        const t = el.textContent.trim().toLowerCase();
                        if (keywords.some(k => t === k || t.startsWith(k))) { el.click(); return; }
                    }
                }""")
                await page.wait_for_timeout(400)
            except Exception:
                pass

        async def fetch_one(context, url: str) -> dict:
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=NETWORKIDLE_TIMEOUT)
                except Exception:
                    pass
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await _dismiss_popups_async(page)
                await page.wait_for_timeout(SCROLL_DELAY)
                html = await page.content()
                text = _extract_text(html)
                return {"url": page.url, "text": text, "segments": []}
            except Exception as e:
                return {"url": url, "text": f"Error fetching page: {e}", "segments": []}
            finally:
                try:
                    await page.close()
                except Exception:
                    pass

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent=self._USER_AGENT,
                    viewport={"width": 390, "height": 844},
                    is_mobile=True,
                    device_scale_factor=3,
                )
                try:
                    results = await asyncio.gather(*[fetch_one(context, url) for url in urls])
                    return list(results)
                finally:
                    await context.close()
            finally:
                await browser.close()

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

    def _screenshot_element(self, page, url: str, promo_index, extract_string: str, out_dir=None) -> tuple[str, list[str]]:
        """Full-width screenshot clipped to the element containing extract_string."""
        from pathlib import Path
        slug = re.sub(r"[^a-zA-Z0-9]", "_", url)[:60].strip("_")
        base = Path(out_dir or self.screenshot_dir)
        out = str(base / f"{slug}_{promo_index}.png")

        def fallback():
            try:
                self._dismiss_popups(page)
                page.screenshot(path=out, full_page=False)
            except Exception:
                pass
            return (out, [out])

        if not extract_string:
            return fallback()

        vw = page.viewport_size["width"]

        # Try progressively shorter substrings to find the element
        el = None
        for length in [len(extract_string), 40, 25]:
            snippet = extract_string[:length].strip()
            if len(snippet) < 10:
                continue
            try:
                loc = page.get_by_text(snippet, exact=False)
                if loc.count() > 0:
                    el = loc.first
                    break
            except Exception:
                continue

        if el is None:
            return fallback()

        try:
            el.scroll_into_view_if_needed()
            bbox = el.bounding_box()
            if not bbox:
                return fallback()
        except Exception:
            return fallback()

        # Walk up DOM to find a container that spans most of the page width
        try:
            bbox = page.evaluate("""(el) => {
                const vw = window.innerWidth;
                let cur = el;
                for (let i = 0; i < 8 && cur && cur.tagName !== 'BODY'; i++) {
                    const r = cur.getBoundingClientRect();
                    if (r.width >= vw * 0.7 || cur.tagName === 'SECTION' || cur.tagName === 'ARTICLE')
                        return { x: r.x, y: r.y, width: r.width, height: r.height };
                    cur = cur.parentElement;
                }
                const r = el.getBoundingClientRect();
                return { x: r.x, y: r.y, width: r.width, height: r.height };
            }""", el.element_handle()) or bbox
        except Exception:
            pass  # use original bbox

        self._dismiss_popups(page)

        # Inject a <mark> directly into the text node for true inline highlighting
        try:
            page.evaluate("""(snippet) => {
                const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                let node;
                for (const len of [40, 30, 20, 15, 10]) {
                    const short = snippet.substring(0, len);
                    const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                    while (node = tw.nextNode()) {
                        const idx = node.textContent.indexOf(short);
                        if (idx === -1) continue;
                        const end = Math.min(idx + snippet.length, node.textContent.length);
                        try {
                            const range = document.createRange();
                            range.setStart(node, idx);
                            range.setEnd(node, end);
                            const mark = document.createElement('mark');
                            mark.setAttribute('data-cc-hl', '1');
                            mark.style.cssText = 'background-color:#ffff00 !important;color:inherit !important;padding:1px 0;border-radius:2px;';
                            mark.appendChild(range.extractContents());
                            range.insertNode(mark);
                        } catch(e) {}
                        return;
                    }
                }
            }""", extract_string)
        except Exception:
            pass

        clip = {
            "x": 0,
            "y": max(0.0, bbox["y"] - 8),
            "width": float(vw),
            "height": min(bbox["height"] + 16, page.viewport_size["height"]),
        }
        try:
            page.screenshot(path=out, clip=clip)
            return (out, [out])
        except Exception:
            return fallback()
        finally:
            # Unwrap all injected <mark> elements
            try:
                page.evaluate("""() => {
                    document.querySelectorAll('mark[data-cc-hl]').forEach(mark => {
                        const parent = mark.parentNode;
                        while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
                        parent.removeChild(mark);
                        parent.normalize();
                    });
                }""")
            except Exception:
                pass

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
            self._dismiss_popups(page)
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

    def fetch_page_live(self, url: str) -> dict:
        """Like fetch_page() but does NOT close the page. Caller is responsible for closing.
        Returns dict with 'page' key containing the live Playwright Page object."""
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
            self._dismiss_popups(page)
            page.wait_for_timeout(SCROLL_DELAY)
            js_text, segments = self._build_js_segments(page)
            html = page.content()
            bs4_text = _extract_text(html)
            primary_text = js_text if js_text else bs4_text
            links = _extract_links(page, page.url, max_links=MAX_LINKS_TO_SCORE)
            return {"url": page.url, "text": primary_text, "links": links, "segments": segments, "page": page}
        except Exception as e:
            try:
                page.close()
            except Exception:
                pass
            return {"url": url, "text": f"Error fetching page: {e}", "links": [], "segments": [], "page": None}


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
            self._dismiss_popups(page)
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
    address = pub.get("address") or None
    print(f"[{name}] starting", flush=True)
    try:
        llm = make_llm_client(provider)
        search = make_search_tool(search_provider)
        agent = ResearchAgent(llm=llm, search=search, browser=BrowserTool(screenshot_dir=screenshot_dir), schema=schema)
        result = agent.run(query, start_url=url, pub_name=name, pub_address=address)
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
    def __init__(self, llm: OllamaClient, search: SearchTool, browser: BrowserTool, schema=None,
                 cache: SearchCache | None = None):
        self.llm = llm
        self.search = search
        self.browser = browser
        self.schema = schema
        self._cache = cache
        self._url_segments: dict[str, tuple[str, list[dict]]] = {}  # url → (page_text, segments)
        self._live_pages: list[dict] = []

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
                for i, promo in group:
                    anchor = promo.get("source_text_anchor", "")
                    primary_path, all_paths = self.browser._screenshot_element(
                        page, url, i, anchor
                    )
                    if primary_path:
                        promos[i]["screenshot_path"] = primary_path
                        print(f"  [screenshot] promo {i}: {primary_path} ({len(all_paths)} levels)")
            except Exception as e:
                print(f"  [screenshot] failed for {url}: {e}")
            finally:
                page.close()

        return json.dumps(promos, ensure_ascii=False)

    def _scrape_social_links(self) -> list[dict]:
        """Extract social media links from all live pages without any LLM call."""
        PLATFORMS = {
            "instagram.com":  ("instagram",  r"instagram\.com/([A-Za-z0-9_.]+)"),
            "facebook.com":   ("facebook",   r"facebook\.com/(?:pages/[^/]+/)?([A-Za-z0-9_.%-]+)"),
            "twitter.com":    ("twitter",    r"twitter\.com/([A-Za-z0-9_]+)"),
            "x.com":          ("x",          r"x\.com/([A-Za-z0-9_]+)"),
            "tiktok.com":     ("tiktok",     r"tiktok\.com/@?([A-Za-z0-9_.]+)"),
            "youtube.com":    ("youtube",    r"youtube\.com/(?:@|c/|channel/|user/)?([A-Za-z0-9_%-]+)"),
            "linkedin.com":   ("linkedin",   r"linkedin\.com/company/([A-Za-z0-9_-]+)"),
            "threads.net":    ("threads",    r"threads\.net/@?([A-Za-z0-9_.]+)"),
            "pinterest.com":  ("pinterest",  r"pinterest\.com/([A-Za-z0-9_]+)"),
            "snapchat.com":   ("snapchat",   r"snapchat\.com/add/([A-Za-z0-9_.-]+)"),
        }
        SKIP_PATHS = {"about", "help", "legal", "terms", "privacy", "login", "signup", "explore", "ads"}

        seen_urls: set[str] = set()
        results: list[dict] = []

        for record in self._live_pages:
            page = record.get("page")
            if not page:
                continue
            try:
                hrefs = page.evaluate("() => Array.from(document.querySelectorAll('a[href]')).map(a => a.href)")
            except Exception:
                continue
            for href in hrefs:
                href = href.split("?")[0].rstrip("/")
                if href in seen_urls:
                    continue
                for domain, (platform, pattern) in PLATFORMS.items():
                    if domain not in href:
                        continue
                    m = re.search(pattern, href, re.IGNORECASE)
                    username = m.group(1) if m else ""
                    if username.lower() in SKIP_PATHS:
                        continue
                    seen_urls.add(href)
                    results.append({"platform": platform, "url": href, "username": username})
                    break

        return results

    @staticmethod
    def _fuzzy_find(page_text: str, extract_string: str, threshold: float = 0.6) -> int:
        """Return char offset of best fuzzy match for extract_string in page_text, or -1.
        Both strings are normalised (collapse whitespace, strip non-alphanumeric) before comparison,
        but the returned offset is into the original page_text."""
        import difflib, re as _re
        if not extract_string or not page_text:
            return -1

        def _norm(s):
            s = _re.sub(r'[^a-zA-Z0-9 ]', ' ', s)
            return _re.sub(r'\s+', ' ', s).strip().lower()

        norm_extract = _norm(extract_string)
        n = len(extract_string)
        best_ratio, best_pos = 0.0, -1
        for i in range(0, len(page_text) - n + 1):
            ratio = difflib.SequenceMatcher(None, norm_extract, _norm(page_text[i:i + n]), autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio, best_pos = ratio, i
        return best_pos if best_ratio >= threshold else -1

    def _add_rich_screenshots(self, rich_json_str: str, website: str | None = None) -> str:
        """Post-process pub-site-mode JSON: inject website, screenshot each deal/event using live pages."""
        try:
            data = json.loads(rich_json_str)
            if not isinstance(data, dict):
                return rich_json_str
        except (json.JSONDecodeError, AttributeError):
            return rich_json_str

        if website:
            data["website"] = website

        social_links = self._scrape_social_links()
        if social_links:
            data["social_media"] = social_links
            print(f"  [social] found {len(social_links)} link(s): {[s['platform'] for s in social_links]}")

        ss_dir = self.browser.screenshot_dir or "tmp"
        os.makedirs(ss_dir, exist_ok=True)
        live_lookup = {record["url"]: record["page"] for record in self._live_pages if "url" in record and "page" in record}

        for label, items in [("deal", data.get("deals", [])), ("event", data.get("events", []))]:
            for i, item in enumerate(items):
                source_url = item.get("source_url", "")
                page = live_lookup.get(source_url)
                if not page and source_url:
                    # Sub-pages were fetched via HTTP for speed; open with Playwright now for screenshot
                    print(f"  [screenshot] opening page for screenshot: {source_url}")
                    sub_data = self.browser.fetch_page_live(source_url)
                    page = sub_data.get("page")
                    if page:
                        live_lookup[source_url] = page
                        self._live_pages.append({"url": source_url, "page": page, "segments": sub_data.get("segments", [])})
                if not page:
                    continue
                extract_string = item.get("extract_string", "")
                print(f"  [screenshot] {label}_{i}: extract_string={repr(extract_string[:40]) if extract_string else 'NONE'}")
                try:
                    primary, all_paths = self.browser._screenshot_element(
                        page, source_url, f"{label}_{i}", extract_string, out_dir=ss_dir
                    )
                    if primary:
                        item["screenshot_path"] = primary
                        item["screenshot_all_levels"] = all_paths
                        print(f"  [screenshot] {label}_{i}: saved {primary}")
                except Exception as e:
                    print(f"  [screenshot] {label}_{i} failed: {e}")

        return json.dumps(data, ensure_ascii=False)

    def _find_pub_home_url(self, pub_name: str, pub_address: str, query: str) -> str | None:
        """Single search to find a pub's official website URL."""
        AGGREGATORS = {
            "tripadvisor", "yelp", "designmynight", "timeout", "google",
            "facebook", "instagram", "twitter", "foursquare", "opentable",
            "bookatable", "squaremeal", "hardens", "matchpint", "pubsgalore",
        }
        search_query = f'"{pub_name}" pub website {pub_address}'
        print(f"  [find_home] Searching: {search_query}")
        results = self.search.search(search_query, max_results=5)
        for r in results:
            url = r.get("url", "")
            if not any(agg in url.lower() for agg in AGGREGATORS):
                return url
        return None

    def _crawl_pub_site(self, home_url: str, pub_name: str, pub_address: str) -> tuple[str, list[dict]]:
        """Load home page + top scored sub-pages. Returns (combined_text, live_page_records)."""
        import time as _time
        t0 = _time.time()

        # ── Home page (must be sequential — need links to score sub-pages) ──
        print(f"  [crawl] Loading home: {home_url}")
        home_data = self.browser.fetch_page_live(home_url)
        t1 = _time.time()
        print(f"  [timing] home: {t1 - t0:.1f}s")

        self._url_segments[home_data["url"]] = (home_data["text"], home_data["segments"])
        page_texts = [(home_data["url"], home_data["text"])]
        if home_data.get("page"):
            self._live_pages.append({
                "url": home_data["url"],
                "page": home_data["page"],
                "segments": home_data["segments"],
            })

        # Score links and pick top sub-pages (check cache first)
        cached_subpages = self._cache.get_subpages(home_data["url"]) if self._cache else None
        if cached_subpages is not None:
            print(f"  [cache] subpage hit: {home_data['url']} → {cached_subpages}")
            top_links = cached_subpages
        else:
            scored = _score_links(home_data["links"], home_data["url"])
            seen_urls = {home_data["url"]}
            top_links = []
            for link in scored:
                url = link["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    top_links.append(url)
                if len(top_links) >= MAX_SUBPAGES:
                    break
            if self._cache:
                self._cache.set_subpages(home_data["url"], top_links)

        # ── Sub-pages in parallel via async Playwright (JS-rendered, concurrent) ──
        if top_links:
            t2 = _time.time()
            print(f"  [crawl] Loading {len(top_links)} sub-pages in parallel (async Playwright): {top_links}")

            # asyncio.run() fails if a loop is already running (Playwright sync keeps one).
            # Run the coroutine in a dedicated thread with its own fresh event loop.
            def _run_async_in_thread(coro):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            with ThreadPoolExecutor(max_workers=1) as _ex:
                sub_results = _ex.submit(
                    _run_async_in_thread, self.browser._fetch_subpages_async(top_links)
                ).result()

            # Preserve top_links order; live pages opened lazily for screenshots
            result_by_url = {r["url"]: r for r in sub_results}
            for url in top_links:
                sub_data = result_by_url.get(url) or next(
                    (r for r in sub_results if r["url"].rstrip("/") == url.rstrip("/")), None
                ) or {"url": url, "text": "", "segments": []}
                if sub_data["text"] and not sub_data["text"].startswith("Error"):
                    self._url_segments[sub_data["url"]] = (sub_data["text"], sub_data["segments"])
                    page_texts.append((sub_data["url"], sub_data["text"]))

            t3 = _time.time()
            print(f"  [timing] {len(top_links)} sub-pages parallel async: {t3 - t2:.1f}s  (total crawl: {t3 - t0:.1f}s)")

        parts = [
            f"=== SOURCE: {url} ===\n{text[:MAX_SUBPAGE_CHARS]}"
            for url, text in page_texts
        ]
        combined = "\n\n".join(parts)
        # Strip zero-width and other invisible unicode characters that confuse the LLM
        combined = re.sub(r'[\u200b\u200c\u200d\u00ad\ufeff\u2060]', '', combined)
        if len(combined) > MAX_TOTAL_CHARS:
            combined = combined[:MAX_TOTAL_CHARS]
        return combined, self._live_pages

    def _build_rich_extraction_prompt(self, query: str, combined_text: str) -> tuple[str, str]:
        """Build (system, user) prompt for rich pub data extraction."""
        _schedule = {"recurring": True, "days": [], "date": "", "time_open": "", "time_close": ""}
        schema_example = {
            "pub_emoji": "",
            "description": {"text": "", "source_url": ""},
            "deals": [{"title": "", "description": "", "category": "", "schedule": _schedule, "source_url": "", "extract_string": ""}],
            "events": [{"title": "", "description": "", "category": "", "schedule": _schedule, "source_url": "", "extract_string": ""}],
            "opening_times": [{"day": "", "open": "", "close": ""}],
            "facilities": [{"name": "", "source_url": ""}],
        }
        emoji_options = (
            "🍺 classic/traditional pub  |  🎸 live music venue  |  📺 sports bar  |  "
            "🌿 beer garden/outdoor  |  🎯 games pub (darts/pool)  |  "
            "🥂 cocktail bar  |  🥃 whisky bar  |  🍷 wine bar  |  🍸 craft cocktails  |  "
            "🎭 events & entertainment  |  🏘️ neighbourhood local  |  🪩 late night bar  |  "
            "☕ brunch & coffee"
        )
        system = (
            "You are a pub data extraction assistant. The text below contains content from multiple web pages.\n"
            "Each page section starts with a header: === SOURCE: <url> ===\n\n"
            "Extract structured pub information and return a JSON object matching this schema exactly:\n"
            f"{json.dumps(schema_example, indent=2)}\n\n"
            "Field guidance:\n"
            f"- pub_emoji: pick ONE emoji that best matches the pub's overall vibe from this list:\n"
            f"  {emoji_options}\n"
            "- description.text: 1–2 sentence pub description covering vibe, style, what makes it special\n"
            "- deals[]: discounts, drink specials, happy hours, recurring price offers (NOT events)\n"
            "  - category: one of: '2-for-1', 'happy hour', 'meal deal', 'drink special', 'free item', 'discount', 'loyalty offer'\n"
            "  - schedule.recurring: true for weekly/regular deals, false for one-off\n"
            "  - schedule.days: array of lowercase day names e.g. ['monday','thursday'] — use for recurring\n"
            "  - schedule.date: ISO date e.g. '2025-06-14' — use for one-off only, leave empty if recurring\n"
            "  - schedule.time_open / time_close: 24h format e.g. '17:00', '21:00' — empty if all day\n"
            "- events[]: things happening at the pub — quiz nights, live music, themed nights, sport screenings\n"
            "  - category: one of: 'quiz night', 'live music', 'karaoke', 'comedy night', 'DJ set', 'open mic', 'sports screening', 'themed night', 'bingo', 'games night'\n"
            "  - schedule: same structure as deals — recurring weekly events use days[], one-off use date\n"
            "- opening_times[]: one object per day, all 7 days must be present\n"
            "  - day: lowercase day name e.g. 'monday'\n"
            "  - open / close: 24h time e.g. '10:00', '23:30' — empty string if closed or unknown\n"
            "- facilities[]: amenities such as beer garden, pool table, darts, TV screens, private hire,\n"
            "  dog-friendly, wheelchair access, Wi-Fi, live sports, etc.\n\n"
            "IMPORTANT RULES:\n"
            "1. source_url: copy EXACTLY from the nearest === SOURCE: <url> === header above the item\n"
            "2. extract_string: a verbatim snippet copied EXACTLY from the page text that UNIQUELY identifies\n"
            "   the item's main content block — used to locate the precise DOM element for screenshotting.\n"
            "   RULES:\n"
            "   - Must be 40–80 characters. Longer is better for uniqueness.\n"
            "   - Must appear verbatim in the source text (exact characters, exact order).\n"
            "   - MUST be unique on the page. Many sites repeat item names in footers, nav menus, or\n"
            "     sidebar lists — do NOT use text that could match those. Include surrounding context\n"
            "     (date, time, price, description) to make the string unique to the content card.\n"
            "   - ALWAYS include date/time/price when present — e.g. 'Board Games Tuesday 24th March 17:00'\n"
            "     is far better than just 'Board Games Tuesday' which may appear in a footer.\n"
            "   - Copy from the DETAILED description block of the item, not from compact summary lists.\n"
            "   - BAD:  'Board Games Tuesday' (too short, appears in footer nav)\n"
            "   - GOOD: 'Board Games Tuesday 24th March 17:00 - 22:00 Enjoy our selection of available board games'\n"
            "   - BAD:  'Mixr Mondays' (appears in footer)\n"
            "   - GOOD: 'Mixr Mondays Earn double points on Mondays when you use Stonegate app MiXR'\n"
            "3. Use [] for list fields with no data found, {} for object fields, empty string if unknown\n"
            "4. schedule.recurring MUST be a boolean (true or false), NOT a string\n"
            "5. Reply with ONLY valid JSON — no markdown fences, no explanation\n\n"
            "Example of a correct response:\n"
            '{"pub_emoji":"🍻","description":{"text":"A proper local pub with a courtyard and well-cooked British classics.","source_url":"https://example.com/"},'
            '"deals":[{"title":"Quiz Night","description":"Weekly pub quiz with great prizes.","category":"quiz night","schedule":{"recurring":true,"days":["thursday"],"date":"","time_open":"19:30","time_close":""},"source_url":"https://example.com/events/","extract_string":"Every Thursday at 7.30pm Quiz Night"}],'
            '"events":[{"title":"Open Mic Night","description":"Live performances by local artists.","category":"open mic","schedule":{"recurring":true,"days":["monday"],"date":"","time_open":"19:00","time_close":""},"source_url":"https://example.com/events/","extract_string":"Every Monday at 7pm Open Mic Night"}],'
            '"opening_times":[{"day":"monday","open":"12:00","close":"23:00"},{"day":"tuesday","open":"12:00","close":"23:00"},{"day":"wednesday","open":"12:00","close":"23:00"},{"day":"thursday","open":"12:00","close":"23:00"},{"day":"friday","open":"12:00","close":"00:00"},{"day":"saturday","open":"12:00","close":"00:00"},{"day":"sunday","open":"12:00","close":"22:30"}],'
            '"facilities":[{"name":"Beer Garden","source_url":"https://example.com/"},{"name":"Dog Friendly","source_url":"https://example.com/"}],'
            '"social_media":[{"platform":"instagram","url":"https://instagram.com/example","username":"example"}],'
            '"website":"https://example.com"}'
        )
        user = f"Query: {query}\n\nPage content:\n{combined_text}"
        return system, user

    def _run_pub_site(self, query: str, start_url: str | None,
                      pub_name: str | None, pub_address: str | None) -> str:
        """Multi-page pub site crawl with rich schema extraction."""
        if start_url:
            home_url = start_url
            print(f"  [pub_site] Using provided start URL: {home_url}")
        else:
            home_url = self._find_pub_home_url(pub_name or query, pub_address or "", query)
            if not home_url:
                print("  [pub_site] Could not find pub website")
                return json.dumps({"pub_emoji": "", "description": {}, "deals": [], "events": [], "opening_times": [], "facilities": []})
            print(f"  [pub_site] Found home URL: {home_url}")

        combined_text, live_pages = self._crawl_pub_site(home_url, pub_name or "", pub_address or "")
        print(f"  [pub_site] Crawled {len(live_pages)} pages, {len(combined_text)} chars total")

        system, user = self._build_rich_extraction_prompt(query, combined_text)

        _schedule = {"recurring": True, "days": [], "date": "", "time_open": "", "time_close": ""}
        rich_schema = {
            "pub_emoji": "",
            "description": {"text": "", "source_url": ""},
            "deals": [{"title": "", "description": "", "category": "", "schedule": _schedule, "source_url": "", "extract_string": ""}],
            "events": [{"title": "", "description": "", "category": "", "schedule": _schedule, "source_url": "", "extract_string": ""}],
            "opening_times": [{"day": "", "open": "", "close": ""}],
            "facilities": [{"name": "", "source_url": ""}],
        }
        coerced = _coerce_to_schema(rich_schema)
        user_prompt = user
        import time as _time
        last_response = ""
        for attempt in range(3):
            t_llm_start = _time.time()
            response = self.llm.chat(system, user_prompt)
            print(f"  [timing] LLM: {_time.time() - t_llm_start:.1f}s")
            last_response = response
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())
            clean = re.sub(r"<think>.*?</think>\s*", "", clean, flags=re.DOTALL).strip()

            # Save prompt + response to tmp/ for analysis
            try:
                import time as _time
                os.makedirs("tmp", exist_ok=True)
                slug = re.sub(r"[^a-zA-Z0-9]", "_", (pub_name or query or "run"))[:40].strip("_")
                ts = _time.strftime("%Y%m%d_%H%M%S")
                dump_path = f"tmp/{slug}_{ts}_attempt{attempt}.json"
                with open(dump_path, "w", encoding="utf-8") as _f:
                    json.dump({"system": system, "user": user_prompt, "response": response}, _f, indent=2, ensure_ascii=False)
                print(f"  [tmp] saved prompt+response → {dump_path}")
            except Exception as _e:
                print(f"  [tmp] could not save dump: {_e}")

            if _JSONSCHEMA_AVAILABLE:
                try:
                    jsonschema.validate(json.loads(clean), coerced)
                    t_ss_start = _time.time()
                    result = self._add_rich_screenshots(clean, home_url)
                    print(f"  [timing] screenshots: {_time.time() - t_ss_start:.1f}s")
                    return result
                except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                    if attempt < 2:
                        user_prompt = user + f"\n\nYour previous response contained invalid JSON:\n{clean}\n\nError: {e}\n\nFix the JSON and return ONLY the corrected valid JSON."
            else:
                try:
                    json.loads(clean)
                    t_ss_start = _time.time()
                    result = self._add_rich_screenshots(clean, home_url)
                    print(f"  [timing] screenshots: {_time.time() - t_ss_start:.1f}s")
                    return result
                except json.JSONDecodeError as e:
                    if attempt < 2:
                        user_prompt = user + f"\n\nYour previous response contained invalid JSON:\n{clean}\n\nError: {e}\n\nFix the JSON and return ONLY the corrected valid JSON."
        t_ss_start = _time.time()
        result = self._add_rich_screenshots(last_response, home_url)  # graceful fallback
        print(f"  [timing] screenshots: {_time.time() - t_ss_start:.1f}s")
        return result

    def run(self, query: str, start_url: str | None = None,
            pub_name: str | None = None, pub_address: str | None = None,
            pub_site_mode: bool = False) -> str:
        gathered_info: list[str] = []
        self._url_segments = {}
        self._live_pages = []
        self.browser.start()
        try:
            if pub_site_mode:
                return self._run_pub_site(query, start_url, pub_name, pub_address)
            result = self._run(query, start_url, gathered_info, pub_name, pub_address)
            if self.browser.screenshot_dir and self.schema:
                result = self._add_promo_screenshots(result)   # browser still open here
            return result
        finally:
            for record in self._live_pages:
                page = record.get("page")
                if page:
                    try:
                        page.close()
                    except Exception:
                        pass
            self.browser.close()

    def _run(self, query: str, start_url: str | None, gathered_info: list[str],
             pub_name: str | None = None, pub_address: str | None = None) -> str:
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

        # Step 1+2: Build search query and search
        parts = [pub_name or query]
        parts.append("website")
        if pub_address:
            parts.append(pub_address)
        search_query = " ".join(parts)
        print(f"[1/8] Searching: {search_query}")
        results = self.search.search(search_query)

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
    parser.add_argument("--pub-name", default=None,
                        help="Pub name used to build search query: '{name} website {address}'")
    parser.add_argument("--pub-address", default=None,
                        help="Pub address used to build search query alongside --pub-name")
    parser.add_argument("--pub-site-mode", action="store_true", default=False,
                        help="Use multi-page pub site crawl with rich schema extraction")
    parser.add_argument("--cache-file", default=None,
                        help="JSON file for caching search results and subpage URL order")
    parser.add_argument("--log-file", default=None,
                        help="Append all stdout output to this file in addition to printing it")
    args = parser.parse_args()

    if args.log_file:
        import io
        _log_fh = open(args.log_file, "a", encoding="utf-8")

        class _Tee(io.TextIOBase):
            def __init__(self, primary, secondary):
                self._p = primary
                self._s = secondary
            def write(self, s):
                self._p.write(s)
                self._s.write(s)
                return len(s)
            def flush(self):
                self._p.flush()
                self._s.flush()

        sys.stdout = _Tee(sys.__stdout__, _log_fh)
        sys.stderr = _Tee(sys.__stderr__, _log_fh)

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
    cache = SearchCache(args.cache_file) if args.cache_file else None
    if cache:
        search = CachedSearchTool(search, cache)
    model_label = {"minimax": MINIMAX_MODEL, "claude": CLAUDE_MODEL, "ollama": OLLAMA_MODEL}.get(args.provider, args.provider)

    print(f"\nLLM     : {args.provider} ({model_label})")
    print(f"Search  : {args.search_provider} ({type(search).__name__})")
    if cache:
        print(f"Cache   : {args.cache_file}")
    print(f"Query   : {query}")
    if schema:
        print(f"Schema  : {json.dumps(schema)}")
    if args.screenshot_dir:
        print(f"Screenshots: {args.screenshot_dir}")
    print("=" * 60)
    agent = ResearchAgent(llm=llm, search=search, browser=BrowserTool(screenshot_dir=args.screenshot_dir), schema=schema, cache=cache)
    answer = agent.run(query, start_url=args.start_url,
                       pub_name=args.pub_name, pub_address=args.pub_address,
                       pub_site_mode=args.pub_site_mode)
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
