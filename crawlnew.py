#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
)
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from pydantic import BaseModel, Field

# ── Defaults & paths ──────────────────────────────────────────────────
URL_FILE = "urls.txt"
TMP_DIR = Path("./tmp_results")
HEADLESS = True
DEFAULT_BATCH_SIZE = 3
DEFAULT_PRUNE_THRESHOLD = 0.2
DEFAULT_BM25_THRESHOLD = 1.1


# ── Utility helpers ───────────────────────────────────────────────────
def safe_filename(url: str, max_len: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{urlparse(url).netloc}{urlparse(url).path}")
    return slug.strip("_")[:max_len]


def load_urls(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def jparse(txt: str | None) -> Any:
    if not txt:
        return []
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return []


# ── Token helper ──────────────────────────────────────────────────────
def to_total_tokens(usage: Any) -> int:
    """Return token usage as an int, tolerant of different provider objects."""
    if usage is None:
        return 0
    if isinstance(usage, (int, float)):
        return int(usage)
    if hasattr(usage, "total_tokens"):
        return int(usage.total_tokens or 0)
    if isinstance(usage, dict) and "total_tokens" in usage:
        return int(usage["total_tokens"] or 0)
    try:
        return int(usage)
    except (TypeError, ValueError):
        return 0


# ── Product schema & LLM strategy ─────────────────────────────────────
class Product(BaseModel):
    name: str = Field(..., description="Exact product name as shown on the page.")
    price: str | None = Field(
        None,
        description="Total (discounted) price, or null if no price is visible.",
    )


INSTRUCTION = """
You will receive the HTML of an e-commerce results page.

TASK
----
Extract every iphone as object list from content MAIN MENU:

skip ads, non related content, hidden elements, etc.

OUTPUT
"name":  "<full product name as shown on the page>",
"price": "<total price as string number>"  EXTRACT FULL PRICE, if discounted, show the discounted price

Rules
Use an empty string "" when the price is truly absent.
"""


class CleaningLLMStrategy(LLMExtractionStrategy):
    """Drops elements that have {"error": true} from the parsed result."""

    def postprocess_result(self, parsed_result: Any, page_context):  # noqa: D401
        if isinstance(parsed_result, list):
            return [
                x
                for x in parsed_result
                if not (isinstance(x, dict) and x.get("error") is True)
            ]
        return parsed_result


def make_llm_strategy() -> LLMExtractionStrategy:
    return CleaningLLMStrategy(
        llm_config=LLMConfig(
            provider="deepseek/deepseek-chat", api_token=os.getenv("DEEPSEEK_API")
        ),
        schema=Product.model_json_schema(),
        extraction_type="schema",
        instruction=INSTRUCTION,
        input_format="raw-markdown",
        extra_args={"temperature": 0.0, "max_tokens": 900},
    )


# ── Worker (one process per URL) ──────────────────────────────────────
def worker(  # noqa: PLR0913
    url: str,
    out_path: str,
    raw: bool,
    bm25: bool,
    query: str,
    q: mp.Queue | None = None,
) -> None:
    """Fetch page → LLM extraction → save JSON → put token count on queue."""

    async def _run() -> None:
        # Select content-filter strategy
        if raw:
            content_filter = None
        elif bm25:
            content_filter = BM25ContentFilter(
                user_query=query or None, bm25_threshold=DEFAULT_BM25_THRESHOLD
            )
        else:
            content_filter = PruningContentFilter(
                threshold=DEFAULT_PRUNE_THRESHOLD,
                threshold_type="dynamic",
                min_word_threshold=1,
            )

        md_gen = DefaultMarkdownGenerator(content_filter=content_filter)
        strategy = make_llm_strategy()
        cfg = CrawlerRunConfig(
            markdown_generator=md_gen,
            extraction_strategy=strategy,
            cache_mode=CacheMode.DISABLED,
            remove_overlay_elements=False,
            excluded_tags=["nav", "header", "footer", "aside", "noscript", "iframe"]
            
        )

        async with AsyncWebCrawler(
            config=BrowserConfig(headless=HEADLESS, verbose=False, browser_type="chromium")
        ) as crawler:
            res = await crawler.arun(url, cfg)

        # Persist extraction to disk
        data = jparse(res.extracted_content) if res.success else []
        Path(out_path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ── Token accounting ──────────────────────────────────────────
        raw_usage = strategy.total_usage or 0
        tokens = to_total_tokens(raw_usage)
        calls = len(strategy.usages)
        print(
            f"✓ {url} | items: {len(data)} | tokens: {tokens} ({calls} chunk calls)",
            flush=True,
        )

        # Optional: detailed provider report
        # strategy.show_usage()

        if q is not None:
            q.put(tokens)  # guaranteed int

    asyncio.run(_run())


# ── Batch controller ─────────────────────────────────────────────────
def run_batch(
    batch: List[str], args: argparse.Namespace
) -> tuple[Dict[str, List[dict]], int]:
    """Spawn one worker per URL, collect JSON and token totals."""
    q: mp.Queue = mp.Queue()
    procs: List[mp.Process] = []
    fmap: Dict[str, Path] = {}

    for url in batch:
        out_file = TMP_DIR / f"{safe_filename(url)}.json"
        fmap[url] = out_file
        p = mp.Process(
            target=worker,
            args=(url, str(out_file), args.raw, args.use_bm25, args.query, q),
            daemon=True,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Merge extracted JSON
    merged: Dict[str, List[dict]] = {
        u: jparse(fpath.read_text("utf-8")) if fpath.exists() else []
        for u, fpath in fmap.items()
    }

    # Sum token counts
    batch_tokens = 0
    while not q.empty():
        batch_tokens += q.get()

    return merged, batch_tokens


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Multiprocess e-commerce crawler")
    ap.add_argument("--urls", default=URL_FILE)
    ap.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="# parallel processes to spawn",
    )
    ap.add_argument("--raw", action="store_true", help="Skip content filtering")
    ap.add_argument(
        "--use-bm25", action="store_true", help="Use BM25 filter instead of pruning"
    )
    ap.add_argument("--query", default="all iphones and their price", help="BM25 query")
    ap.add_argument("--out", default="combined_results.json")
    return ap.parse_args()


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    logging.getLogger("crawl4ai").setLevel(logging.ERROR)
    mp.freeze_support()  # Windows safety

    args = parse_args()
    urls = load_urls(args.urls)
    if not urls:
        print("No URLs in list – aborting")
        sys.exit(1)

    TMP_DIR.mkdir(exist_ok=True)
    overall: Dict[str, List[dict]] = {}
    total_tokens = 0
    t_start = time.perf_counter()

    for i in range(0, len(urls), args.batch):
        chunk = urls[i : i + args.batch]
        print(f"\n📦 Batch {i // args.batch + 1} ▸ {len(chunk)} urls …", flush=True)
        chunk_data, chunk_tokens = run_batch(chunk, args)
        overall.update(chunk_data)
        total_tokens += chunk_tokens

    # Drop items with {"error": true}
    cleaned = {
        u: [itm for itm in lst if not (isinstance(itm, dict) and itm.get("error"))]
        for u, lst in overall.items()
    }

    out_path = TMP_DIR / args.out
    out_path.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.perf_counter() - t_start
    print(
        f"\n✅ Done in {elapsed:.2f}s | Combined JSON → {out_path}\n"
        f"🧾 Total usage: {total_tokens} tokens across {len(urls)} URL calls"
    )


if __name__ == "__main__":
    main()
