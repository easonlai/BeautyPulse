"""
IG Accounts Watching Agent
Fetches posts and comments from specified Instagram accounts via Apify.
The merge of posts + comments is done in Python (not by the LLM) to avoid
the model stalling while trying to emit large JSON blobs.
"""
import os
import json
import logging
import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import Annotated
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from apify_client import ApifyClient
from pydantic import Field

# Apify actor IDs used for scraping (public actors maintained by Apify)
IG_POST_SCRAPER_ID = "apify/instagram-post-scraper"
IG_COMMENT_SCRAPER_ID = "apify/instagram-comment-scraper"
logger = logging.getLogger(__name__)


def _fetch_post_comments(client: ApifyClient, short_code: str, comments_limit: int) -> list:
    """Fetch comments for a single post. Returns empty list on failure."""
    try:
        comment_run = client.actor(IG_COMMENT_SCRAPER_ID).call(
            run_input={
                "directUrls": [f"https://www.instagram.com/p/{short_code}/"],
                "resultsLimit": comments_limit,
            },
            timeout_secs=120,
        )
        comments = [
            {
                "id": c.get("id", ""),
                "text": c.get("text", ""),
                "ownerUsername": c.get("ownerUsername", ""),
                "timestamp": c.get("timestamp", ""),
                "likesCount": c.get("likesCount") or 0,
            }
            for c in client.dataset(comment_run["defaultDatasetId"]).iterate_items()
        ]
        logger.info(f"[IGWatcher] Post {short_code}: fetched {len(comments)} comments")
        return comments
    except Exception as exc:
        logger.warning(f"[IGWatcher] Could not fetch comments for {short_code}: {exc}")
        return []


@tool
def fetch_ig_data(
    username: Annotated[str, Field(description="Instagram username to scrape")],
    results_limit: Annotated[int, Field(description="Max number of posts to fetch (overrides IG_RESULTS_LIMIT env var)")] = None,
    comments_limit: Annotated[int, Field(description="Max number of comments per post (overrides IG_COMMENTS_LIMIT env var)")] = None,
    lookback_days: Annotated[int, Field(description="Only fetch posts published within this many days (overrides IG_LOOKBACK_DAYS env var)")] = None,
) -> str:
    """
    Fetch recent Instagram posts AND their comments for a given username,
    merge them in Python, and return a single JSON string:
    {"username": ..., "fetched_at": ..., "posts": [{..., "comments": [...]}]}
    """
    apify_token = os.environ.get("APIFY_API_TOKEN")
    if not apify_token:
        raise EnvironmentError("APIFY_API_TOKEN environment variable is not set")

    if lookback_days is None:
        lookback_days = int(os.environ.get("IG_LOOKBACK_DAYS", "7"))
    if results_limit is None:
        results_limit = int(os.environ.get("IG_RESULTS_LIMIT", "20"))
    if comments_limit is None:
        comments_limit = int(os.environ.get("IG_COMMENTS_LIMIT", "50"))

    from_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(
        f"[IGWatcher] Fetching up to {results_limit} posts for @{username} "
        f"(lookback: {lookback_days} days, fromDate: {from_date})"
    )

    try:
        client = ApifyClient(apify_token)

        # ── Step 1: fetch posts ───────────────────────────────────────────────
        run = client.actor(IG_POST_SCRAPER_ID).call(
            run_input={
                "username": [username],
                "resultsLimit": results_limit,
                "fromDate": from_date,
            },
            timeout_secs=180,
        )
        posts = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            posts.append({
                "id": item.get("id", ""),
                "shortCode": item.get("shortCode", ""),
                "timestamp": item.get("timestamp", ""),
                "caption": item.get("caption", ""),
                "likesCount": item.get("likesCount", 0),
                "commentsCount": item.get("commentsCount", 0),
                "url": item.get("url", ""),
                "imageUrl": item.get("displayUrl", ""),
                "hashtags": item.get("hashtags", []),
                "mentions": item.get("mentions", []),
                "comments": [],
            })
        logger.info(f"[IGWatcher] Fetched {len(posts)} posts for @{username}")

        # ── Step 2: fetch comments per post in parallel ─────────────────────
        # Only posts that have a shortCode can be looked up in the comment scraper.
        # max_workers=5 balances Apify concurrency limits against wall-clock time.
        posts_with_code = [p for p in posts if p.get("shortCode")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            future_to_post = {
                pool.submit(_fetch_post_comments, client, post["shortCode"], comments_limit): post
                for post in posts_with_code
            }
            for future in concurrent.futures.as_completed(future_to_post):
                post = future_to_post[future]
                post["comments"] = future.result()

        result = {
            "username": username,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "posts": posts,
        }
        logger.info(f"[IGWatcher] Done — {len(posts)} posts with comments merged for @{username}")
        # ensure_ascii=False preserves multilingual captions and comments (CJK, Arabic, etc.)
        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        logger.exception(f"[IGWatcher] Failed to fetch data for @{username}: {exc}")
        raise


def create_ig_watcher_agent(chat_client: OpenAIChatClient) -> Agent:
    """Return a Microsoft Agent Framework Agent that wraps fetch_ig_data.
    The LLM in this agent acts purely as a dispatcher — no analysis is done here.
    """
    return Agent(
        name="IGWatcherAgent",
        client=chat_client,
        instructions="""You are the IG Accounts Watching Agent.
Your job:
1. Call fetch_ig_data with the given Instagram username.
2. Return the JSON string exactly as returned by the tool — no extra text, no changes.""",
        tools=[fetch_ig_data],
    )
