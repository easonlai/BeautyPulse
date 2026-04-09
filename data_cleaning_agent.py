"""
Data Cleaning Agent
Cleans and normalises raw IG data produced by the IG Watcher Agent.
"""
import json
import logging
import re
from datetime import datetime, timezone
from typing import Annotated
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

logger = logging.getLogger(__name__)
# Pre-compiled regex for stripping HTML tags from captions and comment text
_strip_html = re.compile(r"<[^>]+>")


def _normalise_timestamp(ts: str) -> str:
    """Parse an arbitrary timestamp string and return a UTC ISO-8601 string."""
    if not ts:
        return ts
    try:
        # Replace trailing 'Z' with '+00:00' for broad Python compatibility
        # (datetime.fromisoformat natively supports 'Z' from 3.11+, but this
        # keeps the function portable across Python versions)
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        logger.warning(f"[DataCleaner] Could not normalise timestamp: {ts!r}")
        return ts


@tool
def clean_ig_data(
    raw_json: Annotated[str, Field(description="Raw JSON string from IGWatcherAgent")],
) -> str:
    """
    Clean and normalise raw Instagram data:
    - Remove null / empty fields
    - Normalise timestamps to ISO-8601 UTC
    - Strip HTML tags from captions and comments
    - Deduplicate comments by id
    - Return cleaned JSON string
    """
    data = json.loads(raw_json)

    cleaned_posts = []
    seen_post_ids: set[str] = set()  # tracks post ids already processed to drop API duplicates
    for post in data.get("posts", []):
        pid = post.get("id", "")
        if not pid or pid in seen_post_ids:
            continue
        seen_post_ids.add(pid)

        caption = _strip_html.sub("", post.get("caption") or "").strip()
        cleaned_comments = []
        seen_comment_ids: set[str] = set()  # tracks comment ids to drop per-post duplicates
        for c in post.get("comments", []):
            cid = c.get("id", "")
            if not cid or cid in seen_comment_ids:
                continue
            seen_comment_ids.add(cid)
            text = _strip_html.sub("", c.get("text") or "").strip()
            # Skip comments whose text is empty after HTML-stripping (e.g. emoji-only
            # comments that the API returned as HTML entities that strip to nothing)
            if text:
                cleaned_comments.append({
                    "id": cid,
                    "text": text,
                    "ownerUsername": c.get("ownerUsername", ""),
                    "timestamp": _normalise_timestamp(c.get("timestamp", "")),
                    "likesCount": c.get("likesCount") or 0,
                })

        cleaned_posts.append({
            "id": pid,
            "shortCode": post.get("shortCode", ""),
            "timestamp": _normalise_timestamp(post.get("timestamp", "")),
            "caption": caption,
            "likesCount": post.get("likesCount") or 0,
            # Use the actual deduplicated comment count, not the stale API value
            "commentsCount": len(cleaned_comments),
            "url": post.get("url", ""),
            "imageUrl": post.get("imageUrl", ""),
            "hashtags": [h for h in post.get("hashtags", []) if h],
            "mentions": [m for m in post.get("mentions", []) if m],
            "comments": cleaned_comments,
        })

    logger.info(
        f"[DataCleaner] Cleaned {len(cleaned_posts)} posts "
        f"(dropped {len(data.get('posts', [])) - len(cleaned_posts)} duplicates/empty) "
        f"— total comments kept: {sum(len(p['comments']) for p in cleaned_posts)}"
    )
    result = {
        "username": data.get("username", ""),
        "fetched_at": data.get("fetched_at", ""),
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "posts": cleaned_posts,
    }
    # ensure_ascii=False preserves multilingual captions and comments (CJK, Arabic, etc.)
    return json.dumps(result, ensure_ascii=False)


def create_data_cleaning_agent(chat_client: OpenAIChatClient) -> Agent:
    """Return a Microsoft Agent Framework Agent that wraps clean_ig_data.
    Pure-Python cleaning — no LLM reasoning involved; the LLM acts only as
    a dispatcher that passes the raw JSON string straight to the tool.
    """
    return Agent(
        name="DataCleaningAgent",
        client=chat_client,
        instructions="""You are the Data Cleaning Agent.
You receive raw Instagram JSON from the IG Watcher Agent.
Call clean_ig_data with the full raw JSON string.
Return the cleaned JSON exactly as returned by the tool — no extra text.""",
        tools=[clean_ig_data],
    )
