"""
Data Analyst Agent
Produces structured analytics from cleaned IG data.
"""
import json
import logging
from typing import Annotated
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

logger = logging.getLogger(__name__)


def prepare_analysis_prompt(cleaned_json: str) -> str:
    """
    Prepare a structured text block summarising all posts and comments
    for the LLM to analyse.
    """
    data = json.loads(cleaned_json)
    lines = [f"Instagram account: @{data['username']}", f"Data fetched at: {data['fetched_at']}", ""]
    post_count = len(data.get("posts", []))
    total_comments = sum(len(p.get("comments", [])) for p in data.get("posts", []))
    logger.info(
        f"[DataAnalyst] Preparing analysis prompt for @{data['username']} "
        f"({post_count} posts, {total_comments} comments total)"
    )
    for i, post in enumerate(data.get("posts", []), 1):
        lines.append(f"--- POST {i} ---")
        lines.append(f"Date: {post['timestamp']}")
        lines.append(f"Caption: {post['caption']}")
        lines.append(f"Hashtags: {', '.join(post.get('hashtags', []))}")
        lines.append(f"Likes: {post['likesCount']}  |  Comments: {post['commentsCount']}")
        lines.append("Comments:")
        # Cap at 30 comments per post to keep the prompt within token limits
        # while still giving the LLM a representative sample
        for c in post.get("comments", [])[:30]:
            lines.append(f"  @{c['ownerUsername']}: {c['text']}")
        lines.append("")
    prompt = "\n".join(lines)
    # ~40 000 chars ≈ 10 000 tokens — well inside the GPT-5 context window while
    # leaving enough room for the system prompt and the structured JSON response.
    MAX_PROMPT_CHARS = 40_000
    if len(prompt) > MAX_PROMPT_CHARS:
        # Truncate at the last complete post boundary to avoid a partial entry at the cut point
        cut = prompt.rfind("\n--- POST ", 0, MAX_PROMPT_CHARS)
        prompt = prompt[:cut if cut != -1 else MAX_PROMPT_CHARS] + "\n\n[... content truncated to fit model context window ...]"
        logger.warning(
            f"[DataAnalyst] Prompt truncated at post boundary ({len(prompt)} chars)"
        )
    logger.info(
        f"[DataAnalyst] Analysis prompt ready \u2014 {len(prompt)} chars, "
        f"{post_count} posts, {total_comments} comments"
    )
    return prompt


@tool
def prepare_ig_analysis_prompt(
    cleaned_json: Annotated[str, Field(description="Cleaned Instagram JSON string from DataCleaningAgent")],
) -> str:
    """
    Convert cleaned Instagram JSON into a compact, human-readable text prompt
    suitable for the DataAnalystAgent. Call this before passing data to the analyst.
    """
    return prepare_analysis_prompt(cleaned_json)


def create_data_analyst_agent(chat_client: OpenAIChatClient) -> Agent:
    """Return a Microsoft Agent Framework Agent for Instagram content analysis.

    This agent has NO tools — the LLM performs all reasoning directly from the
    pre-formatted text prompt produced by prepare_analysis_prompt(). It outputs
    a single structured JSON object that the orchestrator validates and saves.
    """
    return Agent(
        name="DataAnalystAgent",
        client=chat_client,
        instructions="""You are the Data Analyst Agent specialising in health & beauty Instagram content.

You will receive a pre-formatted text summary of Instagram posts and comments.
Analyse the content and return a JSON object with EXACTLY this schema — no extra text, no markdown fences:
{
  "username": "string",
  "analysis_date": "ISO-8601 date",
  "total_posts_analysed": number,
  "keynotes_summary": "paragraph summarising major themes",
  "health_beauty_products": [
    {"product_name": "string", "brand": "string or null", "mention_count": number, "context": "string"}
  ],
  "sentiment_analysis": {
    "overall": "Positive|Neutral|Negative|Mixed",
    "score": number between -1.0 and 1.0,
    "breakdown": {"positive_pct": number, "neutral_pct": number, "negative_pct": number},
    "notable_comments": ["string", ...]
  },
  "top_hashtags": [{"tag": "string", "count": number}],
  "engagement_insights": "paragraph about likes, comments trends",
  "recommendations": ["string", ...]
}
Return only the JSON — no markdown fences, no extra text before or after.""",
        tools=[],
    )
