"""
Headless Orchestrator
Runs the full analytics pipeline by calling each sub-agent directly
(IGWatcher → DataCleaner → DataAnalyst → EmailAgent) without going through
the SupervisorAgent. This gives precise step-level progress tracking.

Entry points:
  python orchestrator.py              — start recurring scheduler
  python orchestrator.py once <user>  — run pipeline once for <user>, then exit
"""
import os
import sys
import json
import asyncio
import logging
import threading
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from agent_framework.openai import OpenAIChatClient
from ig_watcher_agent import fetch_ig_data
from data_cleaning_agent import clean_ig_data
from data_analyst_agent import prepare_analysis_prompt, create_data_analyst_agent
from email_agent import create_email_agent

load_dotenv(override=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PROGRESS_FILE = RESULTS_DIR / "progress.json"
# Shared lock that prevents concurrent Streamlit UI threads and scheduler coroutines
# from corrupting progress.json with interleaved writes.
_PROGRESS_LOCK = threading.Lock()


def _save_progress(state: dict):
    """Write progress state to disk atomically under a threading lock.
    Writes to a temp file then renames to prevent corrupt reads on power-loss
    or mid-write process kills.
    """
    with _PROGRESS_LOCK:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=PROGRESS_FILE.parent, prefix=".progress_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, PROGRESS_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def _load_progress() -> dict:
    """Load progress state from disk; returns an empty state dict if the file is
    missing, corrupt, or unreadable rather than crashing."""
    with _PROGRESS_LOCK:
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"[Orchestrator] progress.json unreadable ({exc}); resetting")
    return {"runs":[]}


def _build_chat_client() -> OpenAIChatClient:
    """Construct an OpenAIChatClient (Azure OpenAI) from environment variables.

    Auth priority:
      1. AZURE_OPENAI_API_KEY — direct API key (preferred for local dev)
      2. DefaultAzureCredential — managed identity / az login (preferred for prod)

    Required env var:
      AZURE_OPENAI_ENDPOINT          — e.g. https://<name>.openai.azure.com/
    Optional env vars:
      AZURE_OPENAI_API_KEY           — API key (omit to use DefaultAzureCredential)
      AZURE_OPENAI_DEPLOYMENT_NAME   — model deployment name (default: gpt-5)
      AZURE_OPENAI_API_VERSION       — API version (default: SDK built-in)
    """
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    # Only pass api_version if explicitly set — otherwise the SDK uses its
    # built-in default ("preview"), which always resolves to the latest
    # supported preview version on Azure OpenAI.
    common_kwargs = dict(azure_endpoint=endpoint, model=deployment)
    if api_version:
        common_kwargs["api_version"] = api_version

    if api_key:
        logger.info(
            f"[Orchestrator] Building OpenAIChatClient — endpoint={endpoint}, "
            f"model={deployment}, api_version={api_version or '(SDK default)'}, auth=api_key"
        )
        return OpenAIChatClient(**common_kwargs, api_key=api_key)
    logger.info(
        f"[Orchestrator] Building OpenAIChatClient — endpoint={endpoint}, "
        f"model={deployment}, api_version={api_version or '(SDK default)'}, auth=DefaultAzureCredential"
    )
    return OpenAIChatClient(**common_kwargs, credential=DefaultAzureCredential())


def _update_step(progress: dict, run_record: dict, step: str, status: str):
    """Update a single step status and immediately persist to disk."""
    run_record["steps"][step] = status
    _save_progress(progress)
    logger.info(f"[Orchestrator] Step '{step}' → {status}")


def _extract_json(text: str) -> dict:
    """Extract the outermost JSON object from a text string."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(
            f"No JSON object found in agent output. "
            f"Raw (first 500 chars): {text[:500]!r}"
        )
    return json.loads(text[start : end + 1])


async def run_pipeline(username: str, recipients: list[str] = None) -> dict:
    """Run the end-to-end analytics pipeline for a single Instagram username.

    Steps:
      1. IGWatcher  — scrapes posts + comments via Apify (no LLM)
      2. DataCleaner — deduplicates and normalises the raw data (no LLM)
      3. DataAnalyst — LLM-powered analysis; returns structured JSON report
      4. EmailAgent  — generates PDF and emails it (LLM + Mailtrap; optional)

    Progress is persisted to disk after every step so the UI can show live status.
    Returns the run_record dict regardless of success or failure.
    """
    username = username.strip()
    logger.info(f"[Orchestrator] Starting pipeline for @{username}")
    started_dt = datetime.now(timezone.utc)
    progress = _load_progress()

    _steps = {
        "ig_watcher": "pending",
        "data_cleaner": "pending",
        "data_analyst": "pending",
    }
    if recipients:
        _steps["email_agent"] = "pending"

    run_record = {
        "username": username,
        "started_at": started_dt.isoformat(),
        "status": "running",
        "steps": _steps,
        "result_file": None,
    }
    # Cap history to the most recent 200 runs to keep progress.json manageable
    MAX_RUNS = 200
    progress["runs"].append(run_record)
    progress["runs"] = progress["runs"][-MAX_RUNS:]
    _save_progress(progress)

    total_steps = 4 if recipients else 3

    try:
        # ── Step 1: IG Watcher (pure Python — Apify, no LLM) ─────────────────
        logger.info(f"[Orchestrator] Step 1/{total_steps} — fetch_ig_data for @{username}")
        _update_step(progress, run_record, "ig_watcher", "running")
        raw_json = fetch_ig_data(username)
        raw_data = json.loads(raw_json)
        logger.info(f"[Orchestrator] fetch_ig_data done — {len(raw_data.get('posts', []))} posts")
        _update_step(progress, run_record, "ig_watcher", "done")

        # ── Step 2: Data Cleaner (pure Python — no LLM) ───────────────────────
        logger.info(f"[Orchestrator] Step 2/{total_steps} — clean_ig_data")
        _update_step(progress, run_record, "data_cleaner", "running")
        cleaned_json = clean_ig_data(raw_json)
        cleaned_data = json.loads(cleaned_json)
        logger.info(f"[Orchestrator] clean_ig_data done — {len(cleaned_data.get('posts', []))} cleaned posts")
        _update_step(progress, run_record, "data_cleaner", "done")

        # ── Step 3: Data Analyst (LLM analyses the compact text prompt) ───────
        logger.info(f"[Orchestrator] Step 3/{total_steps} — DataAnalystAgent (LLM)")
        _update_step(progress, run_record, "data_analyst", "running")
        # Build the compact text prompt in Python first — the LLM gets only this,
        # not the full JSON blob, so it can focus purely on analysis.
        analysis_text = prepare_analysis_prompt(cleaned_json)
        chat_client = _build_chat_client()
        data_analyst = create_data_analyst_agent(chat_client)
        try:
            analyst_response = await asyncio.wait_for(data_analyst.run(analysis_text), timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError("DataAnalystAgent timed out after 300s — Azure OpenAI did not respond")
        analyst_text = analyst_response.text or ""
        logger.info(f"[Orchestrator] DataAnalystAgent responded ({len(analyst_text)} chars)")
        analysis_result = _extract_json(analyst_text)
        # Schema validation — catch LLM omissions before they cause downstream failures
        required_keys = {"username", "analysis_date", "total_posts_analysed",
                         "keynotes_summary", "sentiment_analysis", "recommendations"}
        missing = required_keys - analysis_result.keys()
        if missing:
            raise ValueError(f"DataAnalystAgent response missing required fields: {missing}")
        optional_keys = {"health_beauty_products", "top_hashtags", "engagement_insights"}
        missing_optional = optional_keys - analysis_result.keys()
        if missing_optional:
            logger.warning(
                f"[Orchestrator] DataAnalystAgent omitted optional fields: {missing_optional} — report will be incomplete"
            )
        _update_step(progress, run_record, "data_analyst", "done")

        # ── Save result now (before email) so the UI can display it even if email fails
        result = {
            "status": "success",
            "username": username,
            "raw_data": raw_data,
            "cleaned_data": cleaned_data,
            "analysis_result": analysis_result,
        }
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = RESULTS_DIR / f"{username}_{ts}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        run_record["result_file"] = str(fname.resolve())
        logger.info(f"[Orchestrator] Analysis result saved → {fname}")

        # ── Step 4: Email Agent (generate PDF + send via Mailtrap) ─────────
        if recipients:
            logger.info(
                f"[Orchestrator] Step 4/4 — EmailAgent: sending PDF to {recipients}"
            )
            _update_step(progress, run_record, "email_agent", "running")
            email_agent = create_email_agent(chat_client)
            try:
                email_response = await asyncio.wait_for(
                    email_agent.run(
                        json.dumps({
                            "username": username,
                            "recipient_emails": ",".join(recipients),
                            "analysis_json": json.dumps(analysis_result),
                        })
                    ),
                    timeout=180,
                )
            except asyncio.TimeoutError:
                raise RuntimeError("EmailAgent timed out after 180s")
            email_text = email_response.text or ""
            # Parse the JSON response to confirm delivery — avoids false-positives/negatives
            # from substring matching (e.g. {"status":"sent"} vs {"status": "sent"}).
            _email_confirmed = False
            try:
                _email_status = _extract_json(email_text)
                _email_confirmed = _email_status.get("status") == "sent"
            except (ValueError, json.JSONDecodeError):
                pass
            if not _email_confirmed:
                raise RuntimeError(f"EmailAgent did not confirm email delivery. Response: {email_text[:300]}")
            logger.info(f"[Orchestrator] EmailAgent done — {email_text}")
            _update_step(progress, run_record, "email_agent", "done")

        run_record["status"] = "success"
        elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
        logger.info(f"[Orchestrator] Pipeline succeeded for @{username} in {elapsed:.1f}s")

    except Exception as exc:
        run_record["status"] = "error"
        run_record["error"] = str(exc)
        # Mark any step that was left in 'running' as 'error'
        for s in run_record["steps"]:
            if run_record["steps"][s] == "running":
                run_record["steps"][s] = "error"
        logger.exception("[Orchestrator] Unhandled exception")

    elapsed_total = (datetime.now(timezone.utc) - started_dt).total_seconds()
    run_record["finished_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"[Orchestrator] Run finished for @{username} — "
        f"status={run_record['status']}, duration={elapsed_total:.1f}s"
    )
    _save_progress(progress)
    return run_record


def _parse_recipients(env_var: str = "REPORT_RECIPIENT_EMAILS") -> list[str]:
    """Parse a comma-separated email list from an env var; return empty list if unset/blank."""
    raw = os.environ.get(env_var, "").strip()
    return [e.strip() for e in raw.split(",") if e.strip()] if raw else []


async def run_scheduler():
    """Run the watcher on a recurring schedule."""
    accounts = os.environ.get("IG_TARGET_ACCOUNTS", "hellocatie45").split(",")
    interval_h = float(os.environ.get("WATCH_INTERVAL_HOURS", "24"))
    recipients = _parse_recipients()
    logger.info(f"Scheduler started. Accounts={accounts}, interval={interval_h}h, recipients={recipients}")
    while True:
        for account in accounts:
            await run_pipeline(account.strip(), recipients=recipients if recipients else None)
        logger.info(f"Next run in {interval_h} hour(s) …")
        await asyncio.sleep(interval_h * 3600)


if __name__ == "__main__":
    # CLI usage:
    #   python orchestrator.py               → start recurring scheduler (uses IG_TARGET_ACCOUNTS + WATCH_INTERVAL_HOURS)
    #   python orchestrator.py once          → run once for default account (hellocatie45)
    #   python orchestrator.py once <user>  → run once for specified username
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        username = sys.argv[2] if len(sys.argv) > 2 else "hellocatie45"
        recipients = _parse_recipients()
        asyncio.run(run_pipeline(username, recipients=recipients if recipients else None))
    else:
        asyncio.run(run_scheduler())
