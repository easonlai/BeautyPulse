"""
Streamlit UI  –  BeautyPulse Multi-Agent Analytics Dashboard
Run:  streamlit run streamlit_app.py
"""
import os
import json
import html as _html
import logging
import asyncio
import threading
import tempfile
import nest_asyncio
from pathlib import Path
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# Configure logging for the UI module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# Streamlit runs in a sync context but orchestrator.run_pipeline is async.
# Strategy:
#   1. Force DefaultEventLoopPolicy so nest_asyncio doesn't conflict with uvloop
#      (uvloop's event loop is not patchable by nest_asyncio).
#   2. Create a single event loop, patch it with nest_asyncio so Streamlit's
#      own internal async calls and our coroutines can coexist.
#   3. Background pipeline runs get their OWN fresh loop (see run_pipeline_sync_multi).
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)
nest_asyncio.apply(_loop)
load_dotenv(override=False)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = RESULTS_DIR / "progress.json"

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BeautyPulse Multi-Agent Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_progress() -> dict:
    """Load the pipeline run history from progress.json.
    Returns an empty state dict if the file is missing or corrupt.
    """
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[UI] progress.json is corrupt or unreadable ({exc}); resetting")
            return {"runs": []}
    return {"runs": []}


def load_result(filepath: str) -> dict | None:
    """Load a single analysis result JSON file. Returns None if the file is
    missing, partially written (JSONDecodeError), or otherwise unreadable.
    Falls back to RESULTS_DIR/<filename> for old runs that stored a relative path.
    """
    p = Path(filepath)
    # If the path doesn't exist as-is (e.g. old relative path on a different cwd),
    # try resolving it inside RESULTS_DIR by filename only.
    if not p.exists():
        p = RESULTS_DIR / p.name
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[UI] Could not load result file '{p.name}': {exc}")
    return None


def list_result_files() -> list[Path]:
    """Return all result JSON files in RESULTS_DIR, newest first (by modification time).
    Excludes progress.json which lives in the same directory.
    """
    return sorted(
        (f for f in RESULTS_DIR.glob("*.json") if f.name != "progress.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )


def _cleanup_stuck_runs():
    """On startup, mark any runs left in 'running' state as 'error' (process was interrupted)."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                progress = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[UI] Could not read progress.json during startup cleanup: {exc}")
            return
        changed = False
        for run in progress.get("runs", []):
            if run.get("status") == "running":
                logger.warning(
                    f"[UI] Marking stuck run as interrupted: "
                    f"@{run.get('username','?')} started {run.get('started_at','')}"
                )
                run["status"] = "error"
                run["error"] = "interrupted (process restarted)"
                # Also mark any step left in 'running' as 'error'
                for step_key, step_val in run.get("steps", {}).items():
                    if step_val == "running":
                        run["steps"][step_key] = "error"
                changed = True
        if changed:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=PROGRESS_FILE.parent, prefix=".progress_", suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(progress, f, indent=2)
                os.replace(tmp_path, PROGRESS_FILE)
            except Exception as exc:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                logger.warning(f"[UI] Startup cleanup: could not write progress.json: {exc}")
                return
            logger.info("[UI] Startup cleanup complete — stuck runs resolved")
        else:
            logger.info("[UI] Startup cleanup: no stuck runs found")


# Run once per Streamlit server session, not on every re-render
if "startup_cleanup_done" not in st.session_state:
    _cleanup_stuck_runs()
    st.session_state["startup_cleanup_done"] = True

# Load progress once — used by sidebar guard, progress panel, and auto-refresh
progress_data = load_progress()
runs = progress_data.get("runs", [])

# Non-blocking auto-refresh while a pipeline is running
if any(r.get("status") == "running" for r in runs):
    st_autorefresh(interval=5000, key="pipeline_autorefresh")


def run_pipeline_sync_multi(usernames: list[str], recipients: list[str] = None):
    """Run the pipeline sequentially for multiple usernames in one background thread.

    Each call to run_pipeline is an async coroutine. We must NOT reuse the
    nest_asyncio-patched main-thread loop here — instead we create a fresh event
    loop for this background thread and close it when done.
    """
    from orchestrator import run_pipeline
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        for username in usernames:
            try:
                loop.run_until_complete(
                    run_pipeline(username, recipients=recipients if recipients else None)
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    f"[UI] Unhandled exception in background pipeline thread for @{username}"
                )
    finally:
        loop.close()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 BeautyPulse Multi-Agent System Control")
    st.markdown("---")

    default_accounts = os.environ.get("IG_TARGET_ACCOUNTS", "hellocatie45")
    username_input = st.text_input(
        "Instagram Username(s)",
        value=default_accounts,
        placeholder="e.g. user1, user2, user3",
        help="Enter one or more Instagram usernames separated by commas.",
    )

    if st.button("▶ Run Pipeline Now", type="primary", width="stretch"):
        usernames = [u.strip().lstrip("@") for u in username_input.split(",") if u.strip()]
        recipients_val = [e.strip() for e in os.environ.get("REPORT_RECIPIENT_EMAILS", "").split(",") if e.strip()]
        # Re-read progress from disk at click time — the in-memory snapshot may be stale
        _live_runs = load_progress().get("runs", [])
        if not usernames:
            st.warning("Please enter at least one Instagram username.")
        elif any(r.get("status") == "running" for r in _live_runs):
            st.warning("A pipeline is already running. Please wait for it to complete.")
        else:
            st.session_state["pipeline_usernames"] = usernames
            logger.info(f"[UI] Pipeline triggered for: {', '.join('@' + u for u in usernames)}")
            t = threading.Thread(
                target=run_pipeline_sync_multi,
                args=(usernames, recipients_val if recipients_val else None),
                daemon=True,
            )
            t.start()
            if len(usernames) == 1:
                st.success(f"Pipeline started for @{usernames[0]}")
            else:
                st.success(f"Pipeline started for {len(usernames)} accounts: {', '.join('@' + u for u in usernames)}")

    st.markdown("---")
    st.caption("Auto-refresh every 5s while pipeline runs")
    if st.button("🔄 Refresh", width="stretch"):
        st.rerun()

    st.markdown("---")
    result_files = list_result_files()
    result_names = [f.name for f in result_files]
    selected_file = st.selectbox("📂 Load Result File", result_names) if result_names else None


# ── Main Layout ───────────────────────────────────────────────────────────────
st.title("📊 BeautyPulse Multi-Agent Analytics Dashboard")
st.caption("Powered by Microsoft Agent Framework · GPT-5.2 · Microsoft Foundry")

# ── Agent Progress Panel ──────────────────────────────────────────────────────
st.subheader("🔄 Agent Execution Progress")
if not runs:
    st.info("No pipeline runs yet. Trigger one from the sidebar.")
else:
    latest_runs = runs[-10:][::-1]
    for run in latest_runs:
        is_running = run.get("status") == "running"
        status_icon = {"success": "✅", "error": "❌", "running": "🔄"}.get(run.get("status", ""), "❓")

        # Compute elapsed time for runs still in progress
        time_label = ""
        if is_running and run.get("started_at"):
            try:
                started_dt = datetime.fromisoformat(run["started_at"])
                elapsed_s = (datetime.now(timezone.utc) - started_dt).total_seconds()
                time_label = f"  ⏱ {elapsed_s:.0f}s elapsed"
            except Exception:
                pass
        elif run.get("finished_at") and run.get("started_at"):
            try:
                dur = (datetime.fromisoformat(run["finished_at"]) - datetime.fromisoformat(run["started_at"])).total_seconds()
                time_label = f"  ⏱ {dur:.0f}s"
            except Exception:
                pass

        with st.expander(
            f"{status_icon} @{run.get('username','?')}  —  "
            f"{run.get('started_at','')[:19].replace('T',' ')}{time_label}",
            expanded=is_running,
        ):
            steps = run.get("steps", {})
            has_email_step = "email_agent" in steps
            _icons = {"done": "✅", "pending": "⌛", "error": "❌", "running": "🔄"}
            if has_email_step:
                col1, col2, col3, col4, col5 = st.columns(5)
            else:
                col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall", run.get("status", "—").upper())
            col2.metric("📷 IG Watcher", _icons.get(steps.get("ig_watcher", "pending"), "❓"))
            col3.metric("🧹 Data Cleaner", _icons.get(steps.get("data_cleaner", "pending"), "❓"))
            col4.metric("📈 Data Analyst", _icons.get(steps.get("data_analyst", "pending"), "❓"))
            if has_email_step:
                col5.metric("📧 Email Report", _icons.get(steps.get("email_agent", "pending"), "❓"))
            if is_running:
                # Show which step is actively running
                active = next((k for k, v in steps.items() if v == "running"), None)
                if active:
                    labels = {
                        "ig_watcher": "📷 IG Watcher",
                        "data_cleaner": "🧹 Data Cleaner",
                        "data_analyst": "📈 Data Analyst",
                        "email_agent": "📧 Email Agent",
                    }
                    st.info(f"Running: **{labels.get(active, active)}** …")
            if run.get("error"):
                st.error(f"Error: {run['error']}")

# ── Results Panel ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Analysis Results")

result = None
if selected_file:
    path = RESULTS_DIR / selected_file
    result = load_result(str(path))
    if result is None:
        st.warning(f"Result file '{selected_file}' not found on disk. Showing last successful run instead.")
if result is None and runs:
    # Show results from any run that has a result file — including runs where only email failed
    last_with_result = next((r for r in reversed(runs) if r.get("result_file")), None)
    if last_with_result:
        result = load_result(last_with_result["result_file"])

if result is None:
    st.warning("No successful results to display yet.")
    st.stop()

analysis = result.get("analysis_result", {})
cleaned = result.get("cleaned_data", {})
posts = cleaned.get("posts", [])

# KPI Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("👤 Account", f"@{analysis.get('username', '?')}")
col2.metric("📝 Posts Analysed", analysis.get("total_posts_analysed", len(posts)))
sentiment = analysis.get("sentiment_analysis", {}) or {}
_sentiment_score = float(sentiment.get("score") or 0)
col3.metric("💬 Sentiment", sentiment.get("overall", "—"))
col4.metric("🎯 Sentiment Score", f"{_sentiment_score:.2f}")

st.divider()

# Keynotes
st.markdown("### 🗝 Keynotes Summary")
st.info(analysis.get("keynotes_summary", "No summary available."))

# Two column layout
left, right = st.columns(2)

# ── Health & Beauty Products ──────────────────────────────────────────────────
with left:
    st.markdown("### 💄 Health & Beauty Products Mentioned")
    products = analysis.get("health_beauty_products", [])
    if products:
        df_prod = pd.DataFrame(products)
        st.dataframe(df_prod, width="stretch", hide_index=True)
        if {"product_name", "mention_count"}.issubset(df_prod.columns):
            fig_prod = px.bar(
                df_prod.sort_values("mention_count", ascending=False).head(10),
                x="product_name", y="mention_count",
                color="brand" if "brand" in df_prod.columns else None,
                title="Product Mention Frequency",
                labels={"mention_count": "Mentions", "product_name": "Product"},
            )
            fig_prod.update_layout(xaxis_tickangle=-30, height=350)
            st.plotly_chart(fig_prod, width="stretch")
    else:
        st.write("No products identified.")

# ── Sentiment Breakdown ───────────────────────────────────────────────────────
with right:
    st.markdown("### 😊 Comment Sentiment Breakdown")
    breakdown = sentiment.get("breakdown", {}) or {}
    if breakdown:
        labels = ["Positive", "Neutral", "Negative"]
        values = [
            breakdown.get("positive_pct", 0),
            breakdown.get("neutral_pct", 0),
            breakdown.get("negative_pct", 0),
        ]
        fig_sent = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker_colors=["#2ecc71", "#95a5a6", "#e74c3c"],
        ))
        fig_sent.update_layout(title="Sentiment Distribution", height=350)
        st.plotly_chart(fig_sent, width="stretch")

    notable = sentiment.get("notable_comments", [])
    if notable:
        st.markdown("**Notable Comments:**")
        for nc in notable[:5]:
            st.markdown(
                f"<blockquote>{_html.escape(str(nc))}</blockquote>",
                unsafe_allow_html=True,
            )

# ── Top Hashtags ──────────────────────────────────────────────────────────────
st.markdown("### #️⃣ Top Hashtags")
top_tags = analysis.get("top_hashtags", [])
if top_tags:
    df_tags = pd.DataFrame(top_tags)
    if {"tag", "count"}.issubset(df_tags.columns):
        # px.treemap requires strictly positive values — filter out zero/negative counts
        df_tags = df_tags[df_tags["count"] > 0]
    if not df_tags.empty and {"tag", "count"}.issubset(df_tags.columns):
        fig_tags = px.treemap(df_tags, path=["tag"], values="count", title="Hashtag Frequency")
        st.plotly_chart(fig_tags, width="stretch")
    else:
        st.dataframe(df_tags, width="stretch", hide_index=True)

# ── Engagement Trends (Likes per post) ───────────────────────────────────────
if posts:
    st.markdown("### ❤️ Engagement Trends")
    df_posts = pd.DataFrame([{
        "date": p.get("timestamp", "")[:10],
        "likes": p.get("likesCount", 0),
        "comments": p.get("commentsCount", 0),
        "caption_preview": (p.get("caption") or "")[:60] + ("…" if len(p.get("caption") or "") > 60 else ""),
    } for p in posts])
    df_posts["date"] = pd.to_datetime(df_posts["date"], errors="coerce")
    # Drop rows whose timestamp was missing or malformed (coerce → NaT).
    df_posts = df_posts.dropna(subset=["date"]).sort_values("date")
    fig_eng = go.Figure()
    fig_eng.add_trace(go.Scatter(x=df_posts["date"], y=df_posts["likes"], name="Likes", line=dict(color="#3498db")))
    # Comments use a secondary y-axis (yaxis2) because their scale is typically much
    # smaller than likes — overlaying on a shared axis would flatten the comments line.
    fig_eng.add_trace(go.Scatter(x=df_posts["date"], y=df_posts["comments"], name="Comments", yaxis="y2", line=dict(color="#e67e22")))
    fig_eng.update_layout(
        title="Likes & Comments Over Time",
        yaxis=dict(title="Likes"),
        yaxis2=dict(title="Comments", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig_eng, width="stretch")

# ── Engagement Insights & Recommendations ─────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("### 💡 Engagement Insights")
    st.write(analysis.get("engagement_insights", "—"))
with col_b:
    st.markdown("### ✅ Recommendations")
    for rec in analysis.get("recommendations", []):
        st.markdown(f"- {rec}")

# ── Raw JSON Expander ─────────────────────────────────────────────────────────
with st.expander("🔍 View Raw Analysis JSON"):
    st.json(analysis)
with st.expander("🔍 View Cleaned Data JSON"):
    st.json(cleaned)
