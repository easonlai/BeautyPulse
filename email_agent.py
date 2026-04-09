"""
Email Agent
Generates a PDF analysis report from Instagram data and sends it
via Mailtrap.io email gateway.

Required env vars:
  MAILTRAP_API_TOKEN     — Mailtrap sending API token
  MAILTRAP_SENDER_EMAIL  — From address (default: noreply@minions.app)
  MAILTRAP_SENDER_NAME   — From name   (default: BeautyPulse Analytics)
"""
import base64
import html as _html
import json
import logging
import os
from datetime import datetime, timezone
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Arial Unicode.ttf is a bundled macOS/Office font with full CJK coverage.
_UNICODE_FONT = "/Library/Fonts/Arial Unicode.ttf"


def _safe(text: str, max_len: int = None) -> str:
    """Truncate text to max_len if specified. Keeps all Unicode characters (incl. CJK)."""
    result = str(text)
    if max_len and len(result) > max_len:
        result = result[: max_len - 1] + "…"
    return result


def _generate_pdf(analysis: dict, username: str) -> bytes:
    """Build a BeautyPulse PDF report from analysis data and return raw bytes."""
    from fpdf import FPDF
    from fpdf.enums import MethodReturnValue

    class _PDF(FPDF):
        def header(self):
            self.set_font("Main", "B", 11)
            self.set_text_color(41, 128, 185)
            self.cell(
                0, 8, "BeautyPulse Instagram Analysis Report",
                align="C", new_x="LMARGIN", new_y="NEXT",
            )
            self.set_font("Main", "", 9)
            self.set_text_color(120)
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            self.cell(
                0, 5, f"@{_safe(username)}  |  Generated: {now}",
                align="C", new_x="LMARGIN", new_y="NEXT",
            )
            self.set_text_color(0)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
            self.ln(6)

        def footer(self):
            self.set_y(-14)
            self.set_font("Main", "I", 8)
            self.set_text_color(150)
            self.cell(
                0, 6,
                f"BeautyPulse Multi-Agent Analytics  |  Page {self.page_no()}",
                align="C",
            )
            self.set_text_color(0)

    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 20, 15)
    pdf.set_auto_page_break(auto=True, margin=18)
    # Register Unicode font (supports CJK / Traditional Chinese) before add_page
    # so header() / footer() can reference it on the very first page.
    pdf.add_font("Main", fname=_UNICODE_FONT)
    pdf.add_font("Main", style="B", fname=_UNICODE_FONT)
    pdf.add_font("Main", style="I", fname=_UNICODE_FONT)
    pdf.add_font("Main", style="BI", fname=_UNICODE_FONT)
    pdf.add_page()

    def section_title(title: str):
        """Render a full-width section header with coloured background."""
        pdf.set_fill_color(235, 245, 255)
        pdf.set_text_color(41, 128, 185)
        pdf.set_font("Main", "B", 12)
        pdf.cell(0, 8, title, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)
        pdf.ln(1)

    def body_text(text: str):
        """Render a left-aligned body paragraph. Resets x to left margin before
        multi_cell to prevent cursor drift after preceding cell() calls."""
        pdf.set_font("Main", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5.5, _safe(text), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # ── Account header ───────────────────────────────────────────────────────
    pdf.set_font("Main", "B", 18)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 12, f"@{_safe(username)}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.set_font("Main", "", 10)
    sentiment = analysis.get("sentiment_analysis", {}) or {}
    _score = float(sentiment.get("score") or 0)
    meta = "  |  ".join([
        f"Analysis Date: {analysis.get('analysis_date', 'N/A')}",
        f"Posts Analysed: {analysis.get('total_posts_analysed', 'N/A')}",
        f"Sentiment: {sentiment.get('overall', 'N/A')} ({_score:.2f})",
    ])
    pdf.cell(0, 6, _safe(meta), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Keynotes Summary ─────────────────────────────────────────────────────
    section_title("Keynotes Summary")
    body_text(analysis.get("keynotes_summary") or "Not available.")

    # ── Sentiment Analysis ───────────────────────────────────────────────────
    section_title("Sentiment Analysis")
    breakdown = sentiment.get("breakdown", {}) or {}
    body_text(
        f"Overall: {sentiment.get('overall', 'N/A')}  |  Score: {_score:.2f}\n"
        f"Positive: {breakdown.get('positive_pct', 0):.1f}%  |  "
        f"Neutral: {breakdown.get('neutral_pct', 0):.1f}%  |  "
        f"Negative: {breakdown.get('negative_pct', 0):.1f}%"
    )
    notable = sentiment.get("notable_comments", [])
    if notable:
        pdf.set_font("Main", "B", 9)
        pdf.cell(0, 5, "Notable Comments:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Main", "I", 9)
        pdf.set_text_color(80)
        for nc in notable[:5]:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, _safe(f"  \u2022 {str(nc)[:200]}"), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)
        pdf.ln(2)

    # ── Health & Beauty Products ─────────────────────────────────────────────
    products = analysis.get("health_beauty_products", [])
    if products:
        section_title("Health & Beauty Products Mentioned")
        # Column widths (mm): Product 50, Brand 35, Mentions 17, Context 78 = 180mm usable.
        # Context gets the most space since it holds the longest text.
        col_w = [50, 35, 17, 78]
        col_align = ["L", "L", "C", "L"]  # Mentions centred; rest left-aligned
        headers = ["Product", "Brand", "Mentions", "Context"]
        line_h = 5  # mm per text line inside body cells

        # Header row (single line, filled background)
        pdf.set_fill_color(210, 230, 255)
        pdf.set_font("Main", "B", 8)
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 6, h, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Main", "", 8)
        for p in products[:15]:
            row = [
                _safe(str(p.get("product_name", "")), 60),
                _safe(str(p.get("brand") or "N/A"), 40),
                str(p.get("mention_count", 0)),
                _safe(str(p.get("context", "")), 100),
            ]
            # Dry-run each cell to count wrapped lines; all cells in this row
            # must share the same height so borders stay aligned across columns.
            max_lines = 1
            for txt, w in zip(row, col_w):
                rendered = pdf.multi_cell(
                    w, line_h, txt, wrapmode="CHAR",
                    dry_run=True, output=MethodReturnValue.LINES,
                )
                max_lines = max(max_lines, len(rendered) if rendered else 1)
            row_h = max_lines * line_h

            # Check if this row fits on the current page; if not, add a page
            # and re-draw the table header so the reader can follow the columns.
            if pdf.get_y() + row_h > pdf.h - pdf.b_margin:
                pdf.add_page()
                pdf.set_fill_color(210, 230, 255)
                pdf.set_font("Main", "B", 8)
                for i, h in enumerate(headers):
                    pdf.cell(col_w[i], 6, h, border=1, fill=True)
                pdf.ln()
                pdf.set_font("Main", "", 8)

            y0 = pdf.get_y()
            for i, (txt, w) in enumerate(zip(row, col_w)):
                pdf.set_xy(pdf.l_margin + sum(col_w[:i]), y0)
                pdf.multi_cell(
                    w, line_h, txt, border=1,
                    align=col_align[i], wrapmode="CHAR",
                    new_x="RIGHT", new_y="TOP",
                )
            # Advance past the row — new_y="TOP" keeps the cursor at y0,
            # so we move it down manually by the tallest cell's height.
            pdf.set_y(y0 + row_h)
        pdf.ln(3)

    # ── Top Hashtags ─────────────────────────────────────────────────────────
    top_tags = analysis.get("top_hashtags", [])
    if top_tags:
        section_title("Top Hashtags")
        tags_str = "  ".join(
            f"#{t.get('tag', '').lstrip('#')} ({t.get('count', 0)})" for t in top_tags[:15]
        )
        body_text(tags_str)

    # ── Engagement Insights ──────────────────────────────────────────────────
    section_title("Engagement Insights")
    body_text(analysis.get("engagement_insights") or "Not available.")

    # ── Recommendations ──────────────────────────────────────────────────────
    recs = analysis.get("recommendations", [])
    if recs:
        section_title("Recommendations")
        pdf.set_font("Main", "", 10)
        for rec in recs:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, _safe(f"  \u2022 {rec}"), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    return bytes(pdf.output())


# ── Tool ─────────────────────────────────────────────────────────────────────

@tool
def send_analysis_email(
    analysis_json: Annotated[str, Field(description="Analysis result JSON string from DataAnalystAgent")],
    recipient_emails: Annotated[str, Field(description="Comma-separated list of recipient email addresses")],
    username: Annotated[str, Field(description="Instagram username that was analysed")],
) -> str:
    """
    Generate a PDF report from Instagram analysis results and email it to all recipients via Mailtrap.io.
    Returns a JSON status string: {"status": "sent", "recipients": [...], "attachment": "..."}.
    """
    # Defer mailtrap import to avoid ImportError at module load if the package
    # is not installed in environments that only use the PDF generation path.
    import mailtrap as mt

    mailtrap_token = os.environ.get("MAILTRAP_API_TOKEN")
    sender_email = os.environ.get("MAILTRAP_SENDER_EMAIL", "noreply@minions.app")
    sender_name = os.environ.get("MAILTRAP_SENDER_NAME", "BeautyPulse Analytics")

    if not mailtrap_token:
        raise EnvironmentError("MAILTRAP_API_TOKEN environment variable is not set")

    # Accept either a comma-separated string or a pre-split list
    if isinstance(recipient_emails, str):
        recipients = [e.strip() for e in recipient_emails.split(",") if e.strip()]
    else:
        recipients = [e.strip() for e in recipient_emails if e.strip()]
    if not recipients:
        raise ValueError("No valid recipient email addresses provided")

    analysis = json.loads(analysis_json) if isinstance(analysis_json, str) else analysis_json

    logger.info(f"[EmailAgent] Generating PDF report for @{username}")
    pdf_bytes = _generate_pdf(analysis, username)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    attachment_name = f"BeautyPulse_{username}_{date_str}.pdf"
    logger.info(
        f"[EmailAgent] PDF ready ({len(pdf_bytes):,} bytes) — sending to {recipients}"
    )

    sentiment = analysis.get("sentiment_analysis", {}) or {}
    _score = float(sentiment.get("score") or 0)
    _e = _html.escape  # shorthand for HTML-escaping dynamic values
    html_body = f"""<html>
<body style="font-family:Arial,sans-serif;color:#222;max-width:600px;margin:0 auto;">
  <h2 style="color:#2980b9;">BeautyPulse Instagram Analytics Report</h2>
  <p>Hi,</p>
  <p>Your Instagram analytics report for <strong>@{_e(username)}</strong> is ready.
  Please find the full PDF attached.</p>
  <table style="border-collapse:collapse;width:100%;margin:16px 0;font-size:14px;">
    <tr style="background:#eaf4ff;">
      <td style="padding:8px 12px;font-weight:bold;border:1px solid #d0e8ff;">Posts Analysed</td>
      <td style="padding:8px 12px;border:1px solid #d0e8ff;">{_e(str(analysis.get('total_posts_analysed', 'N/A')))}</td>
    </tr>
    <tr>
      <td style="padding:8px 12px;font-weight:bold;border:1px solid #d0e8ff;">Overall Sentiment</td>
      <td style="padding:8px 12px;border:1px solid #d0e8ff;">
        {_e(str(sentiment.get('overall', 'N/A')))} (score: {_score:.2f})
      </td>
    </tr>
    <tr style="background:#eaf4ff;">
      <td style="padding:8px 12px;font-weight:bold;border:1px solid #d0e8ff;">Analysis Date</td>
      <td style="padding:8px 12px;border:1px solid #d0e8ff;">{_e(str(analysis.get('analysis_date', 'N/A')))}</td>
    </tr>
  </table>
  <hr style="border:none;border-top:1px solid #ddd;margin:20px 0;"/>
  <p style="color:#888;font-size:11px;">
    Generated by BeautyPulse Multi-Agent Analytics System
  </p>
</body>
</html>"""

    mail = mt.Mail(
        sender=mt.Address(email=sender_email, name=sender_name),
        to=[mt.Address(email=addr) for addr in recipients],
        subject=f"BeautyPulse Analysis Report \u2014 @{username}",
        html=html_body,
        attachments=[
            mt.Attachment(
                content=base64.b64encode(pdf_bytes),
                filename=attachment_name,
                disposition=mt.Disposition.ATTACHMENT,
                mimetype="application/pdf",
            )
        ],
    )

    client = mt.MailtrapClient(token=mailtrap_token)
    import concurrent.futures as _cf
    # Mailtrap's client.send() is a blocking synchronous call with no built-in
    # timeout.  Wrapping it in a single-worker ThreadPoolExecutor lets us enforce
    # a wall-clock deadline (future.result(timeout=30)) without blocking the
    # calling async event loop indefinitely on a slow or hung SMTP connection.
    with _cf.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.send, mail)
        try:
            response = future.result(timeout=30)
        except (TimeoutError, _cf.TimeoutError):
            raise RuntimeError("Mailtrap send timed out after 30s")
    logger.info(f"[EmailAgent] Email sent successfully — response: {response}")

    return json.dumps({
        "status": "sent",
        "recipients": recipients,
        "attachment": attachment_name,
    })


# ── Agent factory ─────────────────────────────────────────────────────────────

def create_email_agent(chat_client: OpenAIChatClient) -> Agent:
    """Return a Microsoft Agent Framework Agent that generates and emails PDF reports.

    The agent receives a JSON string with keys {username, recipient_emails, analysis_json}
    and calls send_analysis_email. The LLM acts as a dispatcher — PDF generation and
    email delivery are handled entirely in Python by the tool function.
    """
    return Agent(
        name="EmailAgent",
        client=chat_client,
        instructions="""You are the Email Agent for BeautyPulse Analytics.

You will receive a JSON message with exactly these keys:
  - username: the Instagram username
  - recipient_emails: comma-separated email addresses
  - analysis_json: the full analysis result as a JSON string

Call send_analysis_email with those three exact values.
Return only the JSON status object returned by that tool — no extra text, no markdown.""",
        tools=[send_analysis_email],
    )
