"""Re-send the email report using the latest analysis result — no LLM or scraping needed."""
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

from email_agent import send_analysis_email


def main():
    results_dir = Path("results")

    # Find the latest result JSON (excluding progress.json and test files)
    result_files = sorted(
        (f for f in results_dir.glob("*.json") if f.name != "progress.json" and not f.name.startswith("test")),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not result_files:
        print("ERROR: No result files found in results/")
        sys.exit(1)

    result_file = result_files[0]
    print(f"Using: {result_file}")

    with open(result_file) as f:
        data = json.load(f)

    analysis = data["analysis_result"]
    username = analysis["username"]

    # Resolve recipients: CLI arg > env var
    import os
    if len(sys.argv) > 1:
        recipients = sys.argv[1]
    else:
        recipients = os.environ.get("REPORT_RECIPIENT_EMAILS", "")

    if not recipients.strip():
        print("ERROR: No recipient specified.")
        print("  Usage: python resend_email.py you@example.com")
        print("  Or set REPORT_RECIPIENT_EMAILS in .env")
        sys.exit(1)

    print(f"  Username:   @{username}")
    print(f"  Recipients: {recipients}")
    print(f"  Products:   {len(analysis.get('health_beauty_products', []))}")
    print(f"  Hashtags:   {len(analysis.get('top_hashtags', []))}")
    print(f"  Recs:       {len(analysis.get('recommendations', []))}")
    print()

    result = send_analysis_email(
        analysis_json=json.dumps(analysis),
        recipient_emails=recipients,
        username=username,
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
