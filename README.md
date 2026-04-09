# BeautyPulse — Multi-Agent Instagram Analytics

BeautyPulse is an end-to-end Instagram social listening and analytics system built on **Microsoft Agent Framework 1.0**. It scrapes public Instagram posts and comments, cleans and normalises the data, runs LLM-powered analysis via **Azure OpenAI** (part of [Microsoft Foundry](https://ai.azure.com/)), and delivers structured insight reports via email — all orchestrated by a pipeline of specialised AI agents.

While it defaults to **beauty** content (product mentions, skincare routines, brand sentiment), the architecture is **industry-agnostic**. Swap the analyst prompt, and the same pipeline powers social listening for fashion, F&B, travel, fitness, consumer electronics, or any vertical where Instagram signals matter.

## Use Cases

### Beauty & Retail

- **Product mention tracking** — automatically surfaces which products, brands, and shade/colour names appear in creator posts and follower comments.
- **Sentiment scoring** — quantifies audience reaction (positive / neutral / negative breakdown) so brand teams can spot issues early.
- **Competitive benchmarking** — tracks how often competitors' products are mentioned alongside your own.
- **Promotion effectiveness** — captures engagement spikes around launch events, discount codes, and gifting campaigns.
- **Actionable recommendations** — LLM-generated, data-grounded suggestions for content strategy, posting cadence, and audience engagement.

### Other Industries

BeautyPulse is a general-purpose **Instagram social listening engine**. Customise the Data Analyst agent prompt to extract domain-specific entities and insights for:

| Industry | What to Extract |
|---|---|
| Fashion | Outfit tagging, trend detection, brand co-mentions |
| F&B / Restaurants | Menu item buzz, location sentiment, influencer reach |
| Travel & Hospitality | Destination sentiment, experience highlights, pain points |
| Fitness & Wellness | Supplement/equipment mentions, routine trends |
| Consumer Electronics | Feature requests, defect complaints, competitor comparisons |

## Architecture

BeautyPulse implements a **sequential multi-agent pipeline** using Microsoft Agent Framework 1.0. Each agent is a specialist with a single responsibility. A headless orchestrator wires them together and provides step-level progress tracking.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (UI)                      │
│         Real-time progress · Charts · KPI cards · PDF           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ triggers / monitors
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator (orchestrator.py)               │
│     Sequential pipeline with step-level progress persistence    │
└──┬──────────┬──────────┬──────────┬─────────────────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│  IG  │  │ Data │  │ Data │  │Email │
│Watcher│→│Cleaner│→│Analyst│→│Agent │
│Agent │  │Agent │  │Agent │  │      │
└──────┘  └──────┘  └──────┘  └──────┘
  Apify     Python    Azure     Mailtrap
 (scrape)   (clean)   OpenAI    (PDF+send)
                      (LLM)
```

### Agent Breakdown

| Agent | Role | LLM? | Key Tech |
|---|---|---|---|
| **IG Watcher** | Scrapes Instagram posts + comments via Apify actors | No | `apify-client`, ThreadPoolExecutor |
| **Data Cleaner** | Deduplicates, normalises timestamps, strips HTML, cleans text | No | Pure Python, regex |
| **Data Analyst** | Analyses content and produces a structured JSON report (sentiment, products, hashtags, recommendations) | Yes | Azure OpenAI via Microsoft Foundry |
| **Email Agent** | Generates a branded PDF report and sends it via Mailtrap | Tool-dispatch | `fpdf2`, `mailtrap` |

### Why This Design

- **LLM only where it adds value** — Steps 1 and 2 are pure Python (no LLM cost or latency). The LLM is reserved for Step 3, where reasoning over unstructured text is actually needed. Step 4 uses the LLM only as a tool dispatcher.
- **Step-level progress tracking** — The orchestrator persists state to `progress.json` after every step, enabling the Streamlit UI to show real-time status with auto-refresh.
- **Decoupled agents** — Each agent can be tested, replaced, or extended independently. Swap Apify for a different scraper, swap Mailtrap for SendGrid, or add new agents to the pipeline.

### Data Flow

Agents communicate through JSON strings that form a data contract between pipeline steps:

```
IG Watcher ──► Data Cleaner ──► Data Analyst ──► Email Agent
   │                │                │                │
   ▼                ▼                ▼                ▼
{                {                {                {"status":"sent",
  "username",      "username",      "username",      "recipients":[...],
  "fetched_at",    "fetched_at",    "analysis_date",  "attachment":"..."}
  "posts": [{      "cleaned_at",    "keynotes_summary",
    "id",          "posts": [{      "sentiment_analysis",
    "caption",       (cleaned)      "health_beauty_products",
    "likesCount",  }]               "top_hashtags",
    "comments"   }                  "engagement_insights",
  }]                                "recommendations"
}                                 }
```

The Data Analyst is the only agent that receives a **pre-processed text prompt** (not raw JSON). The `prepare_analysis_prompt()` function converts cleaned JSON into a compact, human-readable summary — keeping the LLM focused on reasoning rather than parsing.

### Two Orchestration Patterns

The project includes two orchestration modes for the same agent pipeline, showing how Agent Framework 1.0 supports both **code-driven** and **LLM-driven** execution:

**1. Headless Orchestrator** (`orchestrator.py`) — production default

```python
# Code-driven: call each agent directly, track progress per step
raw_json = fetch_ig_data(username)          # Step 1: pure Python
cleaned_json = clean_ig_data(raw_json)      # Step 2: pure Python
analysis_text = prepare_analysis_prompt(cleaned_json)
analyst = create_data_analyst_agent(client)
result = await analyst.run(analysis_text)   # Step 3: LLM
```

Deterministic control, step-level error handling, and progress persistence. Best for scheduled / production workloads.

**2. Supervisor Agent** (`supervisor_agent.py`) — ad-hoc invocation

```python
# LLM-driven: wire all agents as tools, let the LLM decide execution order
ig_watcher_tool = ig_watcher_agent.as_tool(name="ig_watcher", ...)
data_cleaner_tool = data_cleaner_agent.as_tool(name="data_cleaner", ...)

supervisor = Agent(
    client=client,
    instructions="Execute these steps IN ORDER: ig_watcher → data_cleaner → ...",
    tools=[ig_watcher_tool, data_cleaner_tool, data_analyst_tool, email_tool],
)
result = await supervisor.run("hellocatie45")
```

Single-call invocation where the LLM reasons about which tool to call next. Simpler code, but less predictable — useful for interactive / exploratory use.

## Microsoft Agent Framework 1.0 Showcase

This project demonstrates key capabilities of [Microsoft Agent Framework 1.0](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/):

### Core API Usage

| API | Where Used | Purpose |
|---|---|---|
| `Agent(client, instructions, tools)` | All 4 agents | Core agent abstraction — each agent has a role, tools, and an LLM client |
| `@tool` decorator | `fetch_ig_data`, `clean_ig_data`, `send_analysis_email` | Wrap Python functions as agent-callable tools with Pydantic-typed parameters |
| `agent.as_tool()` | `supervisor_agent.py` | Compose an agent as a tool for another agent (agent-as-tool pattern) |
| `agent.run()` | `orchestrator.py` | Single-call invocation returning structured response with `.text` |
| `OpenAIChatClient` | `orchestrator.py` | Unified client — auto-routes to Azure OpenAI when `azure_endpoint` is passed |

### Design Patterns Demonstrated

- **Tool-only agents** — IG Watcher, Data Cleaner, and Email Agent wrap deterministic Python functions as `@tool`s. The LLM acts purely as a dispatcher — it decides *when* to call the tool but doesn't modify the data. This keeps behaviour predictable while still fitting the Agent Framework model.
- **Prompt-only agents** — The Data Analyst agent has *no tools at all*. It receives a pre-formatted text prompt and returns structured JSON purely through LLM reasoning. This shows that not every agent needs tools.
- **Agent-as-tool composition** — The Supervisor wires each sub-agent as a tool via `.as_tool()`, letting the LLM orchestrate the full pipeline through tool calls. This is a core multi-agent pattern in Agent Framework 1.0.
- **Dual authentication** — The `OpenAIChatClient` supports both API key (for local development) and `DefaultAzureCredential` (for production with managed identities), configured via environment variables with no code changes.
- **Hybrid orchestration** — The same set of agents supports both code-driven (`orchestrator.py`) and LLM-driven (`supervisor_agent.py`) orchestration, showing that Agent Framework 1.0 doesn't lock you into a single execution model.

The codebase serves as a practical, production-oriented reference for building multi-agent systems with Agent Framework 1.0 — going beyond basic chatbot patterns.

## Quick Start

### Prerequisites

- Python 3.10+
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) resource with a deployed model (part of [Microsoft Foundry](https://ai.azure.com/))
- [Apify](https://apify.com/) account — uses the [Instagram Post Scraper](https://apify.com/apify/instagram-post-scraper) and [Instagram Comment Scraper](https://apify.com/apify/instagram-comment-scraper) actors
- [Mailtrap](https://mailtrap.io/) account (for email delivery, optional)

### Setup

```bash
# Clone the repo
git clone https://github.com/yourname/BeautyPulse.git
cd BeautyPulse

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment variables
cp .env.template .env
# Edit .env with your API keys and endpoints
```

### Run

```bash
# Launch the Streamlit dashboard
streamlit run streamlit_app.py

# Or run the pipeline from CLI (single run)
python orchestrator.py once hellocatie45

# Or start the recurring scheduler
python orchestrator.py
```

### Test Utilities

```bash
# Verify Azure OpenAI connectivity
python test_connection.py

# Test PDF generation + email delivery (no LLM needed)
python test_email.py you@example.com

# Re-send email report from latest result (no re-scraping or LLM)
python resend_email.py you@example.com
```

## Configuration

All configuration lives in environment variables (`.env` file). See [`.env.template`](.env.template) for the full list.

| Variable | Required | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | Yes* | API key (*or use `DefaultAzureCredential`) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | No | Model deployment name (default: `gpt-5`) |
| `APIFY_API_TOKEN` | Yes | Apify API token for Instagram scraping |
| `IG_TARGET_ACCOUNTS` | No | Comma-separated Instagram usernames (default: `hellocatie45`) |
| `IG_LOOKBACK_DAYS` | No | Days of post history to fetch (default: `7`) |
| `MAILTRAP_API_TOKEN` | No | Mailtrap token (required for email delivery) |
| `REPORT_RECIPIENT_EMAILS` | No | Comma-separated email recipients |

## Project Structure

```
BeautyPulse/
├── orchestrator.py           # Headless pipeline orchestrator (production entry point)
├── supervisor_agent.py       # SupervisorAgent for ad-hoc single-call invocation
├── ig_watcher_agent.py       # Agent 1: Instagram scraping via Apify
├── data_cleaning_agent.py    # Agent 2: Data deduplication and normalisation
├── data_analyst_agent.py     # Agent 3: LLM-powered content analysis
├── email_agent.py            # Agent 4: PDF report generation + email delivery
├── streamlit_app.py          # Streamlit dashboard UI
├── test_connection.py        # Azure OpenAI connectivity test
├── test_email.py             # Standalone PDF + email test
├── resend_email.py           # Re-send email from existing result
├── requirements.txt          # Python dependencies
├── .env.template             # Environment variable template
└── results/                  # Pipeline output (JSON results, progress state)
```

## License

MIT
