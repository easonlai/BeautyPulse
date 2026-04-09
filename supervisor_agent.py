"""
Supervisor Agent — available for direct / ad-hoc end-to-end invocation.

NOTE: The production pipeline in orchestrator.py calls each sub-agent directly
for step-level progress tracking. This supervisor is NOT used in the normal run.
It is available for single-call end-to-end invocation, e.g.:
    agent = create_supervisor_agent(chat_client)
    response = await agent.run("hellocatie45")
"""
import logging
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from ig_watcher_agent import create_ig_watcher_agent
from data_cleaning_agent import create_data_cleaning_agent
from data_analyst_agent import create_data_analyst_agent, prepare_ig_analysis_prompt
from email_agent import create_email_agent

logger = logging.getLogger(__name__)


def create_supervisor_agent(chat_client: OpenAIChatClient) -> Agent:
    """Return a Microsoft Agent Framework SupervisorAgent for ad-hoc end-to-end invocation.

    Wires all four sub-agents as tools and orchestrates the full pipeline via LLM
    reasoning. Use this for one-off calls; the production pipeline in orchestrator.py
    calls each sub-agent directly for step-level progress tracking.

    Pipeline order enforced via instructions:
      ig_watcher → data_cleaner → prepare_ig_analysis_prompt → data_analyst → email_agent (optional)
    """
    logger.info("[Supervisor] Creating supervisor and sub-agents")
    ig_watcher = create_ig_watcher_agent(chat_client)
    data_cleaner = create_data_cleaning_agent(chat_client)
    data_analyst = create_data_analyst_agent(chat_client)
    email_agent = create_email_agent(chat_client)
    logger.info("[Supervisor] Sub-agents created: IGWatcherAgent, DataCleaningAgent, DataAnalystAgent, EmailAgent")

    ig_watcher_tool = ig_watcher.as_tool(
        name="ig_watcher",
        description="Fetch Instagram posts and comments for a given username. Input: the Instagram username string.",
        arg_name="username",
        arg_description="The Instagram username to watch (without @)",
    )
    data_cleaner_tool = data_cleaner.as_tool(
        name="data_cleaner",
        description="Clean and normalise raw Instagram JSON. Input: the raw JSON string from ig_watcher.",
        arg_name="raw_json",
        arg_description="Raw JSON string produced by ig_watcher",
    )
    data_analyst_tool = data_analyst.as_tool(
        name="data_analyst",
        description="Analyse Instagram content and return a structured JSON report. Input: the pre-formatted text prompt returned by prepare_ig_analysis_prompt.",
        arg_name="analysis_text",
        arg_description="Formatted text summary produced by prepare_ig_analysis_prompt",
    )
    email_agent_tool = email_agent.as_tool(
        name="email_agent",
        description=(
            "Generate a PDF analysis report and email it to the specified recipients. "
            "Input: a JSON string with keys 'analysis_json', 'recipient_emails' (comma-separated), and 'username'."
        ),
        arg_name="input_json",
        arg_description="JSON string with analysis_json, recipient_emails, and username keys",
    )
    logger.info("[Supervisor] Tools registered: ig_watcher, data_cleaner, prepare_ig_analysis_prompt, data_analyst, email_agent")

    return Agent(
        name="SupervisorAgent",
        client=chat_client,
        instructions="""You are the Supervisor Agent for the Instagram Analytics System.

For each request (which will be an Instagram username, optionally with recipient emails), execute these steps IN ORDER:

Step 1 – Call ig_watcher with the username.
         Store the result as raw_data.

Step 2 – Call data_cleaner with raw_data from Step 1.
         Store the result as cleaned_data.

Step 3 – Call prepare_ig_analysis_prompt with cleaned_data from Step 2.
         Store the result as analysis_text.
         (This converts the raw JSON into a formatted text summary for the analyst.)

Step 4 – Call data_analyst with analysis_text from Step 3.
         Store the result as analysis_result.

Step 5 – If recipient emails were provided, call email_agent with a JSON string:
         {"analysis_json": "<analysis_result as JSON string>", "recipient_emails": "<comma-separated emails>", "username": "<username>"}

Step 6 – Return a JSON object:
{
  "status": "success",
  "username": "<the username>",
  "raw_data": <raw_data object>,
  "cleaned_data": <cleaned_data object>,
  "analysis_result": <analysis_result object>
}

If any step fails, return:
{"status": "error", "step": "<step name>", "message": "<error description>"}

Always return valid JSON — no prose.""",
        tools=[ig_watcher_tool, data_cleaner_tool, prepare_ig_analysis_prompt, data_analyst_tool, email_agent_tool],
    )
