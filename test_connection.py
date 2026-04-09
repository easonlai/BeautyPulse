"""Quick smoke test — verifies Azure OpenAI connectivity with Agent Framework 1.0."""
import os
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


def build_client() -> OpenAIChatClient:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    kwargs = dict(azure_endpoint=endpoint, model=deployment)
    if api_version:
        kwargs["api_version"] = api_version

    print(f"  endpoint    = {endpoint}")
    print(f"  model       = {deployment}")
    if api_version:
        print(f"  api_version = {api_version} (explicit)")
    else:
        print(f"  api_version = (SDK default)")

    if api_key:
        return OpenAIChatClient(**kwargs, api_key=api_key)
    from azure.identity import DefaultAzureCredential
    return OpenAIChatClient(**kwargs, credential=DefaultAzureCredential())


async def main():
    print("\n=== Agent Framework 1.0 Connection Test ===\n")
    client = build_client()
    print(f"  SDK resolved api_version = {client.api_version!r}")
    print()

    agent = Agent(
        client=client,
        instructions="Reply with exactly: HELLO FROM AGENT FRAMEWORK 1.0",
    )

    print("  Sending test prompt...")
    result = await agent.run("Say hello")
    print(f"  Agent response: {result.text}")
    print("\n=== TEST PASSED ===\n")


if __name__ == "__main__":
    asyncio.run(main())
