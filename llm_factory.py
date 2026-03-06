"""
llm_factory.py
Builds the LangChain LLM object from environment variables.

Azure AI Foundry exposes a standard OpenAI-compatible endpoint at:
  https://<resource>.openai.azure.com/openai/v1/
so we use ChatOpenAI (not AzureChatOpenAI) with a custom base_url.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_key  = os.environ["AZURE_OPENAI_API_KEY"]
    deploy   = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

    # Ensure the /openai/v1 suffix is present
    if not endpoint.endswith("/openai/v1"):
        if "/openai/v1" not in endpoint:
            endpoint = endpoint.rstrip("/") + "/openai/v1"

    return ChatOpenAI(
        model=deploy,
        api_key=api_key,
        base_url=endpoint,   # <-- correct param for langchain-openai >= 1.0
        temperature=temperature,
        max_retries=3,
        timeout=120,
    )