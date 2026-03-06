"""
debug_connection.py
Run this BEFORE the main app to pinpoint exactly where the 500 occurs.

    uv run python debug_connection.py

It tests 4 things in order and stops at the first failure.
"""

import os, json, sys
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
API_KEY  = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOY   = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

if not ENDPOINT.endswith("/openai/v1"):
    ENDPOINT = ENDPOINT.rstrip("/") + "/openai/v1"

print(f"\n{'='*60}")
print(f"Endpoint : {ENDPOINT}")
print(f"Model    : {DEPLOY}")
print(f"Key      : {API_KEY[:8]}...")
print(f"{'='*60}\n")

# ── Test 1: raw HTTP — no SDK at all ──────────────────────────
print("TEST 1: Raw HTTP call (no SDK)...")
import urllib.request, ssl

payload = json.dumps({
    "model": DEPLOY,
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "temperature": 0,
    "max_tokens": 10,
}).encode()

req = urllib.request.Request(
    f"{ENDPOINT}/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json", "api-key": API_KEY},
)
try:
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        body = json.loads(resp.read())
        print(f"  ✅ Raw HTTP OK — reply: {body['choices'][0]['message']['content']}\n")
except Exception as e:
    print(f"  ❌ Raw HTTP FAILED: {e}")
    print("\n  → Check your ENDPOINT and API_KEY values in .env")
    sys.exit(1)

# ── Test 2: openai SDK directly ───────────────────────────────
print("TEST 2: openai SDK (no LangChain)...")
try:
    from openai import OpenAI
    client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=DEPLOY,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        temperature=0,
        max_tokens=10,
    )
    print(f"  ✅ openai SDK OK — reply: {resp.choices[0].message.content}\n")
except Exception as e:
    print(f"  ❌ openai SDK FAILED: {e}\n")
    sys.exit(1)

# ── Test 3: LangChain ChatOpenAI, no tools ────────────────────
print("TEST 3: LangChain ChatOpenAI (no tools)...")
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    llm = ChatOpenAI(model=DEPLOY, api_key=API_KEY, base_url=ENDPOINT,
                     temperature=0, max_tokens=10)
    resp = llm.invoke([HumanMessage(content="Say hello in one word.")])
    print(f"  ✅ LangChain OK — reply: {resp.content}\n")
except Exception as e:
    print(f"  ❌ LangChain FAILED: {e}\n")
    sys.exit(1)

# ── Test 4: LangChain with a single simple tool ───────────────
print("TEST 4: LangChain with tool calling...")
try:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"Sunny in {city}, 22°C"

    llm_with_tools = ChatOpenAI(
        model=DEPLOY, api_key=API_KEY, base_url=ENDPOINT, temperature=0
    ).bind_tools([get_weather])

    resp = llm_with_tools.invoke([HumanMessage(content="What's the weather in Warsaw?")])
    print(f"  ✅ Tool calling OK — tool calls: {resp.tool_calls}\n")
except Exception as e:
    print(f"  ❌ Tool calling FAILED: {e}")
    print("\n  → This means the model/endpoint doesn't support tool calling.")
    print("    Possible fixes:")
    print("    1. Confirm your deployment supports function calling in Foundry portal")
    print("    2. Try a different API version header")
    sys.exit(1)

print("="*60)
print("All tests passed! The app should work correctly.")
print("="*60)
