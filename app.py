"""
app.py  —  Chainlit chat interface for the Stock Market Agent
Run with:  chainlit run app.py
"""

import asyncio

import chainlit as cl

from agents.orchestrator import orchestrate

# ──────────────────────────────────────────────
#  Welcome screen
# ──────────────────────────────────────────────

WELCOME_MSG = """
# 📈 Stock Market Intelligence Agent

Welcome! I'm your AI-powered investment research assistant.

⚠️ This tool is for **informational purposes only**. Always consult a licensed financial advisor before investing.
"""


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=WELCOME_MSG).send()


# ──────────────────────────────────────────────
#  Main message handler
# ──────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()

    if not query:
        await cl.Message(content="Please enter a question about a stock or company.").send()
        return

    # ---- Status step container ----
    async with cl.Step(name="🤖 Multi-Agent Analysis Pipeline", type="run") as step:
        status_messages: list[str] = []

        def status_callback(msg: str):
            status_messages.append(msg)
            # We'll update the step content progressively
            asyncio.create_task(
                step.stream_token(f"\n{msg}")
            )

        try:
            report = await orchestrate(query, status_callback=status_callback)
        except Exception as e:
            report = f"❌ An error occurred during analysis:\n\n```\n{str(e)}\n```\n\nPlease check your `.env` configuration and try again."

        step.output = "\n".join(status_messages)

    # ---- Final report ----
    await cl.Message(content=report).send()
