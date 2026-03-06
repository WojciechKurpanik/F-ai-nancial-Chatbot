"""
app.py  —  Chainlit chat interface for the Stock Market Agent
Run with:  chainlit run app.py
"""

import asyncio
import chainlit as cl
from agents.orchestrator import orchestrate

WELCOME_MSG = """
# 📈 Stock Market Intelligence Agent

Welcome! I'm your AI-powered investment research assistant backed by **5 specialised agents**.

⚠️ For informational purposes only. Not financial advice.
"""


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=WELCOME_MSG).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    if not query:
        await cl.Message(content="Please enter a question about a stock or company.").send()
        return

    async with cl.Step(name="🤖 Multi-Agent Analysis Pipeline", type="run") as step:
        status_messages: list[str] = []

        def status_callback(msg: str):
            status_messages.append(msg)
            asyncio.create_task(step.stream_token(f"\n{msg}"))

        try:
            report = await orchestrate(query, status_callback=status_callback)
        except Exception as e:
            report = (
                f"❌ An error occurred:\n\n```\n{str(e)}\n```\n\n"
                "Please check your `.env` configuration and try again."
            )

        step.output = "\n".join(status_messages)

    await cl.Message(content=report).send()