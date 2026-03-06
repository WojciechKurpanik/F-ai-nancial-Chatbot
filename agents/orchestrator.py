"""
agents/orchestrator.py
Orchestrator agent — coordinates worker agents and produces the final report.
Orchestrator-worker pattern:
  1. Parse user query to identify companies.
  2. For each company, run all 4 worker agents concurrently.
  3. Synthesise worker outputs into a comprehensive investment report.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from agents.workers import (
    run_fundamentals_agent,
    run_news_sentiment_agent,
    run_price_tech_agent,
    run_risk_agent,
)


# ──────────────────────────────────────────────
#  Data class
# ──────────────────────────────────────────────

@dataclass
class WorkerResults:
    company: str
    technical: str = ""
    fundamental: str = ""
    sentiment: str = ""
    risk: str = ""


# ──────────────────────────────────────────────
#  Company extraction
# ──────────────────────────────────────────────

EXTRACT_SYSTEM = """You are a parser. Extract all company names or stock ticker symbols from the user query.
Return ONLY a JSON array of strings, e.g. ["Apple", "TSLA", "Microsoft"].
If none found, return []."""


async def extract_companies(query: str) -> list[str]:
    llm = get_llm(temperature=0)
    messages = [
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content=query),
    ]
    response = await llm.ainvoke(messages)
    text = response.content.strip()
    text = re.sub(r"```[a-z]*\n?", "", text).strip("`").strip()
    try:
        companies = json.loads(text)
        return [c for c in companies if isinstance(c, str)]
    except Exception:
        return [c.strip() for c in re.split(r"[,\n]+", text) if c.strip()]


# ──────────────────────────────────────────────
#  Worker runner (thread-pool safe)
# ──────────────────────────────────────────────

async def _run_worker_async(fn, query: str) -> str:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, fn, query)
    except Exception as e:
        return f"[Worker error: {e}]"


async def analyse_company(company: str, original_query: str) -> WorkerResults:
    """Run all 4 workers concurrently for a single company."""
    worker_query = f"Analyse {company}. User context: {original_query}"

    technical, fundamental, sentiment, risk = await asyncio.gather(
        _run_worker_async(run_price_tech_agent, worker_query),
        _run_worker_async(run_fundamentals_agent, worker_query),
        _run_worker_async(run_news_sentiment_agent, worker_query),
        _run_worker_async(run_risk_agent, worker_query),
    )

    return WorkerResults(
        company=company,
        technical=technical,
        fundamental=fundamental,
        sentiment=sentiment,
        risk=risk,
    )


# ──────────────────────────────────────────────
#  Synthesis
# ──────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You are a senior investment advisor synthesising multi-agent research reports.

Given worker analysis outputs for one or more companies, produce a comprehensive
investment report using this structure:

## 📊 Market Analysis Report

For each company analysed:

### [Company Name] ([TICKER])

**📈 Technical Analysis Summary**
- Key findings from technical agent
- 1-week and 1-month price outlook

**📋 Fundamental Analysis Summary**
- Valuation, earnings, growth, financial health highlights

**📰 News & Sentiment Summary**
- Recent catalysts, sentiment direction

**⚠️ Risk Assessment**
- Risk level and key risk factors

**🎯 Investment Prediction**
| Horizon | Direction | Confidence |
|---------|-----------|------------|
| 1 Week  | ↑/↓/→     | Low/Med/High |
| 1 Month | ↑/↓/→     | Low/Med/High |
| 3 Months| ↑/↓/→     | Low/Med/High |

**💡 Investment Opinion**
[Clear BUY / HOLD / SELL recommendation with rationale — 3-5 sentences]

---

End with a **Portfolio Summary** if multiple companies were analysed.

Always remind the user this is AI-generated analysis for informational purposes only
and does not constitute financial advice."""


async def synthesise_report(results: list[WorkerResults], original_query: str) -> str:
    llm = get_llm(temperature=0.2)

    worker_data = ""
    for r in results:
        worker_data += f"""
=== {r.company} ===

[TECHNICAL ANALYSIS]
{r.technical}

[FUNDAMENTAL ANALYSIS]
{r.fundamental}

[NEWS & SENTIMENT]
{r.sentiment}

[RISK ASSESSMENT]
{r.risk}

"""

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM),
        HumanMessage(content=f"User question: {original_query}\n\nWorker reports:\n{worker_data}"),
    ]
    response = await llm.ainvoke(messages)
    return response.content


# ──────────────────────────────────────────────
#  Main orchestrator entry point
# ──────────────────────────────────────────────

async def orchestrate(
    query: str,
    status_callback: Callable[[str], None] | None = None,
) -> str:
    def _status(msg: str):
        if status_callback:
            status_callback(msg)

    _status("🔍 Identifying companies in your query...")
    companies = await extract_companies(query)

    if not companies:
        return (
            "I couldn't identify any company names or ticker symbols in your query. "
            "Please mention specific companies (e.g. 'Apple', 'TSLA', 'Microsoft')."
        )

    _status(f"✅ Found: **{', '.join(companies)}**")

    all_results: list[WorkerResults] = []
    for company in companies:
        _status(
            f"\n🤖 Running 4 parallel agents for **{company}**:\n"
            "  • 🔢 Technical & Price\n"
            "  • 📊 Fundamentals\n"
            "  • 📰 News & Sentiment\n"
            "  • ⚠️ Risk Assessment"
        )
        result = await analyse_company(company, query)
        all_results.append(result)
        _status(f"✔️  {company} — all agents complete.")

    _status("\n📝 Synthesising final investment report...")
    return await synthesise_report(all_results, query)
