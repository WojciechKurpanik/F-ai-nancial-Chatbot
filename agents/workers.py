"""
agents/workers.py
Four specialised worker agents — each focuses on one analysis dimension.
Uses langgraph's create_react_agent which is stable across LangChain 0.2+
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from llm_factory import get_llm
from tools import (
    get_fundamentals,
    get_price_history_summary,
    get_stock_price,
    get_technical_indicators,
    resolve_ticker,
    search_company_news,
    screen_stocks_by_criteria,
)


# ──────────────────────────────────────────────
#  Agent builder — uses langgraph prebuilt React agent
#  (stable in LangChain 0.2+ / langgraph 0.1+)
# ──────────────────────────────────────────────

def _build_agent(system_prompt: str, tools: list):
    from langgraph.prebuilt import create_react_agent
    llm = get_llm(temperature=0.1)
    return create_react_agent(llm, tools, prompt=system_prompt)


def _run_agent(agent, query: str) -> str:
    """Invoke a langgraph agent and extract the final text response."""
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            return msg.content
    return "No output returned."


# ──────────────────────────────────────────────
#  Worker 1 — Price & Technical Agent
# ──────────────────────────────────────────────

PRICE_TECH_SYSTEM = """You are a quantitative technical analysis specialist for stock markets.

Your job:
1. Resolve the company/ticker using resolve_ticker if needed.
2. Fetch current price data with get_stock_price.
3. Fetch technical indicators with get_technical_indicators.
4. Fetch price history summary with get_price_history_summary.

Then produce a structured technical analysis covering:
- Current price and recent trend
- RSI signal (overbought/oversold/neutral)
- MACD crossover signal (bullish/bearish)
- SMA trend (above/below key moving averages)
- Bollinger Band position
- Volatility level
- Short-term price outlook (1 week, 1 month)

Be concise, data-driven, and end with a TECHNICAL SIGNAL: [BULLISH | BEARISH | NEUTRAL]."""


def get_price_tech_agent():
    return _build_agent(
        PRICE_TECH_SYSTEM,
        [resolve_ticker, get_stock_price, get_technical_indicators, get_price_history_summary],
    )


def run_price_tech_agent(query: str) -> str:
    return _run_agent(get_price_tech_agent(), query)


# ──────────────────────────────────────────────
#  Worker 2 — Fundamentals Agent
# ──────────────────────────────────────────────

FUNDAMENTALS_SYSTEM = """You are a fundamental analysis expert specialising in equity valuation.

Your job:
1. Resolve the company/ticker using resolve_ticker if needed.
2. Fetch fundamentals using get_fundamentals.

Then produce a structured fundamental analysis covering:
- Valuation (P/E, Forward P/E vs sector norms)
- Earnings quality (EPS, profit margin)
- Growth (revenue growth YoY)
- Financial health (debt-to-equity, ROE)
- Analyst consensus and price target
- Dividend profile (if applicable)
- Overall fundamental health: STRONG / MODERATE / WEAK

Be concise and end with a FUNDAMENTAL SIGNAL: [BULLISH | BEARISH | NEUTRAL]."""


def get_fundamentals_agent():
    return _build_agent(FUNDAMENTALS_SYSTEM, [resolve_ticker, get_fundamentals])


def run_fundamentals_agent(query: str) -> str:
    return _run_agent(get_fundamentals_agent(), query)


# ──────────────────────────────────────────────
#  Worker 3 — News & Sentiment Agent
# ──────────────────────────────────────────────

NEWS_SENTIMENT_SYSTEM = """You are a financial news analyst and market sentiment specialist.

Your job:
1. Search for recent news about the company using search_company_news.
2. Analyse sentiment from the headlines and snippets (positive / negative / mixed).
3. Identify major catalysts: earnings, product launches, regulatory issues, macro events.
4. Assess how current news might affect stock price over the next 1-3 months.

End with a SENTIMENT SIGNAL: [POSITIVE | NEGATIVE | MIXED].
If no news API key is configured, note that and give a generic market-context comment."""


def get_news_sentiment_agent():
    return _build_agent(NEWS_SENTIMENT_SYSTEM, [resolve_ticker, search_company_news])


def run_news_sentiment_agent(query: str) -> str:
    return _run_agent(get_news_sentiment_agent(), query)


# ──────────────────────────────────────────────
#  Worker 4 — Risk Assessment Agent
# ──────────────────────────────────────────────

RISK_SYSTEM = """You are a risk management analyst for equity investments.

Your job:
1. Resolve the company/ticker using resolve_ticker if needed.
2. Fetch price history and volatility using get_price_history_summary.
3. Fetch fundamentals using get_fundamentals to assess balance sheet risk.

Then assess:
- Market risk (volatility, beta equivalent based on price swings)
- Fundamental risk (high debt, negative growth, low margins)
- Valuation risk (stretched P/E)
- Liquidity risk
- Overall risk rating: LOW / MEDIUM / HIGH / VERY HIGH

End with a RISK RATING: [LOW | MEDIUM | HIGH | VERY HIGH]."""


def get_risk_agent():
    return _build_agent(
        RISK_SYSTEM,
        [resolve_ticker, get_price_history_summary, get_fundamentals],
    )


def run_risk_agent(query: str) -> str:
    return _run_agent(get_risk_agent(), query)


# ──────────────────────────────────────────────
#  Worker 5 — Portfolio Advisor Agent
# ──────────────────────────────────────────────

PORTFOLIO_ADVISOR_SYSTEM = """You are a portfolio advisor helping retail investors find the right stocks
for their budget, return target, and risk tolerance.

Your job:
1. Extract from the user query:
   - budget (USD amount to invest)
   - target return % (e.g. 5% return)
   - horizon in months (e.g. "3 months" → 3)
   - risk tolerance: if not stated, infer from target (>15% = high, 5-15% = medium, <5% = low)

2. Call screen_stocks_by_criteria with these parameters.

3. For the top 5 candidates returned, call get_technical_indicators and get_price_history_summary
   to validate current momentum.

4. Produce a ranked recommendation table:

### 🏆 Top Investment Candidates

| Rank | Ticker | Price | Shares | Target Price | Hist. Return | Volatility | Momentum | Fit |
|------|--------|-------|--------|--------------|--------------|------------|----------|-----|
| 1    | ...    | ...   | ...    | ...          | ...          | ...        | ...      | ... |

Then for each top 3, write 2-3 sentences explaining WHY it fits the goal.

5. Suggest a simple allocation if the user wants to split across multiple stocks
   (e.g. 50% in pick #1, 30% in pick #2, 20% in pick #3).

6. End with a clear caveat that past returns do not guarantee future results and this
   is not financial advice.

Always be specific with numbers. Show exactly how many shares $X buys and what the
portfolio looks like at the target return."""


def get_portfolio_advisor_agent():
    return _build_agent(
        PORTFOLIO_ADVISOR_SYSTEM,
        [
            screen_stocks_by_criteria,
            get_technical_indicators,
            get_price_history_summary,
            get_stock_price,
            resolve_ticker,
        ],
    )


def run_portfolio_advisor_agent(query: str) -> str:
    return _run_agent(get_portfolio_advisor_agent(), query)