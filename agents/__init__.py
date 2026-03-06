"""
agents package
Orchestrator and worker agents for multi-dimensional stock analysis.
"""

from agents.orchestrator import (
    extract_companies,
    analyse_company,
    WorkerResults,
)

from agents.workers import (
    get_price_tech_agent,
    get_fundamentals_agent,
    get_news_sentiment_agent,
    get_risk_agent,
)

__all__ = [
    "extract_companies",
    "analyse_company",
    "WorkerResults",
    "get_price_tech_agent",
    "get_fundamentals_agent",
    "get_news_sentiment_agent",
    "get_risk_agent",
]
