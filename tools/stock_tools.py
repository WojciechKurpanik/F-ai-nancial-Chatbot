"""
tools/stock_tools.py
LangChain tools wrapping yfinance for price, fundamentals, and technicals.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool


# ──────────────────────────────────────────────
#  Helper
# ──────────────────────────────────────────────

def _safe_round(v, digits=2):
    try:
        return round(float(v), digits)
    except Exception:
        return None


# ──────────────────────────────────────────────
#  Tools
# ──────────────────────────────────────────────

@tool
def get_stock_price(ticker: str) -> str:
    """
    Fetch the latest closing price and basic price metadata for a ticker symbol.
    Returns JSON with current price, 52-week high/low, and daily change %.
    """
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period="5d")
        if hist.empty:
            return json.dumps({"error": f"No price data found for {ticker}"})

        latest = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) > 1 else latest
        change_pct = ((latest - prev) / prev) * 100

        info = t.info
        result = {
            "ticker": ticker.upper(),
            "current_price": _safe_round(latest),
            "previous_close": _safe_round(prev),
            "daily_change_pct": _safe_round(change_pct),
            "week_52_high": _safe_round(info.get("fiftyTwoWeekHigh")),
            "week_52_low": _safe_round(info.get("fiftyTwoWeekLow")),
            "currency": info.get("currency", "USD"),
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_fundamentals(ticker: str) -> str:
    """
    Fetch fundamental financial data for a ticker: P/E, EPS, revenue growth,
    profit margin, debt-to-equity, market cap, and analyst recommendation.
    """
    try:
        info = yf.Ticker(ticker.upper()).info
        result = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": _safe_round(info.get("trailingPE")),
            "forward_pe": _safe_round(info.get("forwardPE")),
            "eps_ttm": _safe_round(info.get("trailingEps")),
            "revenue_growth_yoy": _safe_round(info.get("revenueGrowth")),
            "profit_margin": _safe_round(info.get("profitMargins")),
            "debt_to_equity": _safe_round(info.get("debtToEquity")),
            "return_on_equity": _safe_round(info.get("returnOnEquity")),
            "dividend_yield": _safe_round(info.get("dividendYield")),
            "analyst_recommendation": info.get("recommendationKey"),
            "target_mean_price": _safe_round(info.get("targetMeanPrice")),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_technical_indicators(ticker: str) -> str:
    """
    Calculate technical indicators: RSI(14), MACD, Bollinger Bands, SMA20/50/200,
    and average volume. Used to assess momentum and trend direction.
    """
    try:
        hist = yf.Ticker(ticker.upper()).history(period="1y")
        if hist.empty or len(hist) < 50:
            return json.dumps({"error": f"Insufficient history for {ticker}"})

        close = hist["Close"]

        # SMA
        sma20 = _safe_round(close.rolling(20).mean().iloc[-1])
        sma50 = _safe_round(close.rolling(50).mean().iloc[-1])
        sma200 = _safe_round(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = _safe_round(100 - (100 / (1 + rs.iloc[-1])))

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        macd_val = _safe_round(macd_line.iloc[-1])
        signal_val = _safe_round(signal_line.iloc[-1])
        macd_hist = _safe_round(macd_line.iloc[-1] - signal_line.iloc[-1])

        # Bollinger Bands (20-day)
        sma20_full = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = _safe_round((sma20_full + 2 * std20).iloc[-1])
        bb_lower = _safe_round((sma20_full - 2 * std20).iloc[-1])

        # Volume
        avg_vol = _safe_round(hist["Volume"].rolling(20).mean().iloc[-1])
        last_vol = int(hist["Volume"].iloc[-1])

        current_price = _safe_round(close.iloc[-1])

        # Trend signal
        trend = "bullish" if current_price > sma50 > sma20 * 0 else (
            "bearish" if current_price < sma50 else "neutral"
        )
        if sma200 and current_price > sma200:
            long_trend = "above_200sma (long-term bullish)"
        elif sma200:
            long_trend = "below_200sma (long-term bearish)"
        else:
            long_trend = "unknown"

        result = {
            "ticker": ticker.upper(),
            "current_price": current_price,
            "sma_20": sma20,
            "sma_50": sma50,
            "sma_200": sma200,
            "rsi_14": rsi,
            "rsi_signal": "overbought" if rsi and rsi > 70 else ("oversold" if rsi and rsi < 30 else "neutral"),
            "macd": macd_val,
            "macd_signal": signal_val,
            "macd_histogram": macd_hist,
            "macd_crossover": "bullish" if macd_hist and macd_hist > 0 else "bearish",
            "bollinger_upper": bb_upper,
            "bollinger_lower": bb_lower,
            "avg_volume_20d": avg_vol,
            "last_volume": last_vol,
            "volume_spike": last_vol > avg_vol * 1.5 if avg_vol else False,
            "long_term_trend": long_trend,
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_price_history_summary(ticker: str) -> str:
    """
    Return a weekly/monthly/3-month performance summary and volatility metrics.
    Useful for predicting short-term price movement.
    """
    try:
        hist = yf.Ticker(ticker.upper()).history(period="6mo")
        if hist.empty:
            return json.dumps({"error": f"No data for {ticker}"})

        close = hist["Close"]
        now = close.iloc[-1]

        def pct(n_days):
            if len(close) > n_days:
                past = close.iloc[-n_days]
                return _safe_round(((now - past) / past) * 100)
            return None

        # Volatility (annualised)
        daily_returns = close.pct_change().dropna()
        vol_annual = _safe_round(daily_returns.std() * np.sqrt(252) * 100)
        vol_30d = _safe_round(daily_returns.tail(30).std() * np.sqrt(252) * 100)

        result = {
            "ticker": ticker.upper(),
            "performance_1w_pct": pct(5),
            "performance_1m_pct": pct(21),
            "performance_3m_pct": pct(63),
            "performance_6m_pct": pct(126),
            "annualised_volatility_pct": vol_annual,
            "volatility_30d_pct": vol_30d,
            "risk_level": (
                "high" if vol_30d and vol_30d > 40
                else "medium" if vol_30d and vol_30d > 20
                else "low"
            ),
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def search_company_news(query: str) -> str:
    """
    Search for recent news about a company using Tavily web search.
    Input should be a company name or ticker + relevant keywords.
    Falls back to a placeholder if Tavily is not configured.
    """
    import os
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps({
            "note": "Tavily API key not set. Configure TAVILY_API_KEY for live news.",
            "query": query,
            "results": []
        })
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=f"{query} stock news 2024 2025",
            max_results=5,
            search_depth="advanced",
        )
        results = [
            {"title": r.get("title"), "url": r.get("url"), "content": r.get("content", "")[:300]}
            for r in response.get("results", [])
        ]
        return json.dumps({"query": query, "results": results})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def resolve_ticker(company_name: str) -> str:
    """
    Try to resolve a company name to its stock ticker symbol.
    Returns the best-guess ticker or a list of candidates.
    """
    # Simple hard-coded map for common names; yfinance search as fallback
    common = {
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
        "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
        "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA",
        "netflix": "NFLX", "adobe": "ADBE", "salesforce": "CRM",
        "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM",
        "paypal": "PYPL", "uber": "UBER", "airbnb": "ABNB",
        "spotify": "SPOT", "twitter": "X", "x": "X",
    }
    name_lower = company_name.lower().strip()
    if name_lower in common:
        return json.dumps({"company": company_name, "ticker": common[name_lower]})

    # If it looks like a ticker already (short, uppercase)
    if len(company_name) <= 5 and company_name.isupper():
        return json.dumps({"company": company_name, "ticker": company_name})

    return json.dumps({
        "company": company_name,
        "ticker": company_name.upper(),
        "note": "Could not auto-resolve. Using input as ticker — verify manually."
    })


@tool
def screen_stocks_by_criteria(
    budget: float,
    target_return_pct: float,
    horizon_months: int,
    risk_tolerance: str = "medium",
) -> str:
    """
    Screen a curated watchlist of liquid stocks and return candidates
    that historically match the target return over the given horizon.

    Args:
        budget: Available capital in USD
        target_return_pct: Desired return as a percentage (e.g. 5.0 for 5%)
        horizon_months: Investment horizon in months (1, 3, 6, 12)
        risk_tolerance: "low", "medium", or "high"

    Returns JSON with ranked candidates including price, volatility, and fit score.
    """
    # Curated universe split by risk profile
    universes = {
        "low": [
            "MSFT", "AAPL", "JNJ", "PG", "KO", "VZ", "WMT", "BRK-B", "V", "MA",
        ],
        "medium": [
            "NVDA", "GOOGL", "AMZN", "META", "CRM", "AMD", "ADBE", "PYPL",
            "NFLX", "UBER", "NOW", "SNOW",
        ],
        "high": [
            "TSLA", "COIN", "MSTR", "PLTR", "RKLB", "SMCI", "IONQ", "RIVN",
            "ARKK", "SQQQ", "TQQQ",
        ],
    }

    # Include lower-risk tickers in broader pools
    if risk_tolerance == "medium":
        tickers = universes["low"] + universes["medium"]
    elif risk_tolerance == "high":
        tickers = universes["low"] + universes["medium"] + universes["high"]
    else:
        tickers = universes["low"]

    period_map = {1: "3mo", 3: "6mo", 6: "1y", 12: "2y"}
    yf_period = period_map.get(horizon_months, "6mo")
    trading_days = int(horizon_months * 21)

    candidates = []

    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period=yf_period)
            if hist.empty or len(hist) < trading_days:
                continue

            close = hist["Close"]
            current_price = _safe_round(close.iloc[-1])

            # Historical return over the horizon window
            past_price = close.iloc[-trading_days]
            hist_return = ((close.iloc[-1] - past_price) / past_price) * 100

            # Volatility
            daily_ret = close.pct_change().dropna()
            vol = _safe_round(daily_ret.tail(60).std() * (252 ** 0.5) * 100)

            # Shares purchasable
            shares = int(budget // current_price) if current_price else 0
            projected_gain = _safe_round((budget * target_return_pct) / 100)
            required_price = _safe_round(current_price * (1 + target_return_pct / 100)) if current_price else None

            # Simple fit score: how close is historical return to target
            return_gap = abs(hist_return - target_return_pct)
            fit_score = max(0, round(100 - return_gap * 2, 1))

            candidates.append({
                "ticker": ticker,
                "current_price": current_price,
                "shares_affordable": shares,
                "historical_return_pct": _safe_round(hist_return),
                "annualised_volatility_pct": vol,
                "target_return_pct": target_return_pct,
                "required_price_at_target": required_price,
                "projected_gain_usd": projected_gain,
                "fit_score": fit_score,
            })
        except Exception:
            continue

    # Sort by fit score descending, take top 8
    candidates.sort(key=lambda x: x["fit_score"], reverse=True)
    top = candidates[:8]

    return json.dumps({
        "budget": budget,
        "target_return_pct": target_return_pct,
        "horizon_months": horizon_months,
        "risk_tolerance": risk_tolerance,
        "candidates": top,
    })