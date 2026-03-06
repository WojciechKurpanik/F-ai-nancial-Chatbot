# 📈 Stock Market Intelligence Agent

A multi-agent LangChain application that analyses stocks using the **orchestrator-worker** pattern.  
Chat interface powered by **Chainlit**. LLM hosted on **Azure OpenAI** (or standard OpenAI).

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           ORCHESTRATOR AGENT            │
│  1. Extract company names from query    │
│  2. Dispatch to worker agents           │
│  3. Synthesise final investment report  │
└────────────┬────────────────────────────┘
             │  (parallel execution per company)
    ┌────────┴─────────────────────────┐
    │                                  │
    ▼                                  ▼
┌──────────────┐              ┌──────────────┐
│  TECHNICAL   │              │ FUNDAMENTAL  │
│    AGENT     │              │    AGENT     │
│              │              │              │
│ • Price data │              │ • P/E ratio  │
│ • RSI / MACD │              │ • EPS / ROE  │
│ • Bollinger  │              │ • Revenue    │
│ • SMA trends │              │ • Debt/Eq    │
│ • Volatility │              │ • Analysts   │
└──────────────┘              └──────────────┘
┌──────────────┐              ┌──────────────┐
│    NEWS &    │              │    RISK      │
│  SENTIMENT   │              │  ASSESSMENT  │
│    AGENT     │              │    AGENT     │
│              │              │              │
│ • Recent     │              │ • Volatility │
│   headlines  │              │ • Balance    │
│ • Catalysts  │              │   sheet risk │
│ • Sentiment  │              │ • Valuation  │
│   scoring    │              │   risk       │
└──────────────┘              └──────────────┘
             │
             ▼
    Investment Report
    ┌──────────────────┐
    │ Per-company:     │
    │ • Analysis recap │
    │ • 1W/1M/3M pred  │
    │ • BUY/HOLD/SELL  │
    │ Portfolio summary│
    └──────────────────┘
```

---

## ⚙️ Setup

### 1. Clone & install

```bash
cd stock_agent_app
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

#### Azure OpenAI (recommended)

```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01
```

#### Standard OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o
```

#### Optional — Tavily news search

```env
TAVILY_API_KEY=your_tavily_key
```

Without Tavily the News agent will skip live news but all other agents work normally.

### 3. Run

```bash
chainlit run src/app.py
```

Open your browser at **http://localhost:8000**

---

## 💬 Example Queries

| Query | What happens |
|-------|-------------|
| `"Should I invest in Apple?"` | Full 4-agent analysis → BUY/HOLD/SELL |
| `"Compare Tesla and NVIDIA"` | Both analysed in parallel, portfolio summary |
| `"What's the risk of investing in AMZN?"` | Risk-focused analysis with all signals |
| `"Analyse MSFT, GOOGL, META for next month"` | Three-company report with predictions |

---

## 📁 Project Structure

```
stock_agent_app/
├── src/
│   ├── __init__.py
│   ├── app.py                  # Chainlit chat entrypoint
│   ├── llm_factory.py          # Azure / OpenAI LLM builder
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Orchestrator + synthesis
│   │   └── workers.py          # 4 worker agents
│   └── tools/
│       ├── __init__.py
│       └── stock_tools.py      # yfinance + Tavily tools
├── requirements.txt
├── .env.example
├── .chainlit/
│   └── config.toml         # UI settings
```

---

## 🔧 Worker Agents

| Agent | Tools Used | Output |
|-------|-----------|--------|
| **Technical** | `get_stock_price`, `get_technical_indicators`, `get_price_history_summary` | RSI, MACD, SMA, Bollinger Bands, trend signal |
| **Fundamental** | `get_fundamentals` | P/E, EPS, growth, debt, analyst consensus |
| **News & Sentiment** | `search_company_news` | Recent catalysts, sentiment direction |
| **Risk** | `get_price_history_summary`, `get_fundamentals` | Volatility, balance sheet, valuation risk |

---

## ⚠️ Disclaimer

This application is for **informational and educational purposes only**.  
It does not constitute financial advice. Always consult a licensed financial advisor before making investment decisions.
