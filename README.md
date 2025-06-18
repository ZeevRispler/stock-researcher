# Multi-Agent Stock Research System

A sophisticated stock analysis system built with **LangGraph**, **Tavily**, and **OpenAI** that uses multiple AI agents to research stocks and generate comprehensive reports.

## 🚀 Quick Start

### 1. Setup
```bash
git clone <your-repo-url>
cd stock-researcher
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Rename `.env.example` to `.env` and add your API keys:
```env
TAVILY_API_KEY="tvly-..."
OPENAI_API_KEY="sk-..."
```

### 3. Run the System

**CLI Interface:**
```bash
python run_cli.py "Compare Tesla vs Ford"
```

**Web Interface:**
```bash
python app.py
```

## 🤖 How It Works

1. **Query Parser**: Extracts stock tickers and determines analysis type (single vs comparison)
2. **Market Data Agent**: Uses ReAct pattern with Tavily to gather comprehensive market data
3. **News Sentiment Agent**: Analyzes recent news and sentiment with confidence scoring
4. **Risk Assessment Agent**: Evaluates volatility, beta, and risk factors
5. **Synthesis Agent**: Creates executive summaries or comparison dashboards
6. **Validation Agent**: Uses DeepEval to ensure report quality and factual accuracy

### ⚡ Performance Optimization
- **Confidence-Based Validation**: Only runs expensive DeepEval when AI confidence is low (<0.7)
- **Minimal API Calls**: 1-2 Tavily searches total, agents share data efficiently
- **Smart Retry Logic**: Auto-corrects failed validations once before providing results

## 📋 Example Queries

- `"Analyze GOOGL"` → Executive summary with sentiment and risk analysis
- `"Compare Apple and Microsoft"` → Side-by-side comparison with winner recommendation  
- `"NVDA vs AMD stock analysis"` → Detailed comparison of semiconductor stocks
- `"What is the risk profile for Tesla?"` → Focus on risk assessment and volatility
