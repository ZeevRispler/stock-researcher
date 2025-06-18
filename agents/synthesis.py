from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
import json


class SynthesisAgent:
    """Create outputs"""

    def __init__(self):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

    def __call__(self, state: dict) -> dict:
        """Generate outputs"""
        print("Running SynthesisAgent...")

        guidance = ""
        if state.get("needs_retry", False):
            guidance = "IMPORTANT: Only use information explicitly stated in the provided data."

        # Prepare clean context with confidence info
        context = {}
        for ticker in state["query"]["tickers"]:
            stock_data = state["stocks_data"][ticker]

            # Extract structured data from the new models
            news = stock_data.get("news_analysis")
            risk = stock_data.get("risk_assessment")

            # Convert Pydantic models to dict if needed
            if hasattr(news, 'model_dump'):
                news = news.model_dump()
            if hasattr(risk, 'model_dump'):
                risk = risk.model_dump()

            context[ticker] = {
                "news": news or {},
                "risk": risk or {}
            }

        analysis_type = state["query"].get("analysis_type", "single")

        if analysis_type == "single":
            prompt = f"""
            {guidance}
            Write a 150-word executive summary for {state["query"]["tickers"][0]} using:
            {json.dumps(context, indent=2)}

            Structure:
            1. Current sentiment and recent events
            2. Risk assessment
            3. Investment recommendation

            Note any low confidence scores in your analysis.
            """
        else:
            prompt = f"""
            {guidance}
            Compare {' vs '.join(state["query"]["tickers"])} using:
            {json.dumps(context, indent=2)}

            Write 200 words covering:
            1. Key differences
            2. Risk comparison
            3. Which is the better investment and why

            Note any low confidence scores in your analysis.
            """

        response = self.llm.invoke(prompt)
        state["executive_summary"] = response.content

        # Simple dashboard for comparisons
        if analysis_type == "comparison" and len(state["query"]["tickers"]) == 2:
            t1, t2 = state["query"]["tickers"]
            d1, d2 = state["stocks_data"][t1], state["stocks_data"][t2]

            # Get data from structured models
            news1 = d1.get('news_analysis', {})
            news2 = d2.get('news_analysis', {})
            risk1 = d1.get('risk_assessment', {})
            risk2 = d2.get('risk_assessment', {})

            # Handle both dict and Pydantic model formats
            if hasattr(news1, 'model_dump'):
                news1 = news1.model_dump()
            if hasattr(news2, 'model_dump'):
                news2 = news2.model_dump()
            if hasattr(risk1, 'model_dump'):
                risk1 = risk1.model_dump()
            if hasattr(risk2, 'model_dump'):
                risk2 = risk2.model_dump()

            # Convert sentiment strings to numeric scores
            sentiment_map = {"positive": 3, "neutral": 2, "negative": 1}
            sentiment1 = sentiment_map.get(news1.get('sentiment', 'neutral'), 2)
            sentiment2 = sentiment_map.get(news2.get('sentiment', 'neutral'), 2)

            risk_score1 = risk1.get('risk_score', 5)
            risk_score2 = risk2.get('risk_score', 5)

            state["comparison_dashboard"] = f"""
📊 {t1} vs {t2} Quick Comparison
═══════════════════════════════════════════════════════════
                    {t1:<10} {t2:<10} Better
───────────────────────────────────────────────────────────
Sentiment Score     {sentiment1:>6}     {sentiment2:>6}     {'→ ' + (t1 if sentiment1 > sentiment2 else t2)}
Risk Score (/10)    {risk_score1:>6}     {risk_score2:>6}     {'→ ' + (t1 if risk_score1 < risk_score2 else t2)}
Volatility          {risk1.get('volatility', 'N/A'):>6}     {risk2.get('volatility', 'N/A'):>6}
═══════════════════════════════════════════════════════════
Overall Winner: {t1 if (sentiment1 > sentiment2 and risk_score1 <= risk_score2) else t2}
"""
        print("Synthesis complete.")
        print("-" * 40)
        print(state["executive_summary"])
        return state