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

        # Prepare clean context
        context = {}
        for ticker in state["query"]["tickers"]:
            context[ticker] = {
                "news": state["stocks_data"][ticker].get("news_analysis", {}),
                "risk": state["stocks_data"][ticker].get("risk_assessment", {})
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
            """

        response = self.llm.invoke(prompt)
        state["executive_summary"] = response.content

        # Simple dashboard for comparisons
        if analysis_type == "comparison" and len(state["query"]["tickers"]) == 2:
            t1, t2 = state["query"]["tickers"]
            d1, d2 = state["stocks_data"][t1], state["stocks_data"][t2]

            # Convert sentiment strings to numeric scores
            sentiment_map = {"positive": 3, "neutral": 2, "negative": 1}
            sentiment1 = sentiment_map.get(d1.get('news_analysis', {}).get('sentiment', 'neutral'), 2)
            sentiment2 = sentiment_map.get(d2.get('news_analysis', {}).get('sentiment', 'neutral'), 2)

            risk1 = d1.get('risk_assessment', {}).get('risk_score', 5)
            risk2 = d2.get('risk_assessment', {}).get('risk_score', 5)

            state["comparison_dashboard"] = f"""
📊 {t1} vs {t2} Quick Comparison
═══════════════════════════════════════════════════════════
                    {t1:<10} {t2:<10} Better
───────────────────────────────────────────────────────────
Sentiment Score     {sentiment1:>6}     {sentiment2:>6}     {'→ ' + (t1 if sentiment1 > sentiment2 else t2)}
Risk Score (/10)    {risk1:>6}     {risk2:>6}     {'→ ' + (t1 if risk1 < risk2 else t2)}
Volatility          {d1.get('risk_assessment', {}).get('volatility', 'N/A'):>6}     {d2.get('risk_assessment', {}).get('volatility', 'N/A'):>6}
═══════════════════════════════════════════════════════════
Overall Winner: {t1 if (sentiment1 > sentiment2 and risk1 <= risk2) else t2}
"""
        print("Synthesis complete.")
        print("-" * 40)
        print(state["executive_summary"])
        return state