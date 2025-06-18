# zeevrispler/stock-researcher/stock-researcher-4b3618e2c0950ebe10c63249519e1b2dbb61748a/agents/synthesis.py
import json

from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY
from models import StockState


class SynthesisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=0,
        )

    def __call__(self, state: StockState):
        stocks_for_prompt = {ticker: data.dict() for ticker, data in state.stocks_data.items()}

        prompt = f"""
        You are a senior financial analyst. Based on the query and the collected data, generate a comprehensive report.

        **User Query:** {state.query.user_input}

        **Collected Data:**
        {json.dumps(stocks_for_prompt, indent=2)}

        Please provide a detailed analysis and an executive summary.
        If the request is for a comparison, create a comparison dashboard in Markdown table format.
        """
        response = self.llm.invoke(prompt)
        state.synthesis_result = response.content
        return state