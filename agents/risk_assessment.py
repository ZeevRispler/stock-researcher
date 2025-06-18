# zeevrispler/stock-researcher/stock-researcher-4b3618e2c0950ebe10c63249519e1b2dbb61748a/agents/risk_assessment.py
import json

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_API_BASE
from models import StockState


class Risk(BaseModel):
    risk_level: str = Field(..., description="The overall risk level (e.g., 'Low', 'Medium', 'High').")
    confidence_score: float = Field(
        ..., description="A confidence score for the risk assessment, between 0 and 1."
    )
    summary: str = Field(..., description="A brief summary of the key risks identified.")


class RiskAssessmentAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            temperature=0,
        ).with_structured_output(Risk)
        self.validator = FaithfulnessMetric(threshold=0.8, model="gpt-4o")

    def __call__(self, state: StockState):
        for ticker, stock_data in state.stocks_data.items():
            source_text = stock_data.market_data or "No data available."
            if source_text == "No data available.":
                stock_data.risk_analysis = "No market data to analyze for risks."
                continue

            prompt = f"""
            Assess the investment risk for {ticker} based on the following market data.
            Analyze all available information, including stock price, P/E ratio, beta, and company summaries, to determine if the risk is Low, Medium, or High.
            Provide a summary of the key factors driving your risk assessment.

            Market Data:
            {source_text}
            """
            response: Risk = self.llm.invoke(prompt)

            test_case = LLMTestCase(
                input=prompt,
                actual_output=response.summary,
                retrieval_context=[source_text]
            )
            self.validator.measure(test_case)

            if self.validator.is_successful():
                stock_data.risk_analysis = json.dumps(response.dict(), indent=2)
            else:
                stock_data.risk_analysis = "Failed to generate a faithful risk analysis."
        return state