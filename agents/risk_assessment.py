import json
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_API_BASE


class RiskData(BaseModel):
    volatility: str = Field(..., description='The volatility of the stock, can be "high", "medium" or "low".')
    beta: float | None = Field(..., description="The beta of the stock.")
    risk_factors: list[str] = Field(..., description="A list of risk factors for the stock.")
    risk_score: int = Field(..., description="A score from 1-10 (10 = highest risk).")
    confidence_score: float = Field(..., description="Confidence 0-1 in this assessment")
    data_completeness: str = Field(..., description="'complete', 'partial', 'limited'")


class RiskAssessmentAgent:
    def __init__(self):
        kwargs = {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY,
        }
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE

        self.llm = ChatOpenAI(**kwargs).with_structured_output(RiskData)
        self.validator = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")

    def __call__(self, state: dict) -> dict:
        print("⚠️ Assessing stock risks...")

        for ticker in state["query"]["tickers"]:
            context = state["stocks_data"][ticker].get("market_data", "")
            if not context:
                state["error_messages"].append(f"No market data context found for {ticker} to assess risk.")
                continue

            prompt = f"""
            Analyze the risk profile for the stock with the ticker {ticker}, based on the following context.
            Pay special attention to market volatility, beta, and any mentioned competitive, regulatory, or operational risks.

            Provide confidence_score (0-1) based on:
            - Availability of key metrics (beta, volatility data)
            - Quality of risk factor information
            - Completeness of financial data

            Set data_completeness:
            - "complete": All key risk metrics available
            - "partial": Some metrics missing but enough for assessment
            - "limited": Missing critical risk information

            Context:
            ---
            {context}
            ---

            Respond with JSON matching the RiskData schema.
            """

            try:
                risk_data: RiskData = self.llm.invoke(prompt)

                # Only run expensive validation if confidence/completeness is low
                if risk_data.confidence_score < 0.7 or risk_data.data_completeness in ["limited", "partial"]:
                    test_case = LLMTestCase(
                        input=prompt,
                        actual_output=json.dumps(risk_data.dict()),
                        retrieval_context=[context]
                    )
                    self.validator.measure(test_case)
                    score = self.validator.score
                    if score < 0.7:
                        print(f"Risk assessment validation warning ({score})")

                state["stocks_data"][ticker]["risk_assessment"] = risk_data.model_dump()

            except Exception as e:
                state["error_messages"].append(f"Error assessing risk for {ticker}: {str(e)}")
                state["stocks_data"][ticker]["risk_assessment"] = {
                    "volatility": "unknown",
                    "beta": None,
                    "risk_factors": ["Risk assessment failed"],
                    "risk_score": 5,
                }

        return state