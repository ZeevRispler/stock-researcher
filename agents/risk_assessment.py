import json
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_API_BASE


class RiskData(BaseModel):
    """
    The result of risk analysis for a stock.
    - volatility: The volatility of the stock, can be "high", "medium" or "low".
    - beta: The beta of the stock.
    - risk_factors: A list of risk factors for the stock.
    - risk_score: A score from 1-10 (10 = highest risk).
    """

    volatility: str = Field(
        ..., description='The volatility of the stock, can be "high", "medium" or "low".'
    )
    beta: float | None = Field(..., description="The beta of the stock.")
    risk_factors: list[str] = Field(
        ..., description="A list of risk factors for the stock."
    )
    risk_score: int = Field(..., description="A score from 1-10 (10 = highest risk).")


class RiskAssessmentAgent:
    def __init__(self):
        # Use structured output for reliable JSON and add base_url for consistency
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
        ).with_structured_output(RiskData)
        self.validator = FaithfulnessMetric(threshold=0.8, model="gpt-4o-mini")

    def __call__(self, state: dict) -> dict:
        state["messages"].append("Assessing stock risks...")

        for ticker in state["query"]["tickers"]:
            # --- THIS IS THE UPDATED SECTION ---
            # We now use the clean, synthesized text block from the ReAct agent
            # instead of parsing raw, unstructured search results.
            context = state["stocks_data"][ticker].get("market_data", "")
            if not context:
                state["error_messages"].append(
                    f"No market data context found for {ticker} to assess risk."
                )
                continue
            # --- END OF UPDATE ---

            prompt = f"""
            Analyze the risk profile for the stock with the ticker {ticker}, based on the following context.
            Pay special attention to market volatility, beta, and any mentioned competitive, regulatory, or operational risks.

            Context:
            ---
            {context}
            ---

            Respond with a JSON object that strictly follows this schema:
            {{
                "volatility": "high" | "medium" | "low",
                "beta": float_value | null,
                "risk_factors": ["factor_1", "factor_2", ...],
                "risk_score": integer_from_1_to_10
            }}
            """
            risk_data: RiskData = self.llm.invoke(prompt)

            # Validate the extracted data is faithful to the new, higher-quality context
            test_case = LLMTestCase(
                input=prompt,
                actual_output=json.dumps(risk_data.dict()),
                context=[context],
            )
            self.validator.measure(test_case)
            print(f"Faithfulness metric score for {ticker} risk: {self.validator.score}")

            if self.validator.score > 0.8:
                state["stocks_data"][ticker]["risk_assessment"] = risk_data.dict()
                state["messages"].append(
                    f"-> {ticker} risk assessment validated ({self.validator.score:.2f})"
                )
            else:
                state["messages"].append(
                    f"-> {ticker} risk assessment validation failed ({self.validator.score:.2f})."
                )
                state["stocks_data"][ticker]["risk_assessment"] = {
                    "volatility": "unknown",
                    "beta": None,
                    "risk_factors": ["Validation failed to confirm data from source."],
                    "risk_score": 5,
                }

        return state