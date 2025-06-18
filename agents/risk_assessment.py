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
        # Only pass base_url if it's not the default
        kwargs = {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY,
        }
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE

        self.llm = ChatOpenAI(**kwargs).with_structured_output(RiskData)
        self.validator = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")

    def __call__(self, state: dict) -> dict:
        print("Running RiskAssessmentAgent...")
        state["messages"].append("Assessing stock risks...")

        for ticker in state["query"]["tickers"]:
            # We now use the clean, synthesized text block from the ReAct agent
            # instead of parsing raw, unstructured search results.
            context = state["stocks_data"][ticker].get("market_data", "")
            if not context:
                state["error_messages"].append(
                    f"No market data context found for {ticker} to assess risk."
                )
                continue

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

            try:
                risk_data: RiskData = self.llm.invoke(prompt)

                # Validate the extracted data is faithful to the new, higher-quality context
                test_case = LLMTestCase(
                    input=prompt,
                    actual_output=json.dumps(risk_data.dict()),
                    retrieval_context=[context]  # Changed from 'context' to 'retrieval_context'
                )

                self.validator.measure(test_case)
                score = self.validator.score
                print(f"Faithfulness metric score for {ticker} risk: {score}")

                if score > 0.7:
                    state["stocks_data"][ticker]["risk_assessment"] = risk_data.dict()
                    state["messages"].append(
                        f"-> {ticker} risk assessment validated ({score})"
                    )
                else:
                    state["messages"].append(
                        f"-> {ticker} risk assessment validation warning ({score})"
                    )
                    # Still use the data but note the warning
                    state["stocks_data"][ticker]["risk_assessment"] = risk_data.model_dump()

            except Exception as e:
                state["error_messages"].append(f"Error assessing risk for {ticker}: {str(e)}")
                # Add default values so workflow can continue
                state["stocks_data"][ticker]["risk_assessment"] = {
                    "volatility": "unknown",
                    "beta": None,
                    "risk_factors": ["Risk assessment failed"],
                    "risk_score": 5,
                }
        print(f"Risk assessment state: {state['stocks_data']}")
        return state