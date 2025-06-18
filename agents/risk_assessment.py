from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI
from models import RiskData
from config import OPENAI_API_KEY, OPENAI_API_BASE


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

    def _should_validate(self, risk_data: RiskData) -> bool:
        """Determine if deep validation is needed based on confidence."""
        # Use DeepEval only when confidence is low or data is incomplete
        if risk_data.confidence_score < 0.7:
            return True
        if risk_data.data_completeness in ["limited", "partial"]:
            return True
        # Always validate if beta is missing (important metric)
        if risk_data.beta is None:
            return True
        return False

    def __call__(self, state: dict) -> dict:
        print("Running RiskAssessmentAgent...")
        state["messages"].append("Assessing stock risks...")

        for ticker in state["query"]["tickers"]:
            context = state["stocks_data"][ticker].get("market_data", "")
            if not context:
                state["error_messages"].append(
                    f"No market data context found for {ticker} to assess risk."
                )
                continue

            prompt = f"""
            Analyze the risk profile for stock {ticker}.
            Context:
            ---
            {context}
            ---

            Provide confidence_score (0-1) based on:
            - Availability of key metrics (beta, volatility data)
            - Quality of risk factor information
            - Completeness of financial data

            Set data_completeness:
            - "complete": All key risk metrics available
            - "partial": Some metrics missing but enough for assessment
            - "limited": Missing critical risk information

            Respond with JSON matching the RiskData schema.
            """

            try:
                risk_data: RiskData = self.llm.invoke(prompt)

                # Only run expensive validation if confidence/completeness is low
                if self._should_validate(risk_data):
                    state["messages"].append(f"-> {ticker} low confidence, running validation...")

                    test_case = LLMTestCase(
                        input=prompt,
                        actual_output=str(risk_data.model_dump()),
                        retrieval_context=[context]
                    )

                    self.validator.measure(test_case)
                    score = self.validator.score

                    if score < 0.7:
                        state["messages"].append(
                            f"-> {ticker} risk assessment failed validation ({score})"
                        )
                    else:
                        state["messages"].append(
                            f"-> {ticker} risk assessment validated ({score})"
                        )
                else:
                    state["messages"].append(
                        f"-> {ticker} high confidence ({risk_data.confidence_score}), skipping validation"
                    )

                state["stocks_data"][ticker]["risk_assessment"] = risk_data

            except Exception as e:
                state["error_messages"].append(f"Error assessing risk for {ticker}: {str(e)}")
                # Add default values so workflow can continue
                default_risk = RiskData(
                    volatility="unknown",
                    beta=None,
                    risk_factors=["Risk assessment failed"],
                    risk_score=5,
                    confidence_score=0.0,
                    data_completeness="limited"
                )
                state["stocks_data"][ticker]["risk_assessment"] = default_risk

        return state