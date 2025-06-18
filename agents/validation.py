from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


class ValidationAgent:
    """Final validation"""

    def __init__(self):
        self.faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")
        self.relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")

    def __call__(self, state: dict) -> dict:
        """Validate final output"""
        if not state.get("executive_summary"):
            return state

        # Build context from the market data that was used
        context = []
        for ticker, data in state["stocks_data"].items():
            # Include the market data that was fetched
            market_data = data.get("market_data", "")
            if market_data:
                context.append(market_data)

            # Include the processed analysis
            news = data.get("news_analysis", {})
            risk = data.get("risk_assessment", {})
            context.append(f"{ticker} news analysis: {news}")
            context.append(f"{ticker} risk assessment: {risk}")

        # Test faithfulness
        faith_test = LLMTestCase(
            input=str(state["query"]),
            actual_output=state["executive_summary"],
            retrieval_context=context  # Changed from 'context' to 'retrieval_context'
        )

        try:
            self.faithfulness.measure(faith_test)
            faith_score = self.faithfulness.score
        except Exception as e:
            print(f"Faithfulness validation error: {e}")
            faith_score = 0.0

        # Test relevancy
        try:
            self.relevancy.measure(faith_test)
            rel_score = self.relevancy.score
        except Exception as e:
            print(f"Relevancy validation error: {e}")
            rel_score = 0.0

        # Store results
        passed = faith_score > 0.7 and rel_score > 0.7

        val_result = state.get("validation_result", {})
        attempt = val_result.get("attempt", 0) + 1

        state["validation_result"] = {
            "passed": passed,
            "scores": {
                "faithfulness": faith_score,
                "relevancy": rel_score
            },
            "attempt": attempt
        }

        if not passed and attempt == 1:
            state["needs_retry"] = True
            state["executive_summary"] = None
            state["messages"].append(f"Retrying - F:{faith_score}, R:{rel_score}")
        elif not passed:
            state["messages"].append("Note: Some claims could not be fully verified")
        else:
            state["messages"].append(f"Validated - F:{faith_score}, R:{rel_score}")

        return state