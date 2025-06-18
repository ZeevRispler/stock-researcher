from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


class ValidationAgent:
    def __init__(self):
        self.faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")
        self.relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")

    def __call__(self, state: dict) -> dict:
        print("✅ Validating report...")

        if not state.get("executive_summary"):
            return state

        context = []
        for ticker, data in state["stocks_data"].items():
            market_data = data.get("market_data", "")
            if market_data:
                context.append(market_data)

            news = data.get("news_analysis", {})
            risk = data.get("risk_assessment", {})
            context.append(f"{ticker} news analysis: {news}")
            context.append(f"{ticker} risk assessment: {risk}")

        faith_test = LLMTestCase(
            input=str(state["query"]),
            actual_output=state["executive_summary"],
            retrieval_context=context
        )

        try:
            self.faithfulness.measure(faith_test)
            faith_score = self.faithfulness.score
        except Exception as e:
            print(f"Faithfulness validation error: {e}")
            faith_score = 0.0

        try:
            self.relevancy.measure(faith_test)
            rel_score = self.relevancy.score
        except Exception as e:
            print(f"Relevancy validation error: {e}")
            rel_score = 0.0

        passed = faith_score > 0.7 and rel_score > 0.7
        validation_result = state.get("validation_result", {})
        attempt = validation_result.get("attempt", 1) + 1 if validation_result else 1

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
        elif not passed:
            print("Note: Some claims could not be fully verified")

        return state