from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from models import StockState, ValidationResult


class ValidationAgent:
    """Final validation"""

    def __init__(self):
        self.faithfulness = FaithfulnessMetric(threshold=0.8, model="gpt-4o-mini")
        self.relevancy = AnswerRelevancyMetric(threshold=0.8, model="gpt-4o-mini")

    def __call__(self, state: StockState) -> StockState:
        """Validate final output"""
        if not state.executive_summary:
            return state

        # Build context
        context = []
        for ticker, data in state.stocks_data.items():
            # Include raw search
            raw = data.get("raw_search", {}).get("results", [])[:2]
            context.append(f"{ticker} data: {str(raw)}")

            # Include extracted data
            context.append(f"{ticker} analysis: {data.get('news_analysis', {})}, {data.get('risk_assessment', {})}")

        # Test faithfulness
        faith_test = LLMTestCase(
            input=str(state.query),
            actual_output=state.executive_summary,
            context=context
        )

        faith_result = self.faithfulness.measure(faith_test)

        # Test relevancy
        rel_result = self.relevancy.measure(faith_test)

        # Store results
        passed = faith_result.score > 0.8 and rel_result.score > 0.8

        state.validation_result = ValidationResult(
            passed=passed,
            scores={
                "faithfulness": faith_result.score,
                "relevancy": rel_result.score
            },
            attempt=getattr(state.validation_result, 'attempt', 0) + 1
        )

        if not passed and state.validation_result.attempt == 1:
            state.needs_retry = True
            state.executive_summary = None
            state.messages.append(f"Retrying - F:{faith_result.score:.2f}, R:{rel_result.score:.2f}")
        elif not passed:
            state.messages.append("Note: Some claims could not be fully verified")
        else:
            state.messages.append(f"Validated - F:{faith_result.score:.2f}, R:{rel_result.score:.2f}")

        return state