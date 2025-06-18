# zeevrispler/stock-researcher/stock-researcher-4b3618e2c0950ebe10c63249519e1b2dbb61748a/agents/validation.py
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from models import StockState, ValidationResult


def validate_results(state: StockState) -> StockState:
    synthesis_result = state.synthesis_result
    # Create the retrieval context from the raw data collected by the market data agent
    retrieval_context = [
        stock_data.market_data for stock_data in state.stocks_data.values() if stock_data.market_data
    ]

    faithfulness_metric = FaithfulnessMetric(threshold=0.8, model="gpt-4o")
    relevancy_metric = AnswerRelevancyMetric(threshold=0.8, model="gpt-4o")

    test_case = LLMTestCase(
        input=state.query.user_input,
        actual_output=synthesis_result,
        retrieval_context=retrieval_context,
    )

    faithfulness_metric.measure(test_case)
    relevancy_metric.measure(test_case)

    state.validation_result = ValidationResult(
        is_faithful=faithfulness_metric.is_successful(),
        faithfulness_score=faithfulness_metric.score,
        is_relevant=relevancy_metric.is_successful(),
        relevancy_score=relevancy_metric.score,
    )

    return state