from langchain.chat_models import ChatOpenAI
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from models import StockState
from config import OPENAI_API_KEY
import json


class RiskAssessmentAgent:
    """Extract risk metrics with validation"""

    def __init__(self):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
        self.validator = FaithfulnessMetric(threshold=0.85, model="gpt-4o-mini")

    def __call__(self, state: StockState) -> StockState:
        """Extract and validate risk data"""
        for ticker in state.query.tickers:
            search_results = state.stocks_data[ticker]["raw_search"]["results"]

            # Find risk-related content
            risk_content = str(search_results[:5])

            prompt = f"""
            Extract risk information for {ticker} from:
            {risk_content}

            Return JSON:
            {{
                "volatility": "high/medium/low",
                "beta": number or null,
                "risk_factors": ["factor1", "factor2"],
                "risk_score": 1-10 (10 = highest risk)
            }}
            """

            response = self.llm.invoke(prompt)
            risk_data = json.loads(response.content)

            # Validate
            test_case = LLMTestCase(
                input=prompt,
                actual_output=response.content,
                context=[risk_content]
            )

            validation = self.validator.measure(test_case)

            if validation.score > 0.85:
                state.stocks_data[ticker]["risk_assessment"] = risk_data
                state.messages.append(f"{ticker} risk validated ({validation.score:.2f})")
            else:
                state.stocks_data[ticker]["risk_assessment"] = {
                    "volatility": "unknown",
                    "beta": None,
                    "risk_factors": ["Data unavailable"],
                    "risk_score": 5
                }

        return state