from langchain.chat_models import ChatOpenAI
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from models import StockState
from config import OPENAI_API_KEY
import json


class NewsSentimentAgent:
    """Extract news with validation"""

    def __init__(self):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
        self.validator = FaithfulnessMetric(threshold=0.85, model="gpt-4o-mini")

    def __call__(self, state: StockState) -> StockState:
        """Extract and validate news sentiment"""
        for ticker in state.query.tickers:
            search_results = state.stocks_data[ticker]["raw_search"]["results"]

            # Filter news content
            news_content = [r for r in search_results if any(
                keyword in r.get("content", "").lower()
                for keyword in ["news", "announced", "report", "earnings", "update", "stock", "share"]
            )][:4]  # Limit to 4 for context window

            # Extract sentiment
            prompt = f"""
            Analyze news sentiment for {ticker} from these articles:
            {json.dumps(news_content, indent=2)}

            Return JSON:
            {{
                "sentiment": -1.0 to 1.0,
                "events": ["key event 1", "key event 2", "key event 3"],
                "summary": "one sentence summary"
            }}
            """

            response = self.llm.invoke(prompt)
            analysis = json.loads(response.content)

            # Validate extraction
            test_case = LLMTestCase(
                input=prompt,
                actual_output=response.content,
                context=[str(news_content)]
            )

            validation = self.validator.measure(test_case)

            if validation.score > 0.85:
                state.stocks_data[ticker]["news_analysis"] = analysis
                state.messages.append(f"{ticker} news validated ({validation.score:.2f})")
            else:
                state.stocks_data[ticker]["news_analysis"] = {
                    "sentiment": 0,
                    "events": ["Unable to extract verified events"],
                    "summary": "Insufficient data"
                }
                state.messages.append(f"{ticker} news validation failed")

        return state