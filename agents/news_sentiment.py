from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_API_BASE
from models import StockState


class NewsAnalysis(BaseModel):
    """
    The result of news analysis for a stock.
    - sentiment: The sentiment of the news, can be "positive", "negative" or "neutral".
    - notable_events: A list of notable events from the news.
    - summary: A summary of the news.
    """

    sentiment: str = Field(
        ...,
        description='The sentiment of the news, can be "positive", "negative" or "neutral".',
    )
    notable_events: list[str] = Field(
        ..., description="A list of notable events from the news."
    )
    summary: str = Field(..., description="A summary of the news.")


class NewsSentimentAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
        ).with_structured_output(NewsAnalysis)

    def __call__(self, state: StockState) -> StockState:
        state.messages.append("Analyzing news sentiment...")

        for ticker in state.query.tickers:
            # --- THIS IS THE UPDATED SECTION ---
            # Instead of parsing raw search results, we now use the clean,
            # synthesized text block produced by the ReAct agent.
            context = state.stocks_data[ticker].get("market_data", "")
            if not context:
                state.error_messages.append(
                    f"No market data context found for {ticker} to analyze news."
                )
                continue
            # --- END OF UPDATE ---

            prompt = f"""
            Analyze the news sentiment for the stock with the ticker {ticker}.
            Here is the context:
            ---
            {context}
            ---
            Respond with a JSON object with the following schema:
            {{
                "sentiment": "positive" | "negative" | "neutral",
                "notable_events": ["event_1", "event_2", ...],
                "summary": "A summary of the news."
            }}
            """
            news_analysis: NewsAnalysis = self.llm.invoke(prompt)

            metric = SummarizationMetric(
                threshold=0.7,
                model="gpt-4o-mini",
                assessment_questions=[
                    "Is the summary faithful to the context?",
                    "Does the summary mention the stock ticker?",
                    "Does the summary include all the notable events?",
                ],
            )
            test_case = LLMTestCase(
                input=prompt,
                actual_output=news_analysis.summary,
                context=[context],
            )
            metric.measure(test_case)
            print(f"Summarization metric score: {metric.score}")
            if metric.score < 0.7:
                state.error_messages.append(
                    f"Summarization metric score for {ticker} is below threshold."
                )
                news_analysis.summary = (
                    "The summary was not faithful to the context."
                )

            if ticker not in state.stocks_data:
                state.stocks_data[ticker] = {}
            state.stocks_data[ticker]["news_analysis"] = news_analysis.dict()

        return state