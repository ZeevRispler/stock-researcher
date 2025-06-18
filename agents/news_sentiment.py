# zeevrispler/stock-researcher/stock-researcher-4b3618e2c0950ebe10c63249519e1b2dbb61748a/agents/news_sentiment.py
import json

from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_API_BASE
from models import StockState


class SentimentAnalysis(BaseModel):
    sentiment: str = Field(
        ..., description="The overall sentiment (e.g., 'Positive', 'Negative', 'Neutral')."
    )
    confidence_score: float = Field(
        ..., description="A confidence score for the sentiment analysis, between 0 and 1."
    )
    summary: str = Field(
        ..., description="A brief summary of the key news driving the sentiment."
    )


class NewsSentimentAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            temperature=0,
        ).with_structured_output(SentimentAnalysis)
        self.validator = SummarizationMetric(threshold=0.8, model="gpt-4o")

    def __call__(self, state: StockState):
        for ticker, stock_data in state.stocks_data.items():
            source_text = stock_data.market_data or "No data available."
            if source_text == "No data available.":
                stock_data.sentiment_analysis = "No market data to analyze for sentiment."
                continue

            prompt = f"""
            Analyze the sentiment of the following market data and news summary for {ticker}.
            Focus specifically on the news summaries within the text to determine if the overall sentiment is Positive, Negative, or Neutral.
            Provide a confidence score (0-1) and a brief summary of the key points driving the sentiment.

            Collected Data:
            {source_text}
            """
            response: SentimentAnalysis = self.llm.invoke(prompt)

            test_case = LLMTestCase(
                input=prompt,
                actual_output=response.summary,
                retrieval_context=[source_text]
            )
            self.validator.measure(test_case)

            if self.validator.is_successful():
                stock_data.sentiment_analysis = json.dumps(response.dict(), indent=2)
            else:
                stock_data.sentiment_analysis = "Failed to generate a faithful sentiment analysis."
        return state