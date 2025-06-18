from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_API_BASE


class NewsAnalysis(BaseModel):
    sentiment: str = Field(description='The sentiment of the news, can be "positive", "negative" or "neutral".')
    notable_events: list[str] = Field(description="A list of notable events from the news.")
    summary: str = Field(description="A summary of the news.")
    confidence_score: float = Field(description="Confidence 0-1 in this analysis")
    data_quality: str = Field(description="'high', 'medium', 'low' based on source data")


class NewsSentimentAgent:
    def __init__(self):
        kwargs = {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY,
        }
        if OPENAI_API_BASE and OPENAI_API_BASE != "https://api.openai.com/v1":
            kwargs["base_url"] = OPENAI_API_BASE

        self.llm = ChatOpenAI(**kwargs).with_structured_output(NewsAnalysis)
        self.validator = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")

    def __call__(self, state: dict) -> dict:
        print("📰 Analyzing news sentiment...")

        for ticker in state["query"]["tickers"]:
            context = state["stocks_data"][ticker].get("market_data", "")
            if not context:
                state["error_messages"].append(f"No market data context found for {ticker} to analyze news.")
                continue

            prompt = f"""
            Analyze the news sentiment for the stock with the ticker {ticker}.

            Provide confidence_score (0-1) based on:
            - Amount of relevant news found
            - Clarity of sentiment signals
            - Recency of information

            Set data_quality based on:
            - "high": Recent, detailed financial news
            - "medium": Some relevant information 
            - "low": Limited or unclear sources

            Context:
            ---
            {context}
            ---
            Respond with JSON matching the NewsAnalysis schema.
            """

            try:
                news_analysis: NewsAnalysis = self.llm.invoke(prompt)

                # Only run expensive validation if confidence/quality is low
                if news_analysis.confidence_score < 0.7 or news_analysis.data_quality == "low":
                    test_case = LLMTestCase(
                        input=prompt,
                        actual_output=str(news_analysis.model_dump()),
                        retrieval_context=[context]
                    )
                    self.validator.measure(test_case)
                    score = self.validator.score
                    if score < 0.7:
                        print(f"News analysis may not be fully faithful to source ({score})")

                if ticker not in state["stocks_data"]:
                    state["stocks_data"][ticker] = {}
                state["stocks_data"][ticker]["news_analysis"] = news_analysis.model_dump()

            except Exception as e:
                state["error_messages"].append(f"Error analyzing news for {ticker}: {str(e)}")
                if ticker not in state["stocks_data"]:
                    state["stocks_data"][ticker] = {}
                state["stocks_data"][ticker]["news_analysis"] = {
                    "sentiment": "neutral",
                    "notable_events": ["Analysis failed"],
                    "summary": "Failed to analyze news sentiment"
                }

        return state