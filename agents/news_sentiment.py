from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_API_BASE


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
        # Only pass base_url if it's not the default
        kwargs = {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY,
        }
        if OPENAI_API_BASE and OPENAI_API_BASE != "https://api.openai.com/v1":
            kwargs["base_url"] = OPENAI_API_BASE

        self.llm = ChatOpenAI(**kwargs).with_structured_output(NewsAnalysis)
        self.validator = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")

    def __call__(self, state: dict) -> dict:
        state["messages"].append("Analyzing news sentiment...")

        for ticker in state["query"]["tickers"]:
            # Instead of parsing raw search results, we now use the clean,
            # synthesized text block produced by the ReAct agent.
            context = state["stocks_data"][ticker].get("market_data", "")
            if not context:
                state["error_messages"].append(
                    f"No market data context found for {ticker} to analyze news."
                )
                continue

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

            try:
                news_analysis: NewsAnalysis = self.llm.invoke(prompt)

                # Use faithfulness metric to check if the analysis is based on the context
                test_case = LLMTestCase(
                    input=prompt,
                    actual_output=str(news_analysis.model_dump()),
                    retrieval_context=[context]
                )

                self.validator.measure(test_case)
                score = self.validator.score
                print(f"News analysis faithfulness score for {ticker}: {score}")

                if score < 0.7:
                    state["messages"].append(
                        f"-> {ticker} news analysis may not be fully faithful to source ({score})"
                    )
                else:
                    state["messages"].append(
                        f"-> {ticker} news analysis validated ({score})"
                    )

                if ticker not in state["stocks_data"]:
                    state["stocks_data"][ticker] = {}
                state["stocks_data"][ticker]["news_analysis"] = news_analysis.model_dump()

            except Exception as e:
                state["error_messages"].append(f"Error analyzing news for {ticker}: {str(e)}")
                # Add default values so workflow can continue
                if ticker not in state["stocks_data"]:
                    state["stocks_data"][ticker] = {}
                state["stocks_data"][ticker]["news_analysis"] = {
                    "sentiment": "neutral",
                    "notable_events": ["Analysis failed"],
                    "summary": "Failed to analyze news sentiment"
                }

        return state