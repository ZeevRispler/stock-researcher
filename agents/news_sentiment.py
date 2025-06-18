from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI
from models import NewsAnalysis
from config import OPENAI_API_KEY, OPENAI_API_BASE


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

    def _should_validate(self, analysis: NewsAnalysis) -> bool:
        """Determine if deep validation is needed based on confidence."""
        # Use DeepEval only when confidence is low or data quality is poor
        if analysis.confidence_score < 0.7:
            return True
        if analysis.data_quality == "low":
            return True
        # Always validate if we found no notable events (suspicious)
        if not analysis.notable_events or len(analysis.notable_events) == 0:
            return True
        return False

    def __call__(self, state: dict) -> dict:
        print("Running NewsSentimentAgent...")
        state["messages"].append("Analyzing news sentiment...")

        for ticker in state["query"]["tickers"]:
            context = state["stocks_data"][ticker].get("market_data", "")
            if not context:
                state["error_messages"].append(
                    f"No market data context found for {ticker} to analyze news."
                )
                continue

            prompt = f"""
            Analyze the news sentiment for stock {ticker}.
            Context:
            ---
            {context}
            ---

            Provide confidence_score (0-1) based on:
            - Amount of relevant news found
            - Clarity of sentiment signals
            - Recency of information

            Set data_quality based on:
            - "high": Recent, detailed financial news
            - "medium": Some relevant information 
            - "low": Limited or unclear sources

            Respond with JSON matching the NewsAnalysis schema.
            """

            try:
                news_analysis: NewsAnalysis = self.llm.invoke(prompt)

                # Only run expensive validation if confidence/quality is low
                if self._should_validate(news_analysis):
                    state["messages"].append(f"-> {ticker} low confidence, running validation...")

                    test_case = LLMTestCase(
                        input=prompt,
                        actual_output=str(news_analysis.model_dump()),
                        retrieval_context=[context]
                    )

                    self.validator.measure(test_case)
                    score = self.validator.score

                    if score < 0.7:
                        state["messages"].append(
                            f"-> {ticker} news analysis failed validation ({score})"
                        )
                    else:
                        state["messages"].append(
                            f"-> {ticker} news analysis validated ({score})"
                        )
                else:
                    state["messages"].append(
                        f"-> {ticker} high confidence ({news_analysis.confidence_score}), skipping validation"
                    )

                if ticker not in state["stocks_data"]:
                    state["stocks_data"][ticker] = {}
                state["stocks_data"][ticker]["news_analysis"] = news_analysis

            except Exception as e:
                state["error_messages"].append(f"Error analyzing news for {ticker}: {str(e)}")
                # Add default values so workflow can continue
                if ticker not in state["stocks_data"]:
                    state["stocks_data"][ticker] = {}
                default_analysis = NewsAnalysis(
                    sentiment="neutral",
                    notable_events=["Analysis failed"],
                    summary="Failed to analyze news sentiment",
                    confidence_score=0.0,
                    data_quality="low"
                )
                state["stocks_data"][ticker]["news_analysis"] = default_analysis

        return state