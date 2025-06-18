# zeevrispler/stock-researcher/stock-researcher-4b3618e2c0950ebe10c63249519e1b2dbb61748a/agents/query_parser.py
import json
import re

from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY
from models import StockState, StockQuery, AnalysisType


class QueryParserAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=0,
        )

    def __call__(self, state: StockState):
        query_text = state["query"].user_input
        prompt = f"""
        You are a financial expert. Your task is to analyze the user's query and extract the necessary information to proceed with the stock analysis.

        The user's query is: "{query_text}"

        Please extract the following information:
        1.  **Stock Tickers**: Identify the official stock ticker for each company mentioned. If a full company name is given (e.g., "Apple"), convert it to its ticker symbol (e.g., "AAPL"). The output should be a list of tickers.
        2.  **Analysis Type**: Determine if the user is asking for an analysis of a single stock or a comparison between multiple stocks. The output should be either 'single_stock' or 'comparison'.

        Return the information in a JSON object with the keys 'tickers' and 'analysis_type'.
        """
        response = self.llm.invoke(prompt)
        parsed_response = json.loads(response.content)

        state["query"] = StockQuery(
            user_input=query_text,
            tickers=parsed_response["tickers"],
            analysis_type=AnalysisType(parsed_response["analysis_type"]),
        )
        return state