from tavily import TavilyClient
from models import StockState
from config import TAVILY_API_KEY


class MarketDataAgent:
    """Fetch data - no validation needed for raw search"""

    def __init__(self):
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)

    def __call__(self, state: StockState) -> StockState:
        """One comprehensive search per stock"""
        for ticker in state.query.tickers:
            results = self.tavily.search(
                f"{ticker} stock price market cap PE ratio news sentiment volatility beta financial analysis",
                search_depth="advanced",
                max_results=10
            )

            state.stocks_data[ticker] = {
                "raw_search": results,
                "ticker": ticker
            }

        state.messages.append(f"Completed search for {len(state.query.tickers)} stocks")
        return state