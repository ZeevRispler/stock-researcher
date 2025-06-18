# zeevrispler/stock-researcher/stock-researcher-4b3618e2c0950ebe10c63249519e1b2dbb61748a/workflow.py
from langgraph.graph import StateGraph, END

from models import StockState, StockQuery
from agents import (
    QueryParserAgent,
    MarketDataAgent,
    NewsSentimentAgent,
    RiskAssessmentAgent,
    SynthesisAgent,
    validate_results,
)


class Workflow:
    def __init__(self):
        # Initialize agents
        query_parser = QueryParserAgent()
        market_data_agent = MarketDataAgent()
        news_agent = NewsSentimentAgent()
        risk_agent = RiskAssessmentAgent()
        synthesis_agent = SynthesisAgent()

        # Define the graph state
        workflow = StateGraph(StockState)

        # Add nodes
        workflow.add_node("query_parser", query_parser)
        workflow.add_node("market_data", market_data_agent)
        workflow.add_node("news_sentiment", news_agent)
        workflow.add_node("risk_assessment", risk_agent)
        workflow.add_node("synthesis", synthesis_agent)
        workflow.add_node("validation", validate_results)

        # Define the edges
        workflow.set_entry_point("query_parser")
        workflow.add_edge("query_parser", "market_data")
        workflow.add_edge("market_data", "news_sentiment")
        workflow.add_edge("news_sentiment", "risk_assessment")
        workflow.add_edge("risk_assessment", "synthesis")
        workflow.add_edge("synthesis", "validation")
        workflow.add_edge("validation", END)

        self.app = workflow.compile()

    def run(self, query: str):
        """
        Run the research process for a given query.
        """
        # Initialize the state with the user's query
        initial_state = StockState(
            query=StockQuery(user_input=query, tickers=[])
        )
        # Invoke the graph
        return self.app.invoke(initial_state)