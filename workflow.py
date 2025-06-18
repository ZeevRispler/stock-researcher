from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
# Updated imports to use the agents package directly
from agents import (
    QueryParserAgent,
    MarketDataAgent,
    NewsSentimentAgent,
    RiskAssessmentAgent,
    SynthesisAgent,
    ValidationAgent,
)


# Define state as TypedDict for LangGraph compatibility
class WorkflowState(TypedDict):
    query: Dict[str, Any]
    stocks_data: Dict[str, Dict[str, Any]]
    messages: List[str]
    error_messages: List[str]
    executive_summary: Optional[str]
    comparison_dashboard: Optional[str]
    validation_result: Optional[Dict[str, Any]]
    needs_retry: bool


class Workflow:
    """Defines the stock research workflow using a state graph."""

    def __init__(self):
        self.graph = StateGraph(WorkflowState)

        # Instantiate all agents
        query_parser = QueryParserAgent()
        market_data = MarketDataAgent()
        news_agent = NewsSentimentAgent()
        risk_agent = RiskAssessmentAgent()
        synthesis_agent = SynthesisAgent()
        validation_agent = ValidationAgent()

        # Add nodes for each agent
        self.graph.add_node("parse_query", query_parser)
        self.graph.add_node("get_market_data", market_data)
        self.graph.add_node("analyze_news", news_agent)
        self.graph.add_node("assess_risk", risk_agent)
        self.graph.add_node("synthesize_report", synthesis_agent)
        self.graph.add_node("validate_report", validation_agent)

        # Define the graph's edges and conditional logic
        self.graph.set_entry_point("parse_query")
        self.graph.add_conditional_edges(
            "parse_query", self.decide_after_parsing
        )
        self.graph.add_edge("get_market_data", "analyze_news")
        self.graph.add_edge("analyze_news", "assess_risk")
        self.graph.add_edge("assess_risk", "synthesize_report")
        self.graph.add_edge("synthesize_report", "validate_report")
        self.graph.add_conditional_edges(
            "validate_report", self.decide_after_validation
        )

        # Compile the graph into a runnable application
        self.app = self.graph.compile()

    def decide_after_parsing(self, state: WorkflowState):
        """Router: End if parsing fails, otherwise continue."""
        if not state.get("query", {}).get("tickers") or state.get("error_messages"):
            return END
        return "get_market_data"

    def decide_after_validation(self, state: WorkflowState):
        """Router: Retry synthesis on failure, otherwise end."""
        val_result = state.get("validation_result", {})
        if state.get("needs_retry") and val_result.get("attempt") == 1:
            return "synthesize_report"
        return END

    def run(self, query: str):
        """Run the workflow with a user query."""
        initial_state = {
            "query": {"user_input": query},
            "stocks_data": {},
            "messages": [],
            "error_messages": [],
            "executive_summary": None,
            "comparison_dashboard": None,
            "validation_result": None,
            "needs_retry": False
        }
        return self.app.invoke(initial_state)