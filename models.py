from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    """Type of analysis to perform."""

    SINGLE = "single"
    COMPARISON = "comparison"


class StockQuery(BaseModel):
    """Represents the parsed user query."""

    user_input: str = ""
    tickers: List[str] = Field(default_factory=list)
    company_names: List[str] = Field(default_factory=list)
    analysis_type: AnalysisType = AnalysisType.SINGLE


class ValidationResult(BaseModel):
    """Holds the result of a validation step."""

    passed: bool
    scores: Dict[str, float]
    attempt: int


# The main state passed between agents, defined as a Pydantic model
# for structured, attribute-accessible state management.
class StockState(BaseModel):
    """Represents the state of the stock research workflow."""

    query: StockQuery = Field(default_factory=StockQuery)
    stocks_data: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    messages: List[str] = Field(default_factory=list)
    error_messages: List[str] = Field(default_factory=list)
    executive_summary: Optional[str] = None
    comparison_dashboard: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    needs_retry: bool = False

    class Config:
        # Allows the model to work smoothly with LangGraph
        arbitrary_types_allowed = True