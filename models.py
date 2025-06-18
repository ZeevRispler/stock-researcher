from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class AnalysisType(Enum):
    SINGLE_STOCK = "single_stock"
    COMPARISON = "comparison"


class StockQuery(BaseModel):
    """User query and extracted stock information."""
    user_input: str
    tickers: List[str]
    analysis_type: Optional[AnalysisType] = None


class StockData(BaseModel):
    """Data collected for a single stock."""
    market_data: Optional[str] = None
    sentiment_analysis: Optional[str] = None
    risk_analysis: Optional[str] = None


class ValidationResult(BaseModel):
    """Results from DeepEval validation."""
    is_faithful: bool
    faithfulness_score: float
    is_relevant: bool
    relevancy_score: float


class StockState(BaseModel):
    """Main workflow state containing all data."""
    query: StockQuery
    stocks_data: Dict[str, StockData] = Field(default_factory=dict)
    messages: list = Field(default_factory=list)
    error_messages: list = Field(default_factory=list)
    synthesis_result: Optional[str] = None
    validation_result: Optional[ValidationResult] = None