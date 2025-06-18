# zeevrispler/stock-researcher/stock-researcher-4b3618e2c0950ebe10c63249519e1b2dbb61748a/models.py
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class AnalysisType(Enum):
    SINGLE_STOCK = "single_stock"
    COMPARISON = "comparison"


class StockQuery(BaseModel):
    user_input: str
    tickers: List[str]
    analysis_type: Optional[AnalysisType] = None


class StockData(BaseModel):
    """Holds all the data collected for a single stock."""
    market_data: Optional[str] = None
    sentiment_analysis: Optional[str] = None
    risk_analysis: Optional[str] = None


class ValidationResult(BaseModel):
    is_faithful: bool
    faithfulness_score: float
    is_relevant: bool
    relevancy_score: float


class StockState(BaseModel):
    """Represents the state of our workflow."""
    query: StockQuery
    # The new central data store, replacing the old `stocks` field.
    stocks_data: Dict[str, StockData] = Field(default_factory=dict)

    # Fields for logging and error handling, used by the new ReAct agent
    messages: list = Field(default_factory=list)
    error_messages: list = Field(default_factory=list)

    # Fields for final results
    synthesis_result: Optional[str] = None
    validation_result: Optional[ValidationResult] = None