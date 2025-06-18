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


class NewsAnalysis(BaseModel):
    """News analysis with confidence scoring."""
    sentiment: str = Field(description='The sentiment: "positive", "negative" or "neutral"')
    notable_events: list[str] = Field(description="Notable events from the news")
    summary: str = Field(description="Summary of the news")
    confidence_score: float = Field(description="Confidence 0-1 in this analysis", ge=0, le=1)
    data_quality: str = Field(description="'high', 'medium', 'low' based on source data")


class RiskData(BaseModel):
    """Risk assessment with confidence scoring."""
    volatility: str = Field(description='Volatility: "high", "medium" or "low"')
    beta: float | None = Field(description="The beta of the stock")
    risk_factors: list[str] = Field(description="List of risk factors")
    risk_score: int = Field(description="Score from 1-10 (10 = highest risk)", ge=1, le=10)
    confidence_score: float = Field(description="Confidence 0-1 in this assessment", ge=0, le=1)
    data_completeness: str = Field(description="'complete', 'partial', 'limited'")


class StockData(BaseModel):
    """Holds all the data collected for a single stock."""
    market_data: Optional[str] = None
    news_analysis: Optional[NewsAnalysis] = None
    risk_assessment: Optional[RiskData] = None


class ValidationResult(BaseModel):
    is_faithful: bool
    faithfulness_score: float
    is_relevant: bool
    relevancy_score: float


class StockState(BaseModel):
    """Represents the state of our workflow."""
    query: StockQuery
    stocks_data: Dict[str, StockData] = Field(default_factory=dict)
    messages: list = Field(default_factory=list)
    error_messages: list = Field(default_factory=list)
    synthesis_result: Optional[str] = None
    validation_result: Optional[ValidationResult] = None