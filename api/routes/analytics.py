"""Analytics and reporting endpoints for ChequeSense API."""

from __future__ import annotations

import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas import AnalyticsTrendsResponse, ExecutiveSummary
from src.analytics.metrics import calculate_executive_dashboard
from src.analytics.trends import calculate_amount_trend, calculate_volume_trend

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "/summary",
    response_model=ExecutiveSummary,
    status_code=status.HTTP_200_OK,
    summary="Get consolidated executive analytics summary",
    description="Returns aggregate banking metrics including volume, clearing amounts, manual review rates, confidence distribution, field recognition, and pipeline latency.",
)
def get_analytics_summary(
    start_date: Optional[datetime.datetime] = Query(None, description="Start of reporting window (ISO format)"),
    end_date: Optional[datetime.datetime] = Query(None, description="End of reporting window (ISO format)"),
    db: Session = Depends(get_db),
) -> ExecutiveSummary:
    return calculate_executive_dashboard(
        session=db,
        start_date=start_date,
        end_date=end_date,
    )


@router.get(
    "/trends",
    response_model=AnalyticsTrendsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get processing volume and monetary clearing trends",
    description="Returns time-series buckets (day, hour, or week) tracking volume, review rates, and recognized cheque values.",
)
def get_analytics_trends(
    interval: str = Query("day", pattern="^(day|hour|week)$", description="Aggregation bucket interval: 'day', 'hour', or 'week'"),
    start_date: Optional[datetime.datetime] = Query(None, description="Start timestamp (ISO format)"),
    end_date: Optional[datetime.datetime] = Query(None, description="End timestamp (ISO format)"),
    db: Session = Depends(get_db),
) -> AnalyticsTrendsResponse:
    vol_trends = calculate_volume_trend(
        session=db,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )
    amt_trends = calculate_amount_trend(
        session=db,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )
    return AnalyticsTrendsResponse(
        interval=interval,
        volume_trends=vol_trends,
        amount_trends=amt_trends,
    )
