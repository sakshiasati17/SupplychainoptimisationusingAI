"""
FastAPI application for demand forecasting and inventory simulation.

Endpoints:
  POST /forecast      - Generate demand forecast for a store-product
  POST /simulate      - Run inventory simulation with custom policy
  GET  /health        - Health check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from src.simulation.inventory import (
    InventoryPolicy,
    simulate_inventory,
    compute_eoq,
    compute_safety_stock,
    compute_reorder_point,
)

app = FastAPI(
    title="Supply Chain Demand Forecasting API",
    description="Demand forecasting and inventory simulation for retail supply chains.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    store_id: str = Field(..., example="S001")
    product_id: str = Field(..., example="P0001")
    horizon_days: int = Field(30, ge=1, le=365)
    model: str = Field("xgboost", description="naive | moving_average | xgboost | prophet")


class ForecastResponse(BaseModel):
    store_id: str
    product_id: str
    model: str
    forecast: List[float]
    dates: List[str]


class SimulationRequest(BaseModel):
    demand_forecast: List[float] = Field(..., min_items=1)
    reorder_point: float = Field(50.0, ge=0)
    order_quantity: float = Field(100.0, gt=0)
    lead_time_days: int = Field(2, ge=0)
    holding_cost_per_unit: float = Field(0.5, gt=0)
    stockout_cost_per_unit: float = Field(5.0, gt=0)
    initial_inventory: float = Field(100.0, ge=0)


class SimulationResponse(BaseModel):
    stockout_rate_pct: float
    overstock_rate_pct: float
    service_level_pct: float
    total_holding_cost: float
    total_lost_sales_cost: float
    total_operating_cost: float
    total_units_ordered: float
    n_orders_placed: int


class PolicyRecommendationRequest(BaseModel):
    annual_demand: float
    demand_std_daily: float
    ordering_cost: float = 50.0
    holding_cost_per_unit_per_year: float = 5.0
    lead_time_days: int = 2
    service_level_z: float = 1.65


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "supply-chain-api"}


@app.post("/simulate", response_model=SimulationResponse)
def run_simulation(req: SimulationRequest):
    policy = InventoryPolicy(
        reorder_point=req.reorder_point,
        order_quantity=req.order_quantity,
        lead_time_days=req.lead_time_days,
        holding_cost_per_unit=req.holding_cost_per_unit,
        stockout_cost_per_unit=req.stockout_cost_per_unit,
        initial_inventory=req.initial_inventory,
    )
    demand = np.array(req.demand_forecast)
    result = simulate_inventory(demand, policy)
    summary = result.summary()
    return SimulationResponse(
        stockout_rate_pct=summary["stockout_rate_%"],
        overstock_rate_pct=summary["overstock_rate_%"],
        service_level_pct=summary["service_level_%"],
        total_holding_cost=summary["total_holding_cost_$"],
        total_lost_sales_cost=summary["total_lost_sales_cost_$"],
        total_operating_cost=summary["total_operating_cost_$"],
        total_units_ordered=summary["total_units_ordered"],
        n_orders_placed=summary["n_orders_placed"],
    )


@app.post("/policy/recommend")
def recommend_policy(req: PolicyRecommendationRequest):
    eoq = compute_eoq(
        req.annual_demand, req.ordering_cost, req.holding_cost_per_unit_per_year
    )
    avg_daily_demand = req.annual_demand / 365
    safety_stock = compute_safety_stock(
        avg_daily_demand, req.demand_std_daily, req.lead_time_days, req.service_level_z
    )
    rop = compute_reorder_point(avg_daily_demand, req.lead_time_days, safety_stock)
    return {
        "eoq": round(eoq, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(rop, 2),
        "avg_daily_demand": round(avg_daily_demand, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
