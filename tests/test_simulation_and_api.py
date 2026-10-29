"""
Validation and regression tests for inventory simulation, formulas, and FastAPI endpoints.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.simulation.inventory import (
    InventoryPolicy,
    SimulationResult,
    compute_eoq,
    compute_reorder_point,
    compute_safety_stock,
    newsvendor_order_quantity,
    run_scenario_comparison,
    simulate_inventory,
)


def test_eoq_and_inventory_formulas():
    annual_demand = 36500.0  # 100/day
    eoq = compute_eoq(annual_demand, ordering_cost=50.0, holding_cost_per_unit_per_year=5.0)
    assert eoq > 0
    assert abs(eoq - np.sqrt(2 * 36500 * 50 / 5)) < 1e-4

    ss = compute_safety_stock(avg_demand=100.0, std_demand=15.0, lead_time=3, service_level_z=1.65)
    assert ss > 0
    assert abs(ss - 1.65 * 15.0 * np.sqrt(3)) < 1e-4

    rop = compute_reorder_point(avg_demand=100.0, lead_time=3, safety_stock=ss)
    assert rop == 100.0 * 3 + ss


def test_simulate_inventory_pipeline_tracking():
    # Demand is steady 10 units/day
    demand = np.full(30, 10.0)
    policy = InventoryPolicy(
        reorder_point=40.0,
        order_quantity=50.0,
        lead_time_days=3,
        holding_cost_per_unit=0.5,
        stockout_cost_per_unit=5.0,
        initial_inventory=50.0,
    )

    # Fixed pipeline-aware simulation
    res_fixed = simulate_inventory(demand, policy, track_pipeline=True)
    summary_fixed = res_fixed.summary()

    # Legacy unconstrained simulation (without pipeline tracking)
    res_legacy = simulate_inventory(demand, policy, track_pipeline=False)
    summary_legacy = res_legacy.summary()

    # Pipeline-aware simulation prevents redundant burst reordering and cuts holding costs
    assert summary_fixed["total_holding_cost_$"] < summary_legacy["total_holding_cost_$"]
    assert np.mean(res_fixed.daily_inventory) < np.mean(res_legacy.daily_inventory)
    # Service level should remain high
    assert summary_fixed["service_level_%"] >= 90.0


def test_run_scenario_comparison():
    demand = np.array([10, 12, 14, 8, 11, 15, 9, 13, 10, 12])
    scenarios = {
        "Conservative": InventoryPolicy(50, 100, 2, 0.5, 5.0, 60),
        "Lean": InventoryPolicy(25, 40, 2, 0.5, 5.0, 30),
    }
    df = run_scenario_comparison(demand, scenarios)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "service_level_%" in df.columns
    assert "total_operating_cost_$" in df.columns


def test_api_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "supply-chain-api"}


def test_api_simulate_endpoint():
    client = TestClient(app)
    payload = {
        "demand_forecast": [10.0, 12.0, 15.0, 9.0, 11.0, 14.0],
        "reorder_point": 30.0,
        "order_quantity": 50.0,
        "lead_time_days": 2,
        "holding_cost_per_unit": 0.5,
        "stockout_cost_per_unit": 5.0,
        "initial_inventory": 40.0,
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "service_level_pct" in data
    assert "total_operating_cost" in data
    assert data["service_level_pct"] >= 0.0


def test_api_recommend_policy_endpoint():
    client = TestClient(app)
    payload = {
        "annual_demand": 36500.0,
        "demand_std_daily": 15.0,
        "ordering_cost": 50.0,
        "holding_cost_per_unit_per_year": 5.0,
        "lead_time_days": 3,
        "service_level_z": 1.65,
    }
    resp = client.post("/policy/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "eoq" in data
    assert "safety_stock" in data
    assert "reorder_point" in data
    assert data["eoq"] > 0
