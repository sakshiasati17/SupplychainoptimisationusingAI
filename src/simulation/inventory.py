"""
Inventory simulation and reorder decision engine.

Given demand forecasts, simulates:
- Reorder point (ROP) decisions
- Stockout events
- Overstock events
- Holding cost
- Lost sales cost
- Total operating cost

Business metrics:
- Stockout rate (%)
- Overstock rate (%)
- Service level (%)
- Total cost per period
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class InventoryPolicy:
    reorder_point: float          # Trigger reorder when inventory <= ROP
    order_quantity: float         # How much to order (EOQ or fixed)
    lead_time_days: int = 2       # Days until order arrives
    holding_cost_per_unit: float = 0.5   # $ per unit per day
    stockout_cost_per_unit: float = 5.0  # $ per unit of lost sales
    initial_inventory: float = 100.0


@dataclass
class SimulationResult:
    daily_inventory: List[float] = field(default_factory=list)
    daily_demand: List[float] = field(default_factory=list)
    stockout_days: int = 0
    overstock_days: int = 0
    total_holding_cost: float = 0.0
    total_lost_sales_cost: float = 0.0
    total_units_ordered: float = 0.0
    n_orders_placed: int = 0

    @property
    def total_days(self):
        return len(self.daily_inventory)

    @property
    def stockout_rate(self):
        return self.stockout_days / max(self.total_days, 1) * 100

    @property
    def overstock_rate(self):
        return self.overstock_days / max(self.total_days, 1) * 100

    @property
    def service_level(self):
        return 100 - self.stockout_rate

    @property
    def total_operating_cost(self):
        return self.total_holding_cost + self.total_lost_sales_cost

    def summary(self) -> dict:
        return {
            "stockout_rate_%": round(self.stockout_rate, 2),
            "overstock_rate_%": round(self.overstock_rate, 2),
            "service_level_%": round(self.service_level, 2),
            "total_holding_cost_$": round(self.total_holding_cost, 2),
            "total_lost_sales_cost_$": round(self.total_lost_sales_cost, 2),
            "total_operating_cost_$": round(self.total_operating_cost, 2),
            "total_units_ordered": round(self.total_units_ordered, 2),
            "n_orders_placed": self.n_orders_placed,
        }


def simulate_inventory(
    demand_forecast: np.ndarray,
    policy: InventoryPolicy,
) -> SimulationResult:
    """
    Simulate day-by-day inventory under a given reorder policy.

    Args:
        demand_forecast: Array of daily demand values (units).
        policy: InventoryPolicy with thresholds and cost parameters.

    Returns:
        SimulationResult with all business metrics.
    """
    result = SimulationResult()
    inventory = policy.initial_inventory
    pending_orders = []  # list of (arrival_day, qty)

    for day, demand in enumerate(demand_forecast):
        # Receive pending orders
        arrived = [qty for (arr_day, qty) in pending_orders if arr_day == day]
        for qty in arrived:
            inventory += qty
        pending_orders = [(d, q) for (d, q) in pending_orders if d != day]

        # Fulfill demand
        units_sold = min(demand, inventory)
        lost_sales = max(0, demand - inventory)
        inventory -= units_sold

        # Track stockout / overstock
        if inventory <= 0:
            result.stockout_days += 1
        if inventory > 2 * demand + 1:
            result.overstock_days += 1

        # Costs
        result.total_holding_cost += inventory * policy.holding_cost_per_unit
        result.total_lost_sales_cost += lost_sales * policy.stockout_cost_per_unit

        # Reorder decision
        if inventory <= policy.reorder_point:
            arrival_day = day + policy.lead_time_days
            pending_orders.append((arrival_day, policy.order_quantity))
            result.total_units_ordered += policy.order_quantity
            result.n_orders_placed += 1

        result.daily_inventory.append(inventory)
        result.daily_demand.append(demand)

    return result


def compute_eoq(
    annual_demand: float,
    ordering_cost: float = 50.0,
    holding_cost_per_unit_per_year: float = 5.0,
) -> float:
    """Economic Order Quantity formula."""
    return np.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit_per_year)


def compute_safety_stock(
    avg_demand: float,
    std_demand: float,
    lead_time: int = 2,
    service_level_z: float = 1.65,  # 95% service level
) -> float:
    return service_level_z * std_demand * np.sqrt(lead_time)


def compute_reorder_point(
    avg_demand: float,
    lead_time: int = 2,
    safety_stock: float = 0.0,
) -> float:
    return avg_demand * lead_time + safety_stock


def run_scenario_comparison(
    demand_forecast: np.ndarray,
    scenarios: dict,
) -> pd.DataFrame:
    """
    Run multiple policy scenarios and return a comparison DataFrame.

    Args:
        demand_forecast: Array of daily demand values.
        scenarios: dict of {scenario_name: InventoryPolicy}

    Returns:
        DataFrame with one row per scenario showing business metrics.
    """
    rows = []
    for name, policy in scenarios.items():
        result = simulate_inventory(demand_forecast, policy)
        row = {"scenario": name}
        row.update(result.summary())
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")
