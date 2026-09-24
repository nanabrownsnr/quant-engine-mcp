"""Pure, deterministic portfolio calculations used by the MCP tools."""

from collections import defaultdict
from math import isfinite, sqrt

import numpy as np
import pandas as pd


def _value(position: dict) -> float:
    if "market_value" in position:
        value = float(position["market_value"])
    else:
        value = float(position.get("quantity", 0)) * float(position.get("current_price", 0))
    if not isfinite(value) or value < 0:
        raise ValueError("Each holding must have a finite, non-negative market value")
    return value


def _asset_class(position: dict) -> str:
    return str(position.get("asset_class") or position.get("security_type") or "Unknown")


def portfolio_value(holdings: list[dict]) -> float:
    return sum(_value(position) for position in holdings)


def calculate_drift(holdings: list[dict], target: dict[str, float]) -> dict:
    total = portfolio_value(holdings)
    if total <= 0:
        raise ValueError("Current holdings must have positive market value")
    target_total = sum(float(value) for value in target.values())
    if not np.isclose(target_total, 1.0, atol=1e-6):
        raise ValueError("target_allocation must sum to 1.0")

    values: dict[str, float] = defaultdict(float)
    for position in holdings:
        values[_asset_class(position)] += _value(position)
    classes = sorted(set(values) | set(target))
    rows = []
    for asset_class in classes:
        current_value = values[asset_class]
        target_pct = float(target.get(asset_class, 0.0))
        current_pct = current_value / total
        rows.append(
            {
                "asset_class": asset_class,
                "current_value": round(current_value, 2),
                "current_percentage": round(current_pct, 8),
                "target_percentage": round(target_pct, 8),
                "drift_percentage": round(current_pct - target_pct, 8),
                "drift_value": round(current_value - total * target_pct, 2),
            }
        )
    return {"portfolio_value": round(total, 2), "drift": rows}


def rebalance(holdings: list[dict], target: dict[str, float], max_tax_impact: bool) -> dict:
    drift = calculate_drift(holdings, target)
    total = drift["portfolio_value"]
    plans = []
    for row in drift["drift"]:
        amount = -row["drift_value"]
        if abs(amount) < 0.01:
            continue
        direction = "buy" if amount > 0 else "sell"
        affected = [p for p in holdings if _asset_class(p) == row["asset_class"]]
        taxable = any(
            str(p.get("account_type", "")).lower() in {"taxable", "brokerage"} for p in affected
        )
        estimated_gain = 0.0
        if direction == "sell":
            for p in affected:
                mv = _value(p)
                basis = float(p.get("cost_basis", 0))
                estimated_gain += max(0.0, mv - basis) * min(
                    abs(amount) / max(sum(_value(x) for x in affected), 1), 1
                )
        plans.append(
            {
                "asset_class": row["asset_class"],
                "action": direction,
                "dollar_amount": round(abs(amount), 2),
                "taxable_account_affected": taxable,
                "estimated_taxable_gain": round(estimated_gain, 2),
                "tax_impact_flag": bool(max_tax_impact and taxable and estimated_gain > 0),
            }
        )
    return {"portfolio_value": total, "target_allocation": target, "trades": plans}


def metrics(holdings: list[dict]) -> dict:
    values = [_value(p) for p in holdings]
    total = sum(values)
    if total <= 0:
        raise ValueError("Current holdings must have positive market value")
    returns_by_asset = {}
    for position in holdings:
        series = position.get("returns")
        if series:
            returns_by_asset.setdefault(_asset_class(position), []).append(series)
    result = {
        "portfolio_value": round(total, 2),
        "portfolio_volatility": None,
        "sharpe_ratio": None,
        "correlation_matrix": {},
    }
    if not returns_by_asset:
        return result
    asset_returns = {}
    for asset_class, series_list in returns_by_asset.items():
        frame = pd.DataFrame(series_list).astype(float)
        asset_returns[asset_class] = frame.mean(axis=0)
    frame = pd.DataFrame(asset_returns).dropna()
    weights = np.array(
        [sum(_value(p) for p in holdings if _asset_class(p) == c) / total for c in frame.columns]
    )
    portfolio_returns = frame.to_numpy() @ weights
    volatility = float(np.std(portfolio_returns, ddof=1)) if len(portfolio_returns) > 1 else 0.0
    result["portfolio_volatility"] = round(volatility * sqrt(252), 8)
    result["sharpe_ratio"] = (
        round(float(np.mean(portfolio_returns) / volatility * sqrt(252)), 8) if volatility else None
    )
    result["correlation_matrix"] = frame.corr().round(8).to_dict()
    return result
