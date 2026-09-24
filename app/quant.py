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


def optimize_allocation(
    historical_returns: dict[str, list[float]], objective: str, risk_free_rate: float
) -> dict:
    """Optimize weights using PyPortfolioOpt and Ledoit-Wolf covariance shrinkage."""
    from pypfopt import EfficientFrontier
    from sklearn.covariance import LedoitWolf

    frame = pd.DataFrame(historical_returns, dtype=float).dropna()
    if frame.shape[1] < 2 or frame.shape[0] < 2:
        raise ValueError("At least two assets and two aligned observations are required")
    expected_returns = frame.mean() * 252
    covariance = pd.DataFrame(
        LedoitWolf().fit(frame.to_numpy()).covariance_ * 252,
        index=frame.columns,
        columns=frame.columns,
    )
    frontier = EfficientFrontier(expected_returns, covariance)
    if objective == "max_sharpe":
        frontier.max_sharpe(risk_free_rate=risk_free_rate)
    elif objective == "min_volatility":
        frontier.min_volatility()
    else:
        raise ValueError("objective must be max_sharpe or min_volatility")
    weights = frontier.clean_weights()
    performance = frontier.portfolio_performance(risk_free_rate=risk_free_rate)
    return {
        "objective": objective,
        "weights": weights,
        "expected_return": round(float(performance[0]), 8),
        "volatility": round(float(performance[1]), 8),
        "sharpe_ratio": round(float(performance[2]), 8),
        "covariance_estimator": "LedoitWolf",
    }


def risk_analysis(holdings: list[dict]) -> dict:
    """Analyze concentration, correlations, and volatility contribution."""
    values = {_asset_class(p): 0.0 for p in holdings}
    for position in holdings:
        values[_asset_class(position)] += _value(position)
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Current holdings must have positive market value")
    weights = {key: value / total for key, value in values.items()}
    concentration = sorted(
        ({"asset_class": key, "weight": round(value, 8)} for key, value in weights.items()),
        key=lambda item: item["weight"],
        reverse=True,
    )
    returns = {
        key: series
        for key, series in ((_asset_class(p), p.get("returns")) for p in holdings)
        if series
    }
    result = {
        "portfolio_value": round(total, 2),
        "asset_class_weights": concentration,
        "largest_exposure": concentration[0] if concentration else None,
        "concentration_flags": [
            item["asset_class"] for item in concentration if item["weight"] >= 0.5
        ],
        "correlation_matrix": {},
        "volatility_contribution": {},
    }
    if len(returns) >= 2:
        frame = pd.DataFrame(returns).dropna()
        result["correlation_matrix"] = frame.corr().round(8).to_dict()
        covariance = frame.cov().to_numpy() * 252
        weight_vector = np.array([weights.get(column, 0.0) for column in frame.columns])
        portfolio_vol = float(np.sqrt(weight_vector @ covariance @ weight_vector))
        if portfolio_vol:
            contributions = weight_vector * (covariance @ weight_vector) / portfolio_vol
            result["volatility_contribution"] = {
                column: round(float(value), 8)
                for column, value in zip(frame.columns, contributions)
            }
    return result
