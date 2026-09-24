import pytest

from app.quant import calculate_drift, metrics, rebalance

HOLDINGS = [
    {
        "ticker": "VTI",
        "asset_class": "Equity",
        "quantity": 6,
        "current_price": 100,
        "account_type": "Taxable",
        "cost_basis": 500,
    },
    {
        "ticker": "BND",
        "asset_class": "Bond",
        "quantity": 4,
        "current_price": 100,
        "account_type": "Roth IRA",
        "cost_basis": 400,
    },
]


def test_drift_is_calculated_by_value():
    result = calculate_drift(HOLDINGS, {"Equity": 0.5, "Bond": 0.5})
    assert result["portfolio_value"] == 1000
    equity = next(item for item in result["drift"] if item["asset_class"] == "Equity")
    assert equity["drift_value"] == 100


def test_rebalance_flags_taxable_gain():
    result = rebalance(HOLDINGS, {"Equity": 0.5, "Bond": 0.5}, True)
    equity = next(item for item in result["trades"] if item["asset_class"] == "Equity")
    assert equity["action"] == "sell"
    assert equity["tax_impact_flag"] is True


def test_target_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        calculate_drift(HOLDINGS, {"Equity": 0.6})


def test_metrics_use_supplied_returns():
    holdings = [
        {**HOLDINGS[0], "returns": [0.01, -0.005, 0.02]},
        {**HOLDINGS[1], "returns": [0.0, 0.002, -0.001]},
    ]
    result = metrics(holdings)
    assert result["portfolio_volatility"] is not None
    assert set(result["correlation_matrix"]) == {"Equity", "Bond"}


def test_optimization_returns_weights():
    from app.quant import optimize_allocation

    result = optimize_allocation(
        {"VTI": [0.01, 0.02, -0.01, 0.015, 0.005], "BND": [0.002, 0.001, 0.003, -0.001, 0.002]},
        "max_sharpe",
        0.04,
    )
    assert set(result["weights"]) == {"VTI", "BND"}
    assert result["covariance_estimator"] == "LedoitWolf"


def test_risk_analysis_flags_concentration():
    from app.quant import risk_analysis

    result = risk_analysis(
        [
            {"ticker": "VTI", "asset_class": "Equity", "market_value": 800},
            {"ticker": "BND", "asset_class": "Bond", "market_value": 200},
        ]
    )
    assert result["concentration_flags"] == ["Equity"]
