# Quant Engine MCP

Deterministic portfolio allocation and risk calculations exposed through Streamable HTTP.

## Tools

- `calculate_portfolio_drift(request)` compares current and target asset-class weights. The request contains `current_holdings` and `target_allocation`.
- `run_rebalance_plan(request, max_tax_impact=true)` returns buy/sell dollar amounts and tax-impact flags. The request contains `current_holdings` and `target_allocation`. It never places trades.
- `compute_portfolio_metrics(current_holdings)` calculates volatility, Sharpe ratio, and correlation from supplied return series.

Holdings use `market_value`, or `quantity` and `current_price`. Each holding requires `ticker` and `asset_class`; optional fields include `cost_basis`, `account_type`, and `returns`. Asset classes use values such as `Equity` and `Bond`. Target allocations are decimal weights summing to `1.0`.

## Run

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

MCP endpoint: `http://localhost:8000/mcp`
Health endpoint: `http://localhost:8000/api/v1/health`

This service performs local calculations and does not require an API key. Optional Twynity usage and license settings can be supplied through `.env`.

## Docker

```bash
docker build -t quant-engine-mcp:local .
docker run --rm -p 8000:8000 quant-engine-mcp:local
```
