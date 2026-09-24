"""FastMCP server for deterministic portfolio calculations."""

import asyncio
from contextlib import asynccontextmanager, suppress

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware as MCPMiddleware
from fastmcp.server.middleware import MiddlewareContext
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from app.config import settings
from app.license import license_watcher
from app.models import AllocationRequest, Holding, OptimizationRequest
from app.quant import calculate_drift, metrics, optimize_allocation, rebalance, risk_analysis
from app.twynity import register_routes
from app.usage import save_usage_report


@asynccontextmanager
async def lifespan(server):
    task = asyncio.create_task(license_watcher())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


mcp = FastMCP(settings.APP_TITLE, lifespan=lifespan)


@mcp.tool
def calculate_portfolio_drift(request: AllocationRequest) -> dict:
    """Compare holdings with target weights. Example: calculate_portfolio_drift([{asset_class: Equity, market_value: 600}, {asset_class: Bond, market_value: 400}], {Equity: 0.6, Bond: 0.4}). Holdings need asset_class plus market_value, or quantity and current_price. Target weights are decimals summing to 1. Returns current percentage, target percentage, percentage drift, and dollar drift."""
    return calculate_drift(
        [item.model_dump() for item in request.current_holdings], request.target_allocation
    )


@mcp.tool
def run_rebalance_plan(request: AllocationRequest, max_tax_impact: bool = True) -> dict:
    """Plan buys and sells without placing trades. Example: run_rebalance_plan([{asset_class: Equity, market_value: 800, account_type: Taxable, cost_basis: 600}, {asset_class: Bond, market_value: 200}], {Equity: 0.6, Bond: 0.4}). Set max_tax_impact=true to flag taxable sales with estimated gains."""
    return rebalance(
        [item.model_dump() for item in request.current_holdings],
        request.target_allocation,
        max_tax_impact,
    )


@mcp.tool
def compute_portfolio_metrics(current_holdings: list[Holding]) -> dict:
    """Compute annualized volatility, Sharpe ratio, and correlation. Add returns as decimal lists, e.g. returns: [0.01, -0.005, 0.02]. Example: compute_portfolio_metrics([{asset_class: Equity, market_value: 600, returns: [0.01, -0.005, 0.02]}]). Without returns, risk metrics are null rather than invented."""
    return metrics([item.model_dump() for item in current_holdings])


@mcp.tool
def optimize_target_allocation(request: OptimizationRequest) -> dict:
    """Optimize asset weights using historical returns.

    Use ``objective=\"max_sharpe\"`` for risk-adjusted return or
    ``objective=\"min_volatility\"`` for the lowest estimated volatility.
    Returns weights, expected annual return, volatility, Sharpe ratio, and
    the covariance estimator used. This is an analysis only; it never trades.
    """
    return optimize_allocation(
        request.historical_returns, request.objective, request.risk_free_rate
    )


@mcp.tool
def analyze_portfolio_risk(current_holdings: list[Holding]) -> dict:
    """Analyze concentration, correlations, and volatility contribution.

    Each holding may include aligned decimal ``returns``. Flags asset classes
    representing at least 50 percent of the portfolio as concentrated.
    """
    return risk_analysis([item.model_dump() for item in current_holdings])


class UsageTrackingMiddleware(MCPMiddleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        await save_usage_report("TOOL_CALL", context.message.name, None)
        return await call_next(context)


mcp.add_middleware(UsageTrackingMiddleware())
register_routes(mcp)
origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()] or [
    "*"
]

app = mcp.http_app(
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["mcp-session-id"],
        )
    ],
    transport="streamable-http",
    stateless_http=True,
    json_response=True,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
