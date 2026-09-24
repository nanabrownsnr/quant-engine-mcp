"""FastMCP server for deterministic portfolio calculations."""

import asyncio
from contextlib import asynccontextmanager, suppress

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware as MCPMiddleware
from fastmcp.server.middleware import MiddlewareContext

from app.config import settings
from app.license import license_watcher
from app.quant import calculate_drift, metrics, rebalance
from app.routes import register_routes
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
def calculate_portfolio_drift(current_holdings: list[dict], target_allocation: dict[str, float]) -> dict:
    """Compare holdings with target weights. Example: calculate_portfolio_drift([{asset_class: Equity, market_value: 600}, {asset_class: Bond, market_value: 400}], {Equity: 0.6, Bond: 0.4}). Holdings need asset_class plus market_value, or quantity and current_price. Target weights are decimals summing to 1. Returns current percentage, target percentage, percentage drift, and dollar drift."""
    return calculate_drift(current_holdings, target_allocation)


@mcp.tool
def run_rebalance_plan(current_holdings: list[dict], target_allocation: dict[str, float], max_tax_impact: bool = True) -> dict:
    """Plan buys and sells without placing trades. Example: run_rebalance_plan([{asset_class: Equity, market_value: 800, account_type: Taxable, cost_basis: 600}, {asset_class: Bond, market_value: 200}], {Equity: 0.6, Bond: 0.4}). Set max_tax_impact=true to flag taxable sales with estimated gains."""
    return rebalance(current_holdings, target_allocation, max_tax_impact)


@mcp.tool
def compute_portfolio_metrics(current_holdings: list[dict]) -> dict:
    """Compute annualized volatility, Sharpe ratio, and correlation. Add returns as decimal lists, e.g. returns: [0.01, -0.005, 0.02]. Example: compute_portfolio_metrics([{asset_class: Equity, market_value: 600, returns: [0.01, -0.005, 0.02]}]). Without returns, risk metrics are null rather than invented."""
    return metrics(current_holdings)




class UsageTrackingMiddleware(MCPMiddleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        await save_usage_report("TOOL_CALL", context.message.name, None)
        return await call_next(context)

mcp.add_middleware(UsageTrackingMiddleware())
register_routes(mcp)
app = mcp.http_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
