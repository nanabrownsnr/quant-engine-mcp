"""Validated MCP input models for portfolio calculations."""

from pydantic import BaseModel, Field, model_validator


class Holding(BaseModel):
    ticker: str = Field(description="Ticker or security symbol.")
    asset_class: str = Field(description="Allocation bucket, such as Equity or Bond.")
    quantity: float | None = Field(default=None, ge=0, description="Number of units held.")
    current_price: float | None = Field(default=None, ge=0, description="Current price per unit.")
    market_value: float | None = Field(
        default=None, ge=0, description="Current position value; preferred when available."
    )
    cost_basis: float | None = Field(default=None, ge=0, description="Total cost basis in dollars.")
    account_type: str | None = Field(
        default=None, description="Taxable, Roth IRA, 401k, or other account type."
    )
    returns: list[float] | None = Field(
        default=None, description="Periodic decimal returns, such as 0.01 for 1%."
    )

    @model_validator(mode="after")
    def require_value_source(self):
        if self.market_value is None and (self.quantity is None or self.current_price is None):
            raise ValueError("Provide market_value or both quantity and current_price")
        return self


class AllocationRequest(BaseModel):
    current_holdings: list[Holding] = Field(description="Current portfolio positions.")
    target_allocation: dict[str, float] = Field(
        description="Asset classes mapped to decimal weights summing to 1.0."
    )

    @model_validator(mode="after")
    def validate_allocation(self):
        if abs(sum(self.target_allocation.values()) - 1.0) > 1e-6:
            raise ValueError("target_allocation must sum to 1.0")
        if any(value < 0 for value in self.target_allocation.values()):
            raise ValueError("target_allocation values cannot be negative")
        return self


class OptimizationRequest(BaseModel):
    historical_returns: dict[str, list[float]] = Field(
        description="Asset ticker mapped to aligned periodic decimal returns."
    )
    objective: str = Field(
        default="max_sharpe", description="Optimization objective: max_sharpe or min_volatility."
    )
    risk_free_rate: float = Field(
        default=0.04, ge=0, lt=1, description="Annual risk-free rate as a decimal."
    )

    @model_validator(mode="after")
    def validate_returns(self):
        if len(self.historical_returns) < 2:
            raise ValueError("At least two assets are required")
        lengths = {len(values) for values in self.historical_returns.values()}
        if len(lengths) != 1 or min(lengths) < 2:
            raise ValueError(
                "All assets need equally sized return series with at least two observations"
            )
        if self.objective not in {"max_sharpe", "min_volatility"}:
            raise ValueError("objective must be max_sharpe or min_volatility")
        return self
