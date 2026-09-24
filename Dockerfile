FROM python:3.11-slim
# Builds the MCP App and production Python runtime. Update the UI COPY paths
# when renaming the example; generated dist/ files stay out of source control.

COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml .
RUN uv sync --no-dev

COPY app/ ./app/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
