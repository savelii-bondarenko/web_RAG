FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY backend/pyproject.toml ./backend/
COPY rag/pyproject.toml ./rag/
COPY frontend/pyproject.toml ./frontend/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --package frontend

COPY frontend/src ./frontend/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package frontend

ENV PATH="/app/.venv/bin:$PATH"
