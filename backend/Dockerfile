FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libmagic1 \
    file \
    curl \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
COPY backend/pyproject.toml ./backend/
COPY rag/pyproject.toml ./rag/
COPY frontend/pyproject.toml ./frontend/

COPY rag/src ./rag/src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project --package backend

COPY backend/src ./backend/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package backend

ENV PATH="/app/.venv/bin:$PATH"

