# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock* ./
# Install into a project-local venv so it can be copied to the runtime stage
RUN poetry config virtualenvs.in-project true \
    && poetry install --without dev --no-root

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the pre-built virtualenv and source code
COPY --from=builder /app/.venv /app/.venv
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Ensure the venv is used for all commands
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default: run Streamlit frontend
# Override CMD in docker-compose to run pipeline or individual steps
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
