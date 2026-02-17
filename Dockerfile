# Use the official uv image for dependency resolution
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies into a portable location
RUN uv sync --no-install-project --no-dev

# Copy the source code
COPY src/ ./src/
COPY data_volume/ ./data_volume/
COPY app.py ./
COPY chainlit.md ./
COPY langgraph.json ./
COPY public/ ./public/
COPY .chainlit/ ./.chainlit/

# Install the project itself
RUN uv sync --no-dev

# Final Stage
FROM python:3.13-slim-bookworm

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment and application from the builder
COPY --from=builder /app /app

# Ensure we use the virtualenv created by uv
ENV PATH="/app/.venv/bin:$PATH"

# Chainlit specific environment variables
ENV CHAINLIT_HOST="0.0.0.0"
ENV PYTHONUNBUFFERED=1

# Cloud Run provides a $PORT environment variable. 
# We use a shell form for CMD to allow environment variable expansion.
CMD ["sh", "-c", "chainlit run app.py --host 0.0.0.0 --port ${PORT:-8080}"]
