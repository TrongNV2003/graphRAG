FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY main.py .
COPY run.py .
COPY src/ src/

# Create necessary directories
RUN mkdir -p data/hybrid_index data/entities_extracted data/format_data data/raw_data

# Install dependencies
RUN uv sync --frozen

# Expose API port
EXPOSE 8000

# Run API
CMD ["uv", "run", "python", "run.py"]
