# Multi-stage build for smaller final image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in user directory
COPY requirements.txt /tmp/requirements.txt
RUN pip install --user --no-cache-dir --no-warn-script-location \
    -r /tmp/requirements.txt

# Final stage - only runtime dependencies
FROM python:3.10-slim

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/app/.local

# Set up working directory
WORKDIR /app
COPY --chown=app:app . .

# Set environment
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]