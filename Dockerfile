FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY scripts ./scripts

# Install PyTorch (CPU) and project + web dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[web]" \
    && python scripts/download_assets.py

# Create writable directories for checkpoints and user-recorded samples (Hugging Face Spaces compatible)
RUN mkdir -p /app/checkpoints /app/data && chmod -R 777 /app/checkpoints /app/data

EXPOSE 7860

CMD ["sh", "-c", "python scripts/run_web.py --host 0.0.0.0 --port ${PORT:-7860}"]
