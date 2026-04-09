FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# faster-whisper: CTranslate2-based Whisper, uses int8 quantization for minimal memory
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    faster-whisper

# Create app directory
WORKDIR /app

# Copy server script
COPY server.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget -q --method=GET -O /dev/null http://localhost:8000/health || exit 1

CMD ["python", "server.py"]