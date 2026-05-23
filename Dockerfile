# Linux image with the full structure-aware parsing stack (hi-res layout model
# + Tesseract OCR + Poppler). Use this to parse PDFs without wrestling native
# deps on Windows.  Build/run via docker-compose 'parser' service.
FROM python:3.11-slim

# System deps: poppler (pdf rasterization), tesseract (OCR), libgl/glib (onnxruntime/opencv).
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better layer caching). README is referenced by pyproject.
COPY pyproject.toml README.md ./
COPY rag_pipeline ./rag_pipeline
RUN pip install --no-cache-dir ".[parse]"

# Default: run the parser/validation CLI. Override args at `docker compose run`.
ENTRYPOINT ["python", "-m", "rag_pipeline.ingestion.cli"]
CMD ["--help"]
