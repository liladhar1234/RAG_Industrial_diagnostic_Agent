FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs: build tools, OpenCV runtime (needed by docling/unstructured),
# poppler + tesseract (PDF rendering and OCR for scanned pages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (avoids a ~2 GB CUDA download; enough for bge-small)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Source code is bind-mounted at runtime (see docker-compose.yml),
# so code edits don't require an image rebuild.
CMD ["bash"]
