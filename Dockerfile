FROM python:3.11-slim

# System dependencies for OpenCV, EasyOCR, and PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package definition first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package (CPU-only torch to keep image size manageable)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .

# Pre-download EasyOCR English model so first run is fast
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)" || true

# Input/output volume mount points
VOLUME ["/data/input", "/data/output"]

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["docanon"]
CMD ["--help"]