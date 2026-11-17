FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# CPU-only PyTorch keeps the image small; then the remaining runtime dependencies.
COPY requirements-docker.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-docker.txt

# Code and notes go into the image. Trained models (artifacts/) and the feature table
# (data/processed/) are mounted at run time, so the image has no data or secrets in it.
COPY src ./src
COPY notes ./notes

RUN useradd --create-home appuser && mkdir -p /app/outbox && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]