FROM python:3.11-slim

WORKDIR /app
# creates app inside the container

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY formula_detector.py .
COPY index.html .
COPY results.html .
COPY floating-formulas.txt .

# .env placeholder that will be overridden by azure environment variables at runtime
RUN echo "Azure credentials will be set via environment variables" > .env

# expose port (azure container apps will use this)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
