FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY luxic/ ./luxic/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Pre-train & export INT8 edge artifact at build time (or mount pre-trained weights at runtime)
RUN python3 scripts/train_and_verify.py

ENV LUXIC_WEIGHTS_PATH=/app/artifacts/luxic_330m_v3_int8.npz
EXPOSE 8080

CMD ["uvicorn", "luxic.serving.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
