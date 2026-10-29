# Minimal single-service image for the FastAPI backend.
# (torch pulls a large CUDA wheel by default; for a CPU-only demo add a CPU index.)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MODEL_PATH=/app/checkpoints/mwfull20_best.pth \
    VOCAB_FILE=/app/checkpoints/vocab.json
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
