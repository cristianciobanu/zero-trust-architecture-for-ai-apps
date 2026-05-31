FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model directly into the image (no HuggingFace download at runtime)
COPY models/all-MiniLM-L6-v2 /app/models/all-MiniLM-L6-v2

COPY src/ ./

RUN useradd --uid 1000 --no-create-home --shell /bin/false appuser

EXPOSE 8000

USER 1000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
