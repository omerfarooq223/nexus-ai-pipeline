FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast; \
                   DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased'); \
                   DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
