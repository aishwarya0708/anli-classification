# Docker Deployment Guide

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed and running

## Build the Image
```bash
docker build -t anli-classifier .
```

## Run the Container
```bash
docker run -p 8000:8000 anli-classifier
```

## Test the API

**Health check:**
```bash
curl http://localhost:8000/
```

**Run inference:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "A man is playing guitar on a street corner.",
    "hypothesis": "A person is making music outdoors."
  }'
```

**Expected response:**
```json
{
  "premise": "A man is playing guitar on a street corner.",
  "hypothesis": "A person is making music outdoors.",
  "prediction": "Entailment",
  "confidence": 0.82,
  "probabilities": {
    "Entailment": 0.82,
    "Neutral": 0.12,
    "Contradiction": 0.06
  }
}
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Run NLI inference |
