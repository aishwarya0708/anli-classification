## Pipeline Steps
1. Exploratory Data Analysis — label distribution, length analysis, hard examples
2. Baseline — TF-IDF + Logistic Regression
3. Transformer — RoBERTa-base fine-tuned on ANLI R2
4. Evaluation — Accuracy, Macro F1, Confusion Matrix
5. Error Analysis — failure mode inspection

## Setup & Run

### Local
```bash
pip install -r requirements.txt
```

### Docker
```bash
docker build -t anli-classifier .
docker run -p 8000:8000 anli-classifier
```

### Inference
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"premise": "A man is playing guitar.", "hypothesis": "A person is making music."}'
```

## Tech Stack
HuggingFace Transformers · PyTorch · scikit-learn · FastAPI · Docker
