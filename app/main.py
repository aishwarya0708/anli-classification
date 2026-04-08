from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = FastAPI(title="ANLI Classifier", version="1.0")

MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.eval()

LABELS = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

class InferenceRequest(BaseModel):
    premise: str
    hypothesis: str

@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/predict")
def predict(request: InferenceRequest):
    inputs = tokenizer(
        request.premise,
        request.hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    pred  = int(np.argmax(probs))
    return {
        "premise":      request.premise,
        "hypothesis":   request.hypothesis,
        "prediction":   LABELS[pred],
        "confidence":   round(float(probs[pred]), 4),
        "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(3)}
    }
