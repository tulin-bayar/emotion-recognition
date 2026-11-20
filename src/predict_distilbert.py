"""DistilBERT emotion predictor.

predict_texts(texts) takes a list of messages and returns a list of results
in the same order. Each result is:
    {"label_id": int, "label": str, "scores": {emotion: probability, ...}}
"""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert"
MAX_LENGTH = 128
BATCH_SIZE = 32
REQUIRED_FILES = ["config.json", "model.safetensors", "vocab.txt", "tokenizer_config.json"]

_cache = {}


def _load():
    """Load tokenizer and model once and reuse them on later calls."""
    if "model" in _cache:
        return _cache["tokenizer"], _cache["model"], _cache["device"]

    missing = [f for f in REQUIRED_FILES if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"DistilBERT model files not found in '{MODEL_DIR}'. "
            f"Missing: {', '.join(missing)}. "
            "Put the 'distilbert' folder from the shared ZIP into models/ "
            "(it must be models/distilbert/, not models/distilbert/distilbert/)."
        )

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    _cache.update(tokenizer=tokenizer, model=model, device=device)
    return tokenizer, model, device


def predict_texts(texts):
    """Predict the emotion of each message. Returns results in the same order."""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    if not texts:
        return []

    import torch

    tokenizer, model, device = _load()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    results = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        for row in probs:
            label_id = max(range(len(row)), key=lambda i: row[i])
            results.append({
                "label_id": label_id,
                "label": id2label[label_id],
                "scores": {id2label[i]: float(p) for i, p in enumerate(row)},
            })
    return results
