
"""DistilBERT emotion predictor.

Both predict_emotion() and predict_texts() are supported.
Input: a list of messages.
Output: predictions in the same order.
"""

from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert"

EMOTION_LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

MAX_LENGTH = 128
BATCH_SIZE = 32

REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "vocab.txt",
    "tokenizer_config.json"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_cache = {}


def load_distilbert():
    """Load the tokenizer and model once."""

    if "model" in _cache:
        return _cache["tokenizer"], _cache["model"], _cache["id2label"]

    missing = [
        file for file in REQUIRED_FILES
        if not (MODEL_DIR / file).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing DistilBERT files in {MODEL_DIR}: {missing}"
        )

    print("Loading DistilBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.to(DEVICE)
    model.eval()

    id2label = {
        int(class_id): emotion.lower()
        for class_id, emotion in model.config.id2label.items()
    }

    if set(id2label.values()) != set(EMOTION_LABELS):
        raise ValueError(
            f"Unexpected model labels: {id2label}"
        )

    _cache.update({
        "tokenizer": tokenizer,
        "model": model,
        "id2label": id2label
    })

    print("DistilBERT loaded successfully!")
    print("Device:", DEVICE)
    print("Labels:", id2label)

    return tokenizer, model, id2label


def predict_emotion(messages, batch_size=BATCH_SIZE):
    """Predict emotions for a list of messages."""

    if isinstance(messages, str):
        messages = [messages]

    if not isinstance(messages, list):
        raise ValueError("Input must be a list of messages.")

    if not messages:
        return []

    if any(
        not isinstance(message, str) or not message.strip()
        for message in messages
    ):
        raise ValueError(
            "Every message must be a non-empty string."
        )

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    tokenizer, model, id2label = load_distilbert()

    results = []

    for start in range(0, len(messages), batch_size):

        batch_messages = messages[start:start + batch_size]

        inputs = tokenizer(
            batch_messages,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

        for probability_row in probabilities:

            scores = {
                emotion: 0.0
                for emotion in EMOTION_LABELS
            }

            for output_id, score in enumerate(
                probability_row.tolist()
            ):
                emotion = id2label[output_id]
                scores[emotion] = float(score)

            predicted_output_id = int(
                torch.argmax(probability_row).item()
            )

            predicted_emotion = id2label[predicted_output_id]

            predicted_label_id = EMOTION_LABELS.index(
                predicted_emotion
            )

            results.append({
                "label_id": predicted_label_id,
                "label": predicted_emotion,
                "scores": scores
            })

    return results


def predict_texts(texts):
    """Alternative function name for compatibility."""

    return predict_emotion(texts)


if __name__ == "__main__":

    messages = [
        "I am so happy today!",
        "I feel very sad.",
        "I am afraid of what might happen."
    ]

    predictions = predict_emotion(messages)

    for message, result in zip(messages, predictions):
        print("\nMessage:", message)
        print("Prediction:", result)
