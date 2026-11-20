
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# --------------------------------------------------
# 1. Project paths and emotion labels
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "distilbert"

EMOTION_LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

DEVICE = torch.device("cpu")

# These will be loaded once and reused
tokenizer = None
model = None
id2label = None


# --------------------------------------------------
# 2. Load DistilBERT model and tokenizer
# --------------------------------------------------

def load_distilbert():
    global tokenizer, model, id2label

    if model is not None and tokenizer is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"DistilBERT model not found: {MODEL_PATH}"
        )

    print("Loading DistilBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )

    model.to(DEVICE)
    model.eval()

    # Read the emotion mapping from the model configuration
    id2label = {
        int(class_id): emotion.lower()
        for class_id, emotion in model.config.id2label.items()
    }

    if set(id2label.values()) != set(EMOTION_LABELS):
        raise ValueError(
            f"Unexpected model labels: {id2label}"
        )

    print("DistilBERT loaded successfully!")
    print("Device:", DEVICE)
    print("Labels:", id2label)


# --------------------------------------------------
# 3. Predict emotions for a list of messages
# --------------------------------------------------

def predict_emotion(messages, batch_size=32):

    if not isinstance(messages, list) or not messages:
        raise ValueError(
            "Please provide a non-empty list of messages."
        )

    if any(
        not isinstance(message, str) or not message.strip()
        for message in messages
    ):
        raise ValueError(
            "Every message must be a non-empty string."
        )

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    load_distilbert()

    results = []

    # Process messages in batches
    for start in range(0, len(messages), batch_size):

        batch_messages = messages[start:start + batch_size]

        # Tokenize the messages
        inputs = tokenizer(
            batch_messages,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # Run model inference
        with torch.no_grad():
            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

        # Convert predictions into the shared output format
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

            # Use the shared project label ID
            predicted_label_id = EMOTION_LABELS.index(
                predicted_emotion
            )

            results.append({
                "label_id": predicted_label_id,
                "label": predicted_emotion,
                "scores": scores
            })

    return results


# --------------------------------------------------
# 4. Test the prediction function
# --------------------------------------------------

if __name__ == "__main__":

    messages = [
        "I am so happy today!",
        "I feel very sad.",
        "I am afraid of what might happen."
    ]

    predictions = predict_emotion(messages)

    print("\nDISTILBERT PREDICTION RESULTS")
    print("----------------------------")

    for message, result in zip(messages, predictions):

        print("\nMessage:", message)
        print("Predicted emotion:", result["label"])
        print("Label ID:", result["label_id"])
        print("Scores:")

        for emotion, score in result["scores"].items():
            print(f"  {emotion}: {score:.4f}")
