
"""Baseline emotion prediction using TF-IDF and Logistic Regression."""

from pathlib import Path
import joblib

# --------------------------------------------------
# 1. Project paths and emotion labels
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"

EMOTION_LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

# --------------------------------------------------
# 2. Load trained baseline model
# --------------------------------------------------

_model = None


def load_baseline_model():
    """Load the trained baseline model once."""

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Baseline model not found: {MODEL_PATH}"
            )

        print("Loading baseline model...")

        saved_data = joblib.load(MODEL_PATH)

        _model = saved_data["model"]

        print("Baseline model loaded successfully!")

    return _model


# --------------------------------------------------
# 3. Predict emotions
# --------------------------------------------------

def predict_emotion(messages):
    """
    Predict emotions for a list of text messages.

    Returns a list of dictionaries containing:
    label_id, label, and scores for all six emotions.
    """

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

    model = load_baseline_model()

    probabilities = model.predict_proba(messages)

    classifier = model.named_steps["classifier"]
    class_ids = classifier.classes_

    results = []

    for probability_row in probabilities:

        scores = {
            emotion: 0.0
            for emotion in EMOTION_LABELS
        }

        for class_id, score in zip(class_ids, probability_row):
            emotion = EMOTION_LABELS[int(class_id)]
            scores[emotion] = float(score)

        predicted_class_id = int(
            class_ids[probability_row.argmax()]
        )

        predicted_emotion = EMOTION_LABELS[predicted_class_id]

        results.append({
            "label_id": predicted_class_id,
            "label": predicted_emotion,
            "scores": scores
        })

    return results


# --------------------------------------------------
# 4. Shared interface for app.py
# --------------------------------------------------

def predict_texts(texts):
    """Predict messages using the shared interface."""

    return predict_emotion(texts)


# --------------------------------------------------
# 5. Test the prediction function
# --------------------------------------------------

if __name__ == "__main__":

    test_messages = [
        "I am so happy today!",
        "I feel very sad.",
        "I am afraid of what might happen."
    ]

    predictions = predict_texts(test_messages)

    for message, prediction in zip(test_messages, predictions):

        print("\nMessage:", message)
        print("Predicted emotion:", prediction["label"])
        print("Label ID:", prediction["label_id"])
        print("Scores:", prediction["scores"])
