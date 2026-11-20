
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"


def load_baseline_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Baseline model not found. Run train_baseline.py first."
        )

    saved_data = joblib.load(MODEL_PATH)
    return saved_data["model"], saved_data["label_names"]


def predict_emotion(messages):
    if not isinstance(messages, list):
        raise TypeError("Input must be a list of messages.")

    if not messages:
        raise ValueError("Message list cannot be empty.")

    if any(not isinstance(message, str) or not message.strip()
           for message in messages):
        raise ValueError("Every message must be a non-empty string.")

    model, label_names = load_baseline_model()

    predicted_ids = model.predict(messages)
    probabilities = model.predict_proba(messages)

    # Get the actual class IDs corresponding to probability columns
    classifier = model.named_steps["classifier"]
    class_ids = classifier.classes_

    results = []

    for predicted_id, probability_row in zip(
        predicted_ids, probabilities
    ):
        predicted_id = int(predicted_id)

        # Map each probability to its correct emotion label
        probability_by_class = dict(zip(class_ids, probability_row))

        scores = {
            emotion: float(
                probability_by_class.get(class_id, 0.0)
            )
            for class_id, emotion in enumerate(label_names)
        }

        results.append({
            "label_id": predicted_id,
            "label": label_names[predicted_id],
            "scores": scores
        })

    return results


if __name__ == "__main__":
    messages = [
        "I am so happy today!",
        "I feel very sad.",
        "I am afraid of what might happen."
    ]

    results = predict_emotion(messages)

    for message, result in zip(messages, results):
        print("\nMessage:", message)
        print("Predicted emotion:", result["label"])
        print("Label ID:", result["label_id"])
        print("Scores:", result["scores"])
