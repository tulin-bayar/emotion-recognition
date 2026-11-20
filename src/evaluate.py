
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data import load_emotion_data
from src.predict_baseline import predict_emotion as predict_baseline
from src.predict_distilbert import predict_emotion as predict_distilbert


def evaluate_model(model_name, predictions, y_true, label_names, reports_dir):

    y_pred = [result["label_id"] for result in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{model_name} RESULTS")
    print("-" * (len(model_name) + 8))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(label_names)))
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues"
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(reports_dir / filename)

    print(f"\nConfusion matrix saved to reports/{filename}")

    plt.show()

    return accuracy, macro_f1


def main():

    # Load the shared dataset
    train_df, validation_df, test_df, label_names = load_emotion_data()

    # Prepare the same test data for both models
    test_texts = test_df["text"].tolist()
    y_true = test_df["label"].tolist()

    print("Test samples:", len(test_texts))
    print("Emotion labels:", label_names)

    # Create reports directory
    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate Baseline model
    print("\nEvaluating Baseline model...")
    baseline_predictions = predict_baseline(test_texts)

    baseline_accuracy, baseline_f1 = evaluate_model(
        "Baseline Model",
        baseline_predictions,
        y_true,
        label_names,
        reports_dir
    )

    # Evaluate DistilBERT model
    print("\nEvaluating DistilBERT model...")
    distilbert_predictions = predict_distilbert(test_texts)

    distilbert_accuracy, distilbert_f1 = evaluate_model(
        "DistilBERT Model",
        distilbert_predictions,
        y_true,
        label_names,
        reports_dir
    )

    # Compare both models
    print("\nMODEL COMPARISON")
    print("----------------")

    print(f"{'Metric':<15}{'Baseline':<15}{'DistilBERT':<15}")
    print("-" * 45)
    print(f"{'Accuracy':<15}{baseline_accuracy:<15.4f}{distilbert_accuracy:<15.4f}")
    print(f"{'Macro F1':<15}{baseline_f1:<15.4f}{distilbert_f1:<15.4f}")


if __name__ == "__main__":
    main()
