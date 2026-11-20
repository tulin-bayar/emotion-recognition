
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data import load_emotion_data


def train_baseline():
    # 1. Load the shared dataset
    train_df, validation_df, test_df, label_names = load_emotion_data()

    print("\nDataset loaded successfully.")
    print("Training samples:", len(train_df))
    print("Validation samples:", len(validation_df))
    print("Test samples:", len(test_df))
    print("Emotions:", label_names)

    # 2. Prepare training and validation data
    X_train = train_df["text"]
    y_train = train_df["label"]

    X_validation = validation_df["text"]
    y_validation = validation_df["label"]

    # 3. Create TF-IDF + Logistic Regression pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=100000,
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    # 4. Train the model
    print("\nTraining baseline model...")
    model.fit(X_train, y_train)

    print("Training completed!")

    # 5. Evaluate on validation data
    validation_predictions = model.predict(X_validation)

    accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    macro_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="macro"
    )

    print("\nVALIDATION RESULTS")
    print("------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_validation,
            validation_predictions,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0
        )
    )

    # 6. Save the trained model and emotion labels
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "baseline_model.joblib"

    joblib.dump(
        {
            "model": model,
            "label_names": label_names
        },
        model_path
    )

    print("\nModel saved successfully!")
    print("Saved to:", model_path)


if __name__ == "__main__":
    train_baseline()
